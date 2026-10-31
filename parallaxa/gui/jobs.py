"""Batch queue: runs each conversion as a `parallaxa convert` child process and tracks its JSON events.

A child process keeps the window responsive and isolates crashes and GPU out-of-memory errors.
Stop asks the child to cancel cleanly, as Ctrl+C would, so a video keeps its finished segments
and Start resumes it.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, QTimer, Signal

from ..pipeline.common import collect_inputs
from .options import ConvertOptions

QUEUED, RUNNING, STOPPING, DONE, SKIPPED, FAILED, STOPPED = "Queued", "Running", "Stopping", "Done", "Already done", "Failed", "Stopped"
FINISHED = (DONE, SKIPPED, FAILED)


@dataclass
class Job:
    path: Path
    kind: str  # photo | video
    status: str = QUEUED
    progress: float = 0.0  # 0..1, or -1 while it can't be measured (model loading, photos)
    done: int = 0
    total: int = 0
    rate: float = 0.0
    eta: float | None = None
    message: str = ""
    outputs: list[str] = field(default_factory=list)
    error: str = ""
    overwrite: bool = False  # set by "Convert again": replace existing outputs


def python_executable() -> str:
    """The console interpreter (pythonw.exe has no stdout to report progress on)."""
    exe = Path(sys.executable)
    if exe.name.lower() == "pythonw.exe" and (exe.parent / "python.exe").exists():
        return str(exe.parent / "python.exe")
    return str(exe)


class QueueRunner(QObject):
    job_changed = Signal(int)  # row
    jobs_reset = Signal()  # rows added / removed
    log = Signal(str)
    running_changed = Signal(bool)

    def __init__(self, parent=None, kill_after_ms: int = 20000):
        super().__init__(parent)
        self.jobs: list[Job] = []
        self.proc: QProcess | None = None
        self.current: int | None = None
        self.options = ConvertOptions()
        self._stop_requested = False
        self._buffer = b""
        self._stderr_tail: list[str] = []
        self._kill_timer = QTimer(self, singleShot=True, interval=kill_after_ms)
        self._kill_timer.timeout.connect(self._force_kill)

    # -- queue editing ----------------------------------------------------------------------
    def add(self, paths: list[str], recursive: bool = True) -> int:
        known = {j.path.resolve() for j in self.jobs if j.status not in FINISHED}
        added = 0
        for path, kind in collect_inputs(paths, recursive):
            if path.resolve() not in known:
                self.jobs.append(Job(path, kind))
                known.add(path.resolve())
                added += 1
        if added:
            self.jobs_reset.emit()
        return added

    def remove(self, rows: list[int]) -> None:
        for r in sorted(set(rows), reverse=True):
            if r != self.current and 0 <= r < len(self.jobs):
                del self.jobs[r]
                if self.current is not None and r < self.current:
                    self.current -= 1
        self.jobs_reset.emit()

    def clear_finished(self) -> None:
        self.remove([i for i, j in enumerate(self.jobs) if j.status in FINISHED])

    @property
    def running(self) -> bool:
        return self.proc is not None

    # -- running ----------------------------------------------------------------------------
    def start(self, options: ConvertOptions) -> None:
        """Run queued (and previously stopped) jobs one after another with these options."""
        if self.running:
            return
        self.options = options
        self._stop_requested = False
        for i, j in enumerate(self.jobs):
            if j.status == STOPPED:
                j.status = QUEUED
                self.job_changed.emit(i)
        self._next()

    def stop(self) -> None:
        self._stop_requested = True
        if self.proc is not None and self.current is not None:
            self.jobs[self.current].status = STOPPING
            self.job_changed.emit(self.current)
            self.proc.write(b"cancel\n")
            self._kill_timer.start()

    def _next(self) -> None:
        nxt = next((i for i, j in enumerate(self.jobs) if j.status == QUEUED), None)
        if nxt is None or self._stop_requested:
            self.current = None
            self.running_changed.emit(False)
            return
        self.current = nxt
        job = self.jobs[nxt]
        job.status, job.progress, job.message, job.error, job.eta = RUNNING, -1, "Starting...", "", None
        self.job_changed.emit(nxt)

        proc = QProcess(self)
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUTF8", "1")
        env.insert("PYTHONIOENCODING", "utf-8")
        env.insert("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        proc.setProcessEnvironment(env)
        proc.readyReadStandardOutput.connect(self._read_stdout)
        proc.readyReadStandardError.connect(self._read_stderr)
        proc.finished.connect(self._finished)
        self.proc, self._buffer, self._stderr_tail = proc, b"", []
        args = ["-m", "parallaxa", "convert", str(job.path), *self.options.to_args(), "--progress", "json", "--control-stdin"]
        if job.overwrite:
            args.append("--overwrite")
        self.log.emit(f"> parallaxa convert \"{job.path}\" {' '.join(self.options.to_args())}")
        proc.start(python_executable(), args)
        self.running_changed.emit(True)

    def _read_stdout(self) -> None:
        self._buffer += bytes(self.proc.readAllStandardOutput())
        *lines, self._buffer = self._buffer.split(b"\n")
        for line in lines:
            try:
                ev = json.loads(line.decode("utf-8", errors="replace"))
            except ValueError:
                continue
            self._handle(ev)

    def _read_stderr(self) -> None:
        text = bytes(self.proc.readAllStandardError()).decode("utf-8", errors="replace")
        for line in text.splitlines():
            if line.strip():
                self._stderr_tail = (self._stderr_tail + [line])[-20:]

    def _handle(self, ev: dict) -> None:
        if self.current is None:
            return
        job = self.jobs[self.current]
        kind = ev.get("event")
        if kind == "progress":
            job.done, job.total = int(ev.get("done", 0)), int(ev.get("total", 0) or 0)
            job.progress = job.done / job.total if job.total else -1
            job.rate, job.eta = float(ev.get("rate") or 0), ev.get("eta_s")
            job.message = f"{job.done}/{job.total or '?'} frames, {job.rate:.1f}/s"
        elif kind == "log":
            msg = ev.get("message", "")
            self.log.emit(msg)
            if job.status == RUNNING and not msg.startswith("timing"):
                job.message = msg.split(" | ")[0][:120]
        elif kind == "file_done":
            job.outputs = ev.get("outputs", [])
            job.status = SKIPPED if ev.get("skipped") else DONE
            job.progress, job.eta = 1.0, None
            job.message = ", ".join(Path(p).name for p in job.outputs) or "Output already exists"
        elif kind == "file_failed":
            job.status, job.error, job.message = FAILED, ev.get("error", ""), ev.get("error", "")
        elif kind == "cancelled":
            job.status, job.message = STOPPED, "Stopped (videos resume from the last finished segment)"
        self.job_changed.emit(self.current)

    def _finished(self, code: int, _status) -> None:
        self._kill_timer.stop()
        self._read_stdout()
        if self.current is not None:
            job = self.jobs[self.current]
            if job.status in (RUNNING, STOPPING):
                if self._stop_requested:
                    job.status, job.message = STOPPED, "Stopped"
                elif code == 0:
                    job.status = DONE
                else:
                    job.status = FAILED
                    job.error = job.message = "\n".join(self._stderr_tail[-5:]) or f"exited with code {code}"
            self.job_changed.emit(self.current)
            if job.status == FAILED and self._stderr_tail:
                self.log.emit("\n".join(self._stderr_tail))
        self.proc.deleteLater()
        self.proc = None
        self._next()

    def _force_kill(self) -> None:
        if self.proc is not None:
            self.log.emit("Conversion did not stop in time; terminating it.")
            self.proc.kill()

    def shutdown(self, timeout_ms: int = 15000) -> None:
        """Stop cleanly when the window closes (keeps resumable segments)."""
        if self.proc is not None:
            self.stop()
            if not self.proc.waitForFinished(timeout_ms):
                self.proc.kill()
                self.proc.waitForFinished(3000)
