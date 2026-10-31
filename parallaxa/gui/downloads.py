"""First-launch model download with a progress window (runs `parallaxa download` as a child process)."""

from __future__ import annotations

import json

from PySide6.QtCore import QProcess, QProcessEnvironment, Qt
from PySide6.QtWidgets import QMessageBox, QProgressDialog, QWidget

from ..downloads import DOWNLOADS
from .jobs import python_executable


class ModelDownloadDialog(QProgressDialog):
    def __init__(self, names: list[str], parent: QWidget | None = None):
        total_mb = sum(DOWNLOADS[n][1] for n in names)
        super().__init__(f"Preparing to download {len(names)} model(s), about {total_mb / 1000:.1f} GB...", "Cancel", 0, len(names), parent)
        self.setWindowTitle("Downloading models")
        self.setWindowModality(Qt.WindowModality.WindowModal)
        self.setMinimumWidth(460)
        self.setMinimumDuration(0)
        self.setAutoClose(False)
        self.setAutoReset(False)
        self.names, self.completed, self.failed = names, [], []
        self.proc = QProcess(self)
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUTF8", "1")
        env.insert("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        self.proc.setProcessEnvironment(env)
        self.proc.readyReadStandardOutput.connect(self._read)
        self.proc.finished.connect(self._finished)
        self.canceled.connect(self.proc.kill)
        self._buf = b""

    def start(self) -> None:
        self.proc.start(python_executable(), ["-m", "parallaxa", "download", *self.names, "--progress", "json"])

    def _read(self) -> None:
        self._buf += bytes(self.proc.readAllStandardOutput())
        *lines, self._buf = self._buf.split(b"\n")
        for line in lines:
            try:
                ev = json.loads(line)
            except ValueError:
                continue
            if ev.get("event") == "start":
                self.setLabelText(f"Downloading {ev['index']} of {ev['total']}: {ev['description']}\n"
                                  f"(about {ev['mb']} MB; this only happens once)")
                self.setValue(ev["index"] - 1)
            elif ev.get("event") == "done":
                self.completed.append(ev["name"])
            elif ev.get("event") == "failed":
                self.failed.append(f"{ev['name']}: {ev['error']}")

    def _finished(self, *_):
        self._read()
        self.setValue(self.maximum())
        if self.wasCanceled():
            self.setLabelText("Cancelled. Models will download when first needed.")
        elif self.failed:
            self.setLabelText("Some downloads failed (check your internet connection):\n" + "\n".join(self.failed))
        else:
            self.setLabelText("Models are ready.")
        self.setCancelButtonText("Close")
        self.canceled.disconnect()
        self.canceled.connect(self.close)


def offer_first_run_download(parent: QWidget, names: list[str]) -> ModelDownloadDialog | None:
    """Ask once whether to fetch the essential models now; returns the running dialog if accepted."""
    mb = sum(DOWNLOADS[n][1] for n in names)
    answer = QMessageBox.question(
        parent, "Download depth models",
        f"Parallaxa needs its depth models before the first conversion (about {mb / 1000:.1f} GB, downloaded once).\n\n"
        "Download them now? (If you choose No, they download automatically when first needed.)")
    if answer != QMessageBox.StandardButton.Yes:
        return None
    dlg = ModelDownloadDialog(names, parent)
    dlg.start()
    dlg.show()
    return dlg
