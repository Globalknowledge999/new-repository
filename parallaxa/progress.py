"""Progress / ETA reporting (human-readable on stderr, or JSON lines on stdout for a GUI)."""

from __future__ import annotations

import json
import sys
import time


def fmt_time(seconds: float) -> str:
    if seconds != seconds or seconds == float("inf"):
        return "--:--"
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


class Progress:
    def __init__(self, total: int, label: str, mode: str = "text", unit: str = "frames", min_interval: float = 0.5):
        self.total, self.label, self.mode, self.unit = total, label, mode, unit
        self.done = 0
        self.start = time.perf_counter()
        self._last = 0.0
        self._emitted_at = -1
        self.min_interval = min_interval

    def update(self, n: int = 1) -> None:
        self.done += n
        now = time.perf_counter()
        if now - self._last >= self.min_interval or (self.total and self.done >= self.total):
            self._last = now
            self._emit()

    @property
    def rate(self) -> float:
        el = time.perf_counter() - self.start
        return self.done / el if el > 0 else 0.0

    @property
    def eta(self) -> float:
        r = self.rate
        return (max(self.total - self.done, 0) / r) if (r > 0 and self.total) else float("nan")

    def _emit(self) -> None:
        self._emitted_at = self.done
        if self.mode == "json":
            print(json.dumps({"event": "progress", "file": self.label, "done": self.done, "total": self.total,
                              "rate": round(self.rate, 3), "eta_s": None if self.eta != self.eta else round(self.eta, 1)}), flush=True)
            return
        if self.mode == "none":
            return
        pct = f"{100.0 * self.done / self.total:5.1f}%" if self.total else "  ?  "
        msg = f"\r  {pct}  {self.done}/{self.total or '?'} {self.unit}  {self.rate:6.2f}/s  ETA {fmt_time(self.eta)}  elapsed {fmt_time(time.perf_counter() - self.start)}"
        sys.stderr.write(msg)
        sys.stderr.flush()

    def finish(self) -> None:
        if self._emitted_at != self.done:
            self._emit()
        if self.mode == "text":
            sys.stderr.write("\n")
            sys.stderr.flush()
