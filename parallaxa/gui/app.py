"""Start the Parallaxa desktop app:  parallaxa gui   (or  parallaxa-gui)."""

from __future__ import annotations

import os
import sys


def main(argv: list[str] | None = None) -> int:
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    from PySide6.QtWidgets import QApplication

    from .main_window import MainWindow

    if sys.platform == "win32":
        # Own taskbar identity, so Windows shows the Parallaxa icon instead of Python's.
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("Parallaxa.App")
    from pathlib import Path

    from PySide6.QtGui import QIcon

    app = QApplication.instance() or QApplication(sys.argv if argv is None else ["parallaxa", *argv])
    app.setApplicationName("Parallaxa")
    app.setOrganizationName("Parallaxa")
    app.setStyle("Fusion")
    app.setWindowIcon(QIcon(str(Path(__file__).resolve().parents[1] / "assets" / "logo-512.png")))
    win = MainWindow(offer_downloads=True)
    files = [a for a in (sys.argv[1:] if argv is None else argv) if not a.startswith("-")]
    if files:
        win.add_paths(files)
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
