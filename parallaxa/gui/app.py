"""Start the Parallaxa desktop app:  parallaxa gui   (or  parallaxa-gui)."""

from __future__ import annotations

import os
import sys


def main(argv: list[str] | None = None) -> int:
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    from PySide6.QtWidgets import QApplication

    from .main_window import MainWindow

    app = QApplication.instance() or QApplication(sys.argv if argv is None else ["parallaxa", *argv])
    app.setApplicationName("Parallaxa")
    app.setOrganizationName("Parallaxa")
    app.setStyle("Fusion")
    win = MainWindow(offer_downloads=True)
    files = [a for a in (sys.argv[1:] if argv is None else argv) if not a.startswith("-")]
    if files:
        win.add_paths(files)
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
