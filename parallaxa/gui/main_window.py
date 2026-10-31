"""Parallaxa main window: batch queue on the left, live preview and settings on the right."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QThread, QTimer, QUrl, Signal
from PySide6.QtGui import QAction, QDesktopServices, QImage, QKeySequence, QPixmap
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDockWidget, QDoubleSpinBox, QFileDialog,
                               QFormLayout, QFrame, QGridLayout, QGroupBox, QHBoxLayout, QHeaderView, QInputDialog,
                               QLabel, QLineEdit, QMainWindow, QMenu, QMessageBox, QPlainTextEdit, QProgressBar,
                               QPushButton, QScrollArea, QSizePolicy, QSlider, QSplitter, QStackedWidget, QTableWidget,
                               QTableWidgetItem, QToolBar, QVBoxLayout, QWidget)

from .. import __version__
from ..config import PHOTO_EXTS, VIDEO_EXTS
from ..progress import fmt_time
from .jobs import DONE, FAILED, FINISHED, RUNNING, SKIPPED, STOPPED, QueueRunner
from .options import ConvertOptions, PresetStore
from .preview import MODES, PreviewWorker

FORMAT_LABELS = [
    ("sbs", "Full side-by-side", "VR headsets, 3D players: each eye at full resolution"),
    ("hsbs", "Half side-by-side", "3D TVs and projectors"),
    ("tb", "Full top-bottom", "Players that prefer over/under"),
    ("htb", "Half top-bottom", "3D TVs"),
    ("anaglyph", "Red/cyan anaglyph", "Any screen with paper 3D glasses"),
    ("rgbd", "RGB + depth", "Colour and depth side by side"),
    ("depth", "Depth map (photos)", "16-bit greyscale PNG, white = near"),
    ("mpo", "MPO stereo file (photos)", "3D cameras, Nintendo 3DS, StereoPhoto Maker"),
]
PROFILE_LABELS = [
    ("auto", "Automatic (best for photos, fast for video)"),
    ("fast", "Fast: DA-V2 Small, per frame"),
    ("balanced", "Steady video: Video Depth Anything Small"),
    ("quality", "Steady video, sharper (518 px)"),
    ("photo-quality", "Most detailed: DA-V2 Large"),
]
HEIGHTS = [(0, "Same as source"), (2160, "2160p (4K)"), (1440, "1440p"), (1080, "1080p"), (720, "720p")]
COLUMNS = ["File", "Type", "Status", "Progress", "Time left", "Details"]


def to_qimage(arr: np.ndarray) -> QImage:
    h, w, _ = arr.shape
    return QImage(arr.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()


class PreviewLabel(QLabel):
    """Scales its pixmap to fit while keeping the aspect ratio; alternates two frames in wobble mode."""

    def __init__(self):
        super().__init__("Select a file in the queue to preview it in 3D")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(320, 200)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet("QLabel { background: #15171c; color: #9aa3b2; border-radius: 6px; }")
        self._frames: list[QPixmap] = []
        self._i = 0
        self._timer = QTimer(self, interval=140)
        self._timer.timeout.connect(self._tick)

    def set_images(self, images: list[np.ndarray]) -> None:
        self._frames = [QPixmap.fromImage(to_qimage(a)) for a in images]
        self._i = 0
        self._timer.start() if len(self._frames) > 1 else self._timer.stop()
        self._show()

    def clear_preview(self, text: str) -> None:
        self._frames, self._i = [], 0
        self._timer.stop()
        self.setPixmap(QPixmap())
        self.setText(text)

    def _tick(self) -> None:
        self._i = (self._i + 1) % len(self._frames)
        self._show()

    def _show(self) -> None:
        if self._frames:
            self.setPixmap(self._frames[self._i].scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                                        Qt.TransformationMode.SmoothTransformation))

    def resizeEvent(self, e) -> None:
        super().resizeEvent(e)
        self._show()


class SliderRow(QWidget):
    """A slider with an editable number next to it."""

    changed = Signal(float)

    def __init__(self, lo: float, hi: float, step: float, decimals: int, suffix: str = ""):
        super().__init__()
        self.step = step
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(int(round(lo / step)), int(round(hi / step)))
        self.spin = QDoubleSpinBox()
        self.spin.setRange(lo, hi)
        self.spin.setSingleStep(step)
        self.spin.setDecimals(decimals)
        self.spin.setSuffix(suffix)
        self.spin.setFixedWidth(84)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.slider, 1)
        lay.addWidget(self.spin)
        self.slider.valueChanged.connect(lambda v: self._set(v * self.step, from_slider=True))
        self.spin.valueChanged.connect(lambda v: self._set(v, from_slider=False))

    def _set(self, v: float, from_slider: bool) -> None:
        widget = self.spin if from_slider else self.slider
        widget.blockSignals(True)
        self.spin.setValue(v) if from_slider else self.slider.setValue(int(round(v / self.step)))
        widget.blockSignals(False)
        self.changed.emit(self.value())

    def value(self) -> float:
        return round(self.spin.value(), 4)

    def setValue(self, v: float) -> None:
        self.blockSignals(True)
        self.spin.setValue(v)
        self.slider.blockSignals(True)
        self.slider.setValue(int(round(v / self.step)))
        self.slider.blockSignals(False)
        self.blockSignals(False)


class MainWindow(QMainWindow):
    preview_requested = Signal(object)

    def __init__(self, presets: PresetStore | None = None, preview_device: str = "auto"):
        super().__init__()
        self.setWindowTitle(f"Parallaxa {__version__}: 2D to 3D")
        self.resize(1360, 820)
        self.setAcceptDrops(True)
        self.presets = presets or PresetStore()
        self.runner = QueueRunner(self)
        self._loading_settings = True  # widgets fire change signals while being built
        self._advanced: dict = {}

        self._build_toolbar()
        self._build_central()
        self._build_log()
        self._build_preview_thread(preview_device)

        self.runner.jobs_reset.connect(self._rebuild_table)
        self.runner.job_changed.connect(self._update_row)
        self.runner.log.connect(self._append_log)
        self.runner.running_changed.connect(self._running_changed)

        self._render_timer = QTimer(self, singleShot=True, interval=120)
        self._render_timer.timeout.connect(self._request_preview)

        self.apply_options(self.presets.load_last())
        self._running_changed(False)
        self.statusBar().showMessage("Drop photos, videos or folders onto the window to start.")

    # -- layout -----------------------------------------------------------------------------
    def _build_toolbar(self) -> None:
        tb = QToolBar("Main")
        tb.setMovable(False)
        tb.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self.addToolBar(tb)

        def act(text, slot, shortcut=None, tip=""):
            a = QAction(text, self)
            a.triggered.connect(slot)
            if shortcut:
                a.setShortcut(QKeySequence(shortcut))
            a.setToolTip(tip or text)
            tb.addAction(a)
            return a

        act("Add files…", self.add_files_dialog, "Ctrl+O")
        act("Add folder…", self.add_folder_dialog, "Ctrl+Shift+O")
        tb.addSeparator()
        self.start_action = act("▶  Start", self.start, "Ctrl+Return", "Convert everything queued")
        self.stop_action = act("■  Stop", self.runner.stop, None,
                               "Stop the current conversion. Videos resume from the last finished segment.")
        tb.addSeparator()
        act("Clear finished", self.runner.clear_finished)
        act("Open output folder", self.open_output_folder)

    def _build_central(self) -> None:
        split = QSplitter(Qt.Orientation.Horizontal)

        # Left: queue (or a drop hint while empty)
        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.setAcceptDrops(False)
        self.table.setMinimumHeight(100)
        hdr = self.table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for c in (1, 2, 3, 4):
            hdr.setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(5, QHeaderView.ResizeMode.Interactive)
        self.table.setColumnWidth(5, 220)
        self.table.itemSelectionChanged.connect(self._selection_changed)
        self.table.doubleClicked.connect(lambda _i: self.open_output_folder())
        self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._table_menu)

        hint = QLabel("Drop photos, videos or whole folders here\n\n"
                      f"Photos: {', '.join(sorted(e.lstrip('.') for e in PHOTO_EXTS))}\n"
                      f"Videos: {', '.join(sorted(e.lstrip('.') for e in VIDEO_EXTS))}")
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hint.setWordWrap(True)
        hint.setMinimumHeight(100)  # may shrink so the settings below always fit
        hint.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Ignored)
        hint.setStyleSheet("QLabel { border: 2px dashed #7c8798; border-radius: 12px; color: #7c8798; font-size: 15px; padding: 12px; }")
        self.queue_stack = QStackedWidget()
        self.queue_stack.addWidget(hint)
        self.queue_stack.addWidget(self.table)

        # Left column: queue above settings (settings scroll if the window is short).
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        settings = self._build_settings()
        scroll.setWidget(settings)
        self._settings_widget, self._settings_scroll = settings, scroll
        left = QSplitter(Qt.Orientation.Vertical)
        # Framed like the Preview panel, so the drop area doesn't touch the window edge.
        queue_box = QGroupBox("Queue")
        ql = QVBoxLayout(queue_box)
        ql.addWidget(self.queue_stack)
        left.addWidget(queue_box)
        left.addWidget(scroll)
        left.setStretchFactor(0, 1)
        left.setStretchFactor(1, 0)
        left.setCollapsible(0, False)
        self._left_split, self._sized = left, False
        split.addWidget(left)

        # Right: the preview gets all the height.
        pv = QGroupBox("Preview")
        pl = QVBoxLayout(pv)
        top = QHBoxLayout()
        self.mode_combo = QComboBox()
        for key, label in MODES.items():
            self.mode_combo.addItem(label, key)
        self.mode_combo.currentIndexChanged.connect(self._settings_changed)
        top.addWidget(QLabel("View:"))
        top.addWidget(self.mode_combo, 1)
        pl.addLayout(top)
        self.preview = PreviewLabel()
        pl.addWidget(self.preview, 1)
        self.preview_caption = QLabel("")
        self.preview_caption.setStyleSheet("color: #7c8798;")
        self.preview_caption.setWordWrap(True)
        pl.addWidget(self.preview_caption)
        split.addWidget(pv)
        split.setStretchFactor(0, 2)
        split.setStretchFactor(1, 3)
        split.setSizes([560, 800])
        self.setCentralWidget(split)

    def _build_settings(self) -> QWidget:
        box = QGroupBox("Settings")
        form = QFormLayout(box)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        prow = QHBoxLayout()
        self.preset_combo = QComboBox()
        self._reload_presets()
        self.preset_combo.activated.connect(self._preset_chosen)
        save = QPushButton("Save as…")
        save.clicked.connect(self.save_preset)
        self.delete_preset_btn = QPushButton("Delete")
        self.delete_preset_btn.clicked.connect(self.delete_preset)
        prow.addWidget(self.preset_combo, 1)
        prow.addWidget(save)
        prow.addWidget(self.delete_preset_btn)
        form.addRow("Preset", prow)

        grid = QGridLayout()
        self.format_checks: dict[str, QCheckBox] = {}
        for i, (key, label, tip) in enumerate(FORMAT_LABELS):
            cb = QCheckBox(label)
            cb.setToolTip(tip)
            cb.toggled.connect(self._settings_changed)
            self.format_checks[key] = cb
            grid.addWidget(cb, i // 2, i % 2)
        form.addRow("Output", grid)

        self.strength = SliderRow(0.5, 8.0, 0.1, 1, " %")
        self.strength.setToolTip("Total depth range as a percentage of the image width. 2 to 4 is comfortable.")
        self.strength.changed.connect(self._settings_changed)
        form.addRow("3D strength", self.strength)

        self.convergence = SliderRow(0.0, 1.0, 0.01, 2)
        self.convergence.setToolTip("Which depth sits at screen level. Lower = more pops out of the screen; 1 = all behind it.")
        self.convergence.changed.connect(self._settings_changed)
        conv = QWidget()
        cl = QVBoxLayout(conv)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(0)
        cl.addWidget(self.convergence)
        ends = QHBoxLayout()
        a, b = QLabel("more pop-out"), QLabel("behind the screen")
        for lab in (a, b):
            lab.setStyleSheet("color: #7c8798; font-size: 11px;")
        ends.addWidget(a)
        ends.addStretch(1)
        ends.addWidget(b)
        ends.addSpacing(90)
        cl.addLayout(ends)
        form.addRow("Convergence", conv)

        self.profile_combo = self._combo(PROFILE_LABELS)
        form.addRow("Depth quality", self.profile_combo)
        self.height_combo = self._combo(HEIGHTS)
        form.addRow("Resolution (per eye)", self.height_combo)
        self.fill_combo = self._combo([("auto", "Automatic (AI for photos, basic for video)"),
                                       ("ai", "AI inpainting (best, slow; Stable Diffusion)"),
                                       ("basic", "Basic (fast)")])
        self.fill_combo.setToolTip("How the gaps that open up beside foreground objects are filled.")
        form.addRow("Gap filling", self.fill_combo)
        self.eye_combo = self._combo([("symmetric", "Shift both eyes (smallest gaps)"),
                                      ("right", "Keep original as left eye"),
                                      ("left", "Keep original as right eye")])
        form.addRow("Eyes", self.eye_combo)

        vrow = QHBoxLayout()
        self.container_combo = self._combo([("auto", "Same as source"), ("mp4", "MP4"), ("mkv", "MKV"), ("mov", "MOV")])
        self.codec_combo = self._combo([("auto", "Automatic codec"), ("hevc", "HEVC (H.265)"), ("h264", "H.264")])
        vrow.addWidget(self.container_combo, 1)
        vrow.addWidget(self.codec_combo, 1)
        form.addRow("Video file", vrow)

        orow = QHBoxLayout()
        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText("A '3d' folder next to each file")
        self.output_edit.editingFinished.connect(self._settings_changed)
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse_output)
        orow.addWidget(self.output_edit, 1)
        orow.addWidget(browse)
        form.addRow("Save to", orow)
        return box

    def _combo(self, items) -> QComboBox:
        c = QComboBox()
        for data, label in items:
            c.addItem(label, data)
        c.currentIndexChanged.connect(self._settings_changed)
        return c

    def _build_log(self) -> None:
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(5000)
        dock = QDockWidget("Log", self)
        dock.setObjectName("log")
        dock.setWidget(self.log_view)
        dock.setAllowedAreas(Qt.DockWidgetArea.BottomDockWidgetArea)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, dock)
        self.resizeDocks([dock], [140], Qt.Orientation.Vertical)

    def _build_preview_thread(self, device: str) -> None:
        self.preview_thread = QThread(self)
        self.preview_worker = PreviewWorker(device)
        self.preview_worker.moveToThread(self.preview_thread)
        self.preview_requested.connect(self.preview_worker.request, Qt.ConnectionType.QueuedConnection)
        self.preview_worker.rendered.connect(self._preview_ready)
        self.preview_worker.status.connect(self._preview_status)
        self.preview_worker.failed.connect(self._preview_status)
        self.preview_thread.start()

    # -- options <-> widgets ----------------------------------------------------------------
    def options(self) -> ConvertOptions:
        return ConvertOptions(
            formats=[k for k, cb in self.format_checks.items() if cb.isChecked()] or ["sbs"],
            strength=self.strength.value(), convergence=self.convergence.value(),
            eye_mode=self.eye_combo.currentData(), profile=self.profile_combo.currentData(),
            height=self.height_combo.currentData(), codec=self.codec_combo.currentData(),
            container=self.container_combo.currentData(), output_dir=self.output_edit.text().strip(),
            fill=self.fill_combo.currentData(), fill_model=self._advanced.get("fill_model", "sd2"),
            model=self._advanced.get("model", ""), device=self._advanced.get("device", "auto"))

    def apply_options(self, o: ConvertOptions) -> None:
        self._loading_settings = True
        self._advanced = {"model": o.model, "device": o.device, "fill_model": o.fill_model}
        for k, cb in self.format_checks.items():
            cb.setChecked(k in o.formats)
        self.strength.setValue(o.strength)
        self.convergence.setValue(o.convergence)
        for combo, value in ((self.eye_combo, o.eye_mode), (self.profile_combo, o.profile), (self.height_combo, o.height),
                             (self.fill_combo, o.fill),
                             (self.codec_combo, o.codec), (self.container_combo, o.container)):
            i = combo.findData(value)
            combo.setCurrentIndex(i if i >= 0 else 0)
        self.output_edit.setText(o.output_dir)
        self._loading_settings = False
        self._settings_changed()

    def _settings_changed(self, *_):
        if self._loading_settings:
            return
        self.presets.save_last(self.options())
        self._render_timer.start()

    # -- presets ----------------------------------------------------------------------------
    def _reload_presets(self, select: str | None = None) -> None:
        self.preset_combo.blockSignals(True)
        self.preset_combo.clear()
        self.preset_combo.addItem("Choose a preset…", None)
        for name in self.presets.names():
            self.preset_combo.addItem(name + ("" if self.presets.is_builtin(name) else "  (mine)"), name)
        if select:
            self.preset_combo.setCurrentIndex(max(0, self.preset_combo.findData(select)))
        self.preset_combo.blockSignals(False)

    def _preset_chosen(self, index: int) -> None:
        name = self.preset_combo.itemData(index)
        if name:
            self.apply_options(self.presets.get(name, base=self.options()))
            self.statusBar().showMessage(f"Preset '{name}' applied", 4000)

    def save_preset(self) -> None:
        name, ok = QInputDialog.getText(self, "Save preset", "Name for these settings:")
        name = name.strip()
        if ok and name:
            try:
                self.presets.save(name, self.options())
            except ValueError as e:
                QMessageBox.warning(self, "Save preset", str(e))
                return
            self._reload_presets(select=name)

    def delete_preset(self) -> None:
        name = self.preset_combo.currentData()
        if not name or self.presets.is_builtin(name):
            QMessageBox.information(self, "Delete preset", "Choose one of your own presets to delete. Built-in presets can't be deleted.")
            return
        self.presets.delete(name)
        self._reload_presets()

    # -- queue ------------------------------------------------------------------------------
    def add_paths(self, paths: list[str]) -> None:
        try:
            n = self.runner.add(paths)
        except FileNotFoundError as e:
            QMessageBox.warning(self, "Add files", f"Not found: {e}")
            return
        self.statusBar().showMessage(f"Added {n} file(s)" if n else "No new photos or videos found there", 4000)
        if n:
            self.reset_3d_sliders()
            self.table.selectRow(self.table.rowCount() - n)  # preview the first new file

    def reset_3d_sliders(self) -> None:
        """New files start from the default 3D strength/convergence (the chosen preset's, if any)."""
        name = self.preset_combo.currentData()
        ref = self.presets.get(name) if name else ConvertOptions()
        self.strength.setValue(ref.strength)
        self.convergence.setValue(ref.convergence)
        self._settings_changed()

    def add_files_dialog(self) -> None:
        exts = " ".join(f"*{e}" for e in sorted(PHOTO_EXTS | VIDEO_EXTS))
        files, _ = QFileDialog.getOpenFileNames(self, "Add photos and videos", "", f"Photos and videos ({exts});;All files (*)")
        if files:
            self.add_paths(files)

    def add_folder_dialog(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Add a folder (sub-folders included)")
        if d:
            self.add_paths([d])

    def _rebuild_table(self) -> None:
        jobs = self.runner.jobs
        self.table.setRowCount(len(jobs))
        for r in range(len(jobs)):
            self._update_row(r)
        self.queue_stack.setCurrentIndex(1 if jobs else 0)
        self._update_status()

    def _update_row(self, r: int) -> None:
        if r >= self.table.rowCount():
            return
        j = self.runner.jobs[r]
        values = [j.path.name, j.kind.capitalize(), j.status, None,
                  fmt_time(j.eta) if (j.eta is not None and j.status == RUNNING) else "", j.error or j.message]
        for c, v in enumerate(values):
            if c == 3:
                continue
            item = self.table.item(r, c)
            if item is None:
                item = QTableWidgetItem()
                self.table.setItem(r, c, item)
            item.setText(v)
            if c == 0:
                item.setToolTip(str(j.path))
            if c == 5:
                item.setToolTip("\n".join(j.outputs) or j.error or j.message)
            if c == 2:
                color = {DONE: "#2e9e5b", SKIPPED: "#2e9e5b", FAILED: "#d24b4b", STOPPED: "#c98a1a"}.get(j.status)
                item.setData(Qt.ItemDataRole.ForegroundRole, None if color is None else _qcolor(color))
        bar = self.table.cellWidget(r, 3)
        if not isinstance(bar, QProgressBar):
            bar = QProgressBar()
            bar.setTextVisible(True)
            bar.setMaximumHeight(18)
            self.table.setCellWidget(r, 3, bar)
        if j.progress < 0:
            bar.setRange(0, 0)  # busy indicator
        else:
            bar.setRange(0, 1000)
            bar.setValue(int(j.progress * 1000))
            bar.setFormat(f"{j.progress * 100:.0f}%")
        self._update_status()

    def _update_status(self) -> None:
        jobs = self.runner.jobs
        finished = sum(j.status in FINISHED for j in jobs)
        failed = sum(j.status == FAILED for j in jobs)
        if jobs:
            msg = f"{finished} of {len(jobs)} finished" + (f", {failed} failed" if failed else "")
            self.statusBar().showMessage(msg)

    def _table_menu(self, pos) -> None:
        rows = sorted({i.row() for i in self.table.selectedIndexes()})
        if not rows:
            return
        m = QMenu(self)
        m.addAction("Open output folder", self.open_output_folder)
        m.addAction("Remove from queue", lambda: self.runner.remove(rows))
        if any(self.runner.jobs[r].status in (FAILED, STOPPED, DONE, SKIPPED) for r in rows):
            m.addAction("Convert again", lambda: self._requeue(rows))
        m.exec(self.table.viewport().mapToGlobal(pos))

    def _requeue(self, rows: list[int]) -> None:
        for r in rows:
            j = self.runner.jobs[r]
            if j.status in FINISHED or j.status == STOPPED:
                j.overwrite = j.status != STOPPED  # redo finished files; a stopped video still resumes
                j.status, j.progress, j.message, j.error = "Queued", 0.0, "", ""
                self._update_row(r)

    def start(self) -> None:
        o = self.options()
        if not any(j.status in ("Queued", STOPPED) for j in self.runner.jobs):
            self.statusBar().showMessage("Nothing to convert: add files first (or right-click a finished one, then Convert again)", 5000)
            return
        self.runner.start(o)

    def _running_changed(self, running: bool) -> None:
        self.start_action.setEnabled(not running)
        self.stop_action.setEnabled(running)

    def open_output_folder(self) -> None:
        rows = sorted({i.row() for i in self.table.selectedIndexes()})
        job = self.runner.jobs[rows[0]] if rows else None
        if job and job.outputs:
            folder = Path(job.outputs[0]).parent
        elif self.output_edit.text().strip():
            folder = Path(self.output_edit.text().strip())
        elif job:
            folder = job.path.parent / "3d"
        else:
            return
        if folder.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))
        else:
            self.statusBar().showMessage(f"{folder} doesn't exist yet", 4000)

    def _browse_output(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Save converted files to")
        if d:
            self.output_edit.setText(d)
            self._settings_changed()

    # -- preview ----------------------------------------------------------------------------
    def _selection_changed(self) -> None:
        self._render_timer.start()

    def _request_preview(self) -> None:
        rows = sorted({i.row() for i in self.table.selectedIndexes()})
        if not rows or rows[0] >= len(self.runner.jobs):
            return
        job = self.runner.jobs[rows[0]]
        self.preview_requested.emit((str(job.path), job.kind, self.options(), self.mode_combo.currentData()))

    def _preview_ready(self, images: list, caption: str) -> None:
        self.preview.set_images(images)
        self.preview_caption.setText(caption)

    def _preview_status(self, text: str) -> None:
        self.preview_caption.setText(text)
        if text.startswith("Preview failed"):
            self._append_log(text)

    # -- misc -------------------------------------------------------------------------------
    def _append_log(self, text: str) -> None:
        self.log_view.appendPlainText(text)

    def showEvent(self, e) -> None:
        super().showEvent(e)
        if not self._sized:  # measured once styled: settings get their full height, the queue the rest
            self._sized = True
            h = self._settings_widget.sizeHint().height() + 2
            self._settings_scroll.setMaximumHeight(h)  # never taller than the settings: no grey space
            total = sum(self._left_split.sizes())
            self._left_split.setSizes([max(140, total - h), h])

    def dragEnterEvent(self, e) -> None:
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e) -> None:
        paths = [u.toLocalFile() for u in e.mimeData().urls() if u.isLocalFile()]
        if paths:
            self.add_paths(paths)

    def closeEvent(self, e) -> None:
        if self.runner.running:
            answer = QMessageBox.question(self, "Quit Parallaxa",
                                          "A conversion is running. Stop it and quit?\n(Videos can be resumed later.)")
            if answer != QMessageBox.StandardButton.Yes:
                e.ignore()
                return
            self.runner.shutdown()
        self.preview_thread.quit()
        self.preview_thread.wait(5000)
        super().closeEvent(e)


def _qcolor(hexcode: str):
    from PySide6.QtGui import QColor

    return QColor(hexcode)
