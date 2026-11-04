"""GUI tests (offscreen Qt): options/presets, preview engine, queue runner with real child processes, main window."""

import os
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")

from parallaxa.gui.jobs import DONE, FAILED, STOPPED, QueueRunner  # noqa: E402
from parallaxa.gui.options import BUILTIN_PRESETS, ConvertOptions, PresetStore  # noqa: E402
from parallaxa.gui.preview import MODES, PreviewEngine  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def pump(app, seconds, until):
    end = time.time() + seconds
    while time.time() < end:
        app.processEvents()
        if until():
            return True
        time.sleep(0.02)
    return False


def test_options_to_cli_args_roundtrip():
    from parallaxa.cli import build_parser

    o = ConvertOptions(formats=["sbs", "anaglyph"], strength=4.5, convergence=0.3, height=1080, output_dir="/tmp/x",
                       model="debug-gradient")
    a = build_parser().parse_args(["convert", "in.jpg", *o.to_args()])
    assert a.format == ("sbs", "anaglyph") and a.strength == 4.5 and a.convergence == 0.3
    assert a.height == 1080 and a.output == "/tmp/x" and a.model == "debug-gradient"


def test_presets(tmp_path):
    store = PresetStore(tmp_path)
    assert set(BUILTIN_PRESETS) <= set(store.names())
    base = ConvertOptions(output_dir="D:/out", device="cpu")
    tv = store.get("3D TV (half SBS)", base=base)
    assert tv.formats == ["hsbs"] and tv.output_dir == "D:/out" and tv.device == "cpu"
    store.save("Mine", ConvertOptions(strength=5.0, convergence=0.2, output_dir="ignored"))
    assert "Mine" in store.names() and store.get("Mine").strength == 5.0 and store.get("Mine").output_dir == ""
    with pytest.raises(ValueError):
        store.save("Red/cyan glasses", ConvertOptions())
    store.delete("Mine")
    assert "Mine" not in store.names()
    store.save_last(ConvertOptions(strength=2.2))
    assert store.load_last().strength == 2.2


def test_preview_engine_modes(tmp_path, scene):
    from PIL import Image

    img, _ = scene
    p = tmp_path / "p.png"
    Image.fromarray(img).save(p)
    eng = PreviewEngine("cpu")
    opts = ConvertOptions(model="debug-gradient", fill_model="debug-tiny")
    eng.load(p, "photo", opts)
    h, w = img.shape[:2]
    assert not eng.wants_ai(opts, "photo") and eng.wants_ai(ConvertOptions(fill="ai"), "video")
    for mode in MODES:
        out = eng.render(opts, mode, ai=mode == "anaglyph")
        assert len(out) == (2 if mode == "wiggle" else 1)
        assert out[0].shape == ((h, 2 * w, 3) if mode in ("sbs", "cross") else (h, w, 3))


def test_preview_engine_video_frame(sample_video):
    eng = PreviewEngine("cpu")
    eng.load(sample_video, "video", ConvertOptions(model="debug-gradient"))
    assert eng.render(ConvertOptions(), "anaglyph")[0].shape == (90, 160, 3)


def test_queue_runs_jobs_in_child_processes(app, tmp_path, sample_video, scene):
    from PIL import Image

    Image.fromarray(scene[0]).save(tmp_path / "photo.png")
    runner = QueueRunner()
    assert runner.add([str(sample_video), str(tmp_path / "photo.png")]) == 2
    assert runner.add([str(sample_video)]) == 0  # no duplicates
    opts = ConvertOptions(model="debug-gradient", device="cpu", output_dir=str(tmp_path / "out"), formats=["sbs"],
                          fill_model="debug-tiny")
    runner.start(opts)
    assert pump(app, 120, lambda: not runner.running)
    statuses = {j.path.name: j.status for j in runner.jobs}
    assert statuses == {"clip.mp4": DONE, "photo.png": DONE}, [(j.path.name, j.status, j.error) for j in runner.jobs]
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == ["clip_3D_SBS.mp4", "photo_3D_SBS.jpg"]


def test_queue_reports_failures(app, tmp_path):
    bad = tmp_path / "broken.mp4"
    bad.write_bytes(b"not a video")
    runner = QueueRunner()
    runner.add([str(bad)])
    runner.start(ConvertOptions(model="debug-gradient", device="cpu", output_dir=str(tmp_path / "out")))
    assert pump(app, 60, lambda: not runner.running)
    assert runner.jobs[0].status == FAILED and runner.jobs[0].error


def test_stop_then_resume(app, tmp_path):
    import subprocess

    src = tmp_path / "long.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=25:duration=20",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(src)], check=True)
    out = tmp_path / "out"
    opts = ConvertOptions(model="debug-random", device="cpu", output_dir=str(out), segment_minutes=0.02)  # 30-frame segments
    runner = QueueRunner()
    runner.add([str(src)])
    runner.start(opts)
    assert pump(app, 120, lambda: runner.jobs[0].done >= 70)
    runner.stop()
    assert pump(app, 60, lambda: not runner.running)
    assert runner.jobs[0].status == STOPPED
    assert (out / "long.parallaxa-work" / "state.json").exists()  # finished segments kept
    runner.start(opts)
    assert pump(app, 240, lambda: not runner.running)
    assert runner.jobs[0].status == DONE, runner.jobs[0].error
    assert (out / "long_3D_SBS.mp4").exists() and not (out / "long.parallaxa-work").exists()


def test_main_window_smoke(app, tmp_path, sample_video):
    from parallaxa.gui.main_window import MainWindow

    store = PresetStore(tmp_path / "cfg")
    store.save_last(ConvertOptions(fill_model="debug-tiny"))
    win = MainWindow(presets=store, preview_device="cpu")
    assert win.options().fill_model == "debug-tiny" and win.options().fill == "basic"
    win.add_paths([str(sample_video)])
    assert win.table.rowCount() == 1 and win.queue_stack.currentIndex() == 1
    win.apply_options(store.get("Strong pop-out", base=win.options()))
    o = win.options()
    assert o.strength == 4.0 and o.convergence == 0.35 and o.formats == ["sbs"]
    win.strength.setValue(2.5)
    win.strength.changed.emit(2.5)
    assert store.load_last().strength == 2.5  # settings persist
    win.close()


def test_sliders_reset_when_files_are_added(app, tmp_path, sample_video, scene):
    from PIL import Image

    from parallaxa.gui.main_window import MainWindow

    win = MainWindow(presets=PresetStore(tmp_path / "cfg"), preview_device="cpu")
    win.add_paths([str(sample_video)])
    win.strength.setValue(7.0)
    win.convergence.setValue(0.1)
    Image.fromarray(scene[0]).save(tmp_path / "new.png")
    win.add_paths([str(tmp_path / "new.png")])
    assert (win.strength.value(), win.convergence.value()) == (3.0, 0.6)
    # with a preset chosen, its values are the defaults
    i = win.preset_combo.findData("Strong pop-out")
    win.preset_combo.setCurrentIndex(i)
    win._preset_chosen(i)
    win.strength.setValue(7.0)
    (tmp_path / "more").mkdir()
    Image.fromarray(scene[0]).save(tmp_path / "more" / "b.png")
    win.add_paths([str(tmp_path / "more")])
    assert (win.strength.value(), win.convergence.value()) == (4.0, 0.35)
    win.close()


def test_convert_again_overwrites(app, tmp_path, scene):
    from PIL import Image

    from parallaxa.gui.jobs import SKIPPED
    from parallaxa.gui.main_window import MainWindow

    Image.fromarray(scene[0]).save(tmp_path / "p.png")
    store = PresetStore(tmp_path / "cfg")
    store.save_last(ConvertOptions(model="debug-gradient", device="cpu", output_dir=str(tmp_path / "out"), fill="basic"))
    win = MainWindow(presets=store, preview_device="cpu")
    win.add_paths([str(tmp_path / "p.png")])
    win.start()
    assert pump(app, 60, lambda: not win.runner.running) and win.runner.jobs[0].status == DONE
    out = tmp_path / "out" / "p_3D_SBS.jpg"
    first = out.stat().st_mtime_ns
    win.runner.jobs[0].status = DONE
    win.runner.jobs[0].overwrite = False
    win.runner.jobs[0].status = "Queued"  # a plain re-queue is skipped: output exists
    win.start()
    assert pump(app, 60, lambda: not win.runner.running) and win.runner.jobs[0].status == SKIPPED
    win.runner.jobs[0].status = DONE
    win._requeue([0])  # "Convert again" replaces the output
    assert win.runner.jobs[0].overwrite
    time.sleep(0.05)
    win.start()
    assert pump(app, 60, lambda: not win.runner.running) and win.runner.jobs[0].status == DONE
    assert out.stat().st_mtime_ns > first
    win.close()


def test_model_download_dialog_reports_failure_offline(app, monkeypatch):
    """Runs the real `parallaxa download` child; with no route to Hugging Face it must fail cleanly."""
    from parallaxa.gui.downloads import ModelDownloadDialog

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")  # no network use in tests
    monkeypatch.setenv("HF_HOME", "/nonexistent-parallaxa-test-cache")
    dlg = ModelDownloadDialog(["da2-small"])
    dlg.start()
    assert pump(app, 120, lambda: dlg.proc.state() == dlg.proc.ProcessState.NotRunning)
    pump(app, 1, lambda: False)
    assert dlg.failed and "da2-small" in dlg.failed[0]
    assert "failed" in dlg.labelText()
    dlg.close()


def test_preview_clears_when_its_file_leaves_the_queue(app, tmp_path, scene):
    from PIL import Image

    from parallaxa.gui.main_window import PREVIEW_HINT, MainWindow

    for name in ("a.png", "b.png"):
        Image.fromarray(scene[0]).save(tmp_path / name)
    store = PresetStore(tmp_path / "cfg")
    store.save_last(ConvertOptions(model="debug-gradient", device="cpu", fill="basic"))
    win = MainWindow(presets=store, preview_device="cpu")
    win.add_paths([str(tmp_path / "a.png"), str(tmp_path / "b.png")])
    win.table.selectRow(0)
    assert pump(app, 30, lambda: bool(win.preview._frames))

    win.runner.remove([1])  # removing another file keeps the preview
    assert win.preview._frames and win._preview_path.endswith("a.png")

    win.runner.jobs[0].status = DONE
    win.runner.clear_finished()  # the previewed file goes: preview and caption clear
    assert not win.preview._frames and win.preview.text() == PREVIEW_HINT and win.preview_caption.text() == ""

    win._preview_ready([scene[0]], "late result")  # a render finishing afterwards is ignored
    assert not win.preview._frames
    win.close()
