"""Phase 2 end-to-end: Video Depth Anything streaming, segmented output, resume after interruption."""

import json
import subprocess

import pytest

from parallaxa.cli import main
from parallaxa.pipeline import video as video_mod

COMMON = ["--device", "cpu", "--encoder", "cpu", "--preset", "ultrafast", "--progress", "none"]


def _streams(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", "-count_frames", str(path)],
                         capture_output=True, text=True, check=True)
    return json.loads(out.stdout)["streams"]


def _video_frames(path):
    return int(next(s for s in _streams(path) if s["codec_type"] == "video")["nb_read_frames"])


def _framemd5(path):
    out = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:v", "-f", "framemd5", "-"],
                         capture_output=True, text=True, check=True)
    return [line.split(",")[-1].strip() for line in out.stdout.splitlines() if not line.startswith("#")]


def test_video_depth_anything_stream(tmp_path, sample_video):
    out = tmp_path / "out"
    assert main(["convert", str(sample_video), "-o", str(out), "--model", "debug-vda", "--depth-res", "42", *COMMON]) == 0
    result = out / "clip_3D_SBS.mp4"
    assert _video_frames(result) == 36
    assert any(s["codec_type"] == "audio" for s in _streams(result))
    assert not list(out.glob("*.parallaxa-work"))  # work folder removed after success


def test_segments_are_joined_seamlessly(tmp_path, sample_video):
    out = tmp_path / "out"
    # 0.005 min at 24 fps = 7 frames per segment -> 6 segments for 36 frames
    assert main(["convert", str(sample_video), "-o", str(out), "--model", "debug-gradient", "--segment-minutes", "0.005",
                 "-f", "sbs,tb", *COMMON]) == 0
    for name in ("clip_3D_SBS.mp4", "clip_3D_TB.mp4"):
        assert _video_frames(out / name) == 36
        assert any(s["codec_type"] == "audio" for s in _streams(out / name))


@pytest.mark.parametrize("model", ["debug-gradient", "debug-vda"])
def test_resume_after_interruption(tmp_path, sample_video, monkeypatch, model):
    args = ["convert", str(sample_video), "--model", model, "--depth-res", "42", "--segment-minutes", "0.005", *COMMON]
    ref_dir, out = tmp_path / "ref", tmp_path / "out"
    assert main(args + ["-o", str(ref_dir)]) == 0

    real_save = video_mod.WorkDir.save
    calls = {"n": 0}

    def save_then_crash(self, *a, **k):
        real_save(self, *a, **k)
        calls["n"] += 1
        if calls["n"] == 3:
            raise KeyboardInterrupt  # simulate the user stopping it after 3 segments

    monkeypatch.setattr(video_mod.WorkDir, "save", save_then_crash)
    assert main(args + ["-o", str(out)]) == 130
    work = out / "clip.parallaxa-work"
    assert json.loads((work / "state.json").read_text())["done_segments"] == 3
    assert not (out / "clip_3D_SBS.mp4").exists()

    monkeypatch.setattr(video_mod.WorkDir, "save", real_save)
    assert main(args + ["-o", str(out)]) == 0
    result = out / "clip_3D_SBS.mp4"
    assert _video_frames(result) == 36
    assert not work.exists()
    if model == "debug-gradient":  # per-frame depth: resumed output must be identical, frame for frame
        assert _framemd5(result) == _framemd5(ref_dir / "clip_3D_SBS.mp4")


def test_changed_settings_restart_instead_of_resume(tmp_path, sample_video, monkeypatch):
    out = tmp_path / "out"
    base = ["convert", str(sample_video), "-o", str(out), "--model", "debug-gradient", "--segment-minutes", "0.005", *COMMON]
    real_save = video_mod.WorkDir.save

    def crash(self, *a, **k):
        real_save(self, *a, **k)
        raise KeyboardInterrupt

    monkeypatch.setattr(video_mod.WorkDir, "save", crash)
    assert main(base) == 130
    monkeypatch.setattr(video_mod.WorkDir, "save", real_save)
    assert main(base + ["--strength", "5"]) == 0  # different settings: old segments must not be reused
    assert _video_frames(out / "clip_3D_SBS.mp4") == 36


def test_vda_resume_uses_saved_anchor(tmp_path, sample_video, monkeypatch):
    """With a short warm-up the resumed run starts mid-video and must seed the stream from the saved anchor."""
    monkeypatch.setattr(video_mod, "WARMUP_TEMPORAL", 8)
    seeded = {}
    from parallaxa.depth.video_depth_anything import VideoDepthStream

    real_seed = VideoDepthStream.seed

    def spy(self, state):
        seeded["keys"] = sorted(state)
        real_seed(self, state)

    monkeypatch.setattr(VideoDepthStream, "seed", spy)
    out = tmp_path / "out"
    args = ["convert", str(sample_video), "-o", str(out), "--model", "debug-vda", "--depth-res", "42",
            "--segment-minutes", "0.01", *COMMON]  # 14-frame segments
    real_save = video_mod.WorkDir.save
    calls = {"n": 0}

    def save_then_crash(self, *a, **k):
        real_save(self, *a, **k)
        calls["n"] += 1
        if calls["n"] == 2:
            raise KeyboardInterrupt

    monkeypatch.setattr(video_mod.WorkDir, "save", save_then_crash)
    assert main(args) == 130
    monkeypatch.setattr(video_mod.WorkDir, "save", real_save)
    assert main(args) == 0  # resumes at frame 28, decoding from frame 20
    assert seeded["keys"] == ["anchor_input", "ref_global"]
    assert _video_frames(out / "clip_3D_SBS.mp4") == 36
