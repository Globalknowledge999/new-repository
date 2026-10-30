"""End-to-end CLI runs with offline debug depth models (no downloads)."""

import json
import subprocess

import numpy as np
from PIL import Image

from parallaxa.cli import main


def _ffprobe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", "-count_frames", str(path)],
                         capture_output=True, text=True, check=True)
    return json.loads(out.stdout)["streams"]


def test_photo_all_formats(tmp_path, scene):
    img, _ = scene
    src = tmp_path / "pic.png"
    Image.fromarray(img).save(src)
    out = tmp_path / "out"
    rc = main(["convert", str(src), "-o", str(out), "--model", "debug-random", "--device", "cpu",
               "-f", "sbs,hsbs,tb,htb,anaglyph,mpo,depth,rgbd"])
    assert rc == 0
    names = sorted(p.name for p in out.iterdir())
    assert names == sorted(["pic_3D_SBS.jpg", "pic_3D_HSBS.jpg", "pic_3D_TB.jpg", "pic_3D_HTB.jpg",
                            "pic_3D_ANAGLYPH.jpg", "pic_3D.mpo", "pic_DEPTH.png", "pic_RGBD.png"])
    assert Image.open(out / "pic_3D_SBS.jpg").size == (256, 96)
    assert Image.open(out / "pic_3D_HSBS.jpg").size == (128, 96)
    assert Image.open(out / "pic_3D_TB.jpg").size == (128, 192)
    d = np.asarray(Image.open(out / "pic_DEPTH.png"))
    assert d.dtype == np.uint16 and d.shape == (96, 128)


def test_photo_resize_and_skip_existing(tmp_path, scene):
    img, _ = scene
    src = tmp_path / "pic.jpg"
    Image.fromarray(img).save(src)
    args = ["convert", str(tmp_path), "-o", str(tmp_path / "o"), "--model", "debug-gradient", "--device", "cpu", "--height", "48"]
    assert main(args) == 0
    assert Image.open(tmp_path / "o" / "pic_3D_SBS.jpg").size == (128, 48)
    mtime = (tmp_path / "o" / "pic_3D_SBS.jpg").stat().st_mtime_ns
    assert main(args) == 0
    assert (tmp_path / "o" / "pic_3D_SBS.jpg").stat().st_mtime_ns == mtime


def test_video_sbs_with_audio(tmp_path, sample_video):
    out = tmp_path / "out"
    rc = main(["convert", str(sample_video), "-o", str(out), "--model", "debug-random", "--device", "cpu",
               "-f", "sbs,hsbs,anaglyph", "--encoder", "cpu", "--preset", "ultrafast", "--progress", "none"])
    assert rc == 0
    sbs = out / "clip_3D_SBS.mp4"
    streams = _ffprobe(sbs)
    v = next(s for s in streams if s["codec_type"] == "video")
    a = next(s for s in streams if s["codec_type"] == "audio")
    assert (v["width"], v["height"]) == (320, 90)
    assert int(v["nb_read_frames"]) == 36
    assert a["codec_name"] == "aac"
    hs = next(s for s in _ffprobe(out / "clip_3D_HSBS.mp4") if s["codec_type"] == "video")
    assert (hs["width"], hs["height"]) == (160, 90)
    assert (out / "clip_3D_ANAGLYPH.mp4").exists()
    assert not list(out.glob("*.partial.*"))


def test_video_mkv_container_stereo_tag(tmp_path, sample_video):
    out = tmp_path / "out"
    rc = main(["convert", str(sample_video), "-o", str(out), "--model", "debug-gradient", "--device", "cpu",
               "--container", "mkv", "--encoder", "cpu", "--preset", "ultrafast", "--progress", "none", "--height", "64"])
    assert rc == 0
    streams = _ffprobe(out / "clip_3D_SBS.mkv")
    v = next(s for s in streams if s["codec_type"] == "video")
    assert (v["width"], v["height"]) == (228, 64)
    assert any(s["codec_type"] == "audio" for s in streams)
