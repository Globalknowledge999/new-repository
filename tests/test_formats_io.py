import numpy as np
import pytest
import torch
from PIL import Image

from parallaxa.io.image_io import load_image, save_mpo
from parallaxa.pipeline.common import collect_inputs, even_size
from parallaxa.stereo.formats import compose


@pytest.mark.parametrize("fmt,shape", [("sbs", (2, 3, 10, 24)), ("hsbs", (2, 3, 10, 12)), ("tb", (2, 3, 20, 12)),
                                       ("htb", (2, 3, 10, 12)), ("anaglyph", (2, 3, 10, 12)), ("rgbd", (2, 3, 10, 24)),
                                       ("depth", (2, 3, 10, 12))])
def test_layout_shapes(fmt, shape):
    l, r = torch.rand(2, 3, 10, 12), torch.rand(2, 3, 10, 12)
    out = compose(fmt, l, r, l, torch.rand(2, 10, 12))
    assert tuple(out.shape) == shape
    assert out.min() >= 0 and out.max() <= 1


def test_sbs_order_left_then_right():
    l, r = torch.zeros(1, 3, 4, 4), torch.ones(1, 3, 4, 4)
    out = compose("sbs", l, r)
    assert out[..., :4].max() == 0 and out[..., 4:].min() == 1


def test_mpo_roundtrip(tmp_path):
    left = np.full((40, 60, 3), (200, 30, 30), np.uint8)
    right = np.full((40, 60, 3), (30, 30, 200), np.uint8)
    p = tmp_path / "x.mpo"
    save_mpo(left, right, p)
    with Image.open(p) as im:
        assert im.format == "MPO" and im.n_frames == 2
        entries = im.mpinfo[0xB002]
        assert entries[0]["Attribute"]["MPType"] == "Multi-Frame Image: (Disparity)"
        assert entries[1]["Attribute"]["MPType"] == "Multi-Frame Image: (Disparity)"
        frames = []
        for i in range(2):
            im.seek(i)
            frames.append(np.asarray(im.convert("RGB")).astype(int))
    assert frames[0][..., 0].mean() > 150 and frames[1][..., 2].mean() > 150


def test_heic_load(tmp_path):
    pillow_heif = pytest.importorskip("pillow_heif")
    p = tmp_path / "a.heic"
    try:
        pillow_heif.from_pillow(Image.new("RGB", (32, 24), (10, 120, 200))).save(p, quality=90)
    except Exception as e:  # encoder not bundled on this platform
        pytest.skip(f"HEIC encoder unavailable: {e}")
    arr, _ = load_image(p)
    assert arr.shape == (24, 32, 3)


def test_even_size():
    assert even_size(1919, 1083) == (1920, 1084)
    assert even_size(3840, 2160, 1080) == (1920, 1080)


def test_collect_inputs_skips_outputs(tmp_path):
    for n in ["a.jpg", "b.MP4", "c.txt", "a_3D_SBS.jpg", "a_3D.mpo"]:
        (tmp_path / n).write_bytes(b"x")
    found = {(p.name, k) for p, k in collect_inputs([str(tmp_path)])}
    assert found == {("a.jpg", "photo"), ("b.MP4", "video")}
