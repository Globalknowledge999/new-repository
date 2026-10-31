"""AI gap filling: masks that exclude the foreground, tiling, and compositing that only touches gaps."""

import torch

from parallaxa.config import StereoSettings
from parallaxa.stereo.dibr import StereoRenderer
from parallaxa.stereo.inpaint import load_inpainter, paint_masks, tile_origins


def test_hide_mask_covers_nearby_foreground_only():
    h, w = 40, 200
    z = torch.zeros(1, h, w)
    z[:, :, 100:130] = 10.0  # near object
    holes = torch.zeros(1, h, w, dtype=torch.bool)
    holes[:, :, 92:100] = True  # gap on its left, background depth 0
    paint, hide = paint_masks(holes, z, context_frac=0.1)  # 20 px radius
    assert paint[:, :, 92:100].all() and paint[:, :, 90].all() and not paint[:, :, 80].any()
    assert hide[:, :, 103:118].all()  # foreground beside the gap is hidden from the model
    assert not hide[:, :, 125:].any()  # ...only within the radius
    assert not hide[:, :, :90].any()  # background is kept as context


def test_tiles_cover_everything():
    for size in (100, 512, 513, 1000, 1024, 1500):
        xs = tile_origins(size, 512, 96)
        covered = torch.zeros(max(size, 512), dtype=torch.bool)
        for x in xs:
            covered[x : x + 512] = True
        assert covered[:size].all() and xs[0] == 0 and xs[-1] + 512 >= size


def test_inpainter_only_changes_gaps(scene):
    img, depth = scene
    t = torch.from_numpy(img).permute(2, 0, 1)[None].float() / 255
    inp = load_inpainter("debug-tiny", torch.device("cpu"), torch.float32)
    holes = torch.zeros(1, 96, 128, dtype=torch.bool)
    holes[:, 30:60, 40:46] = True
    z = torch.from_numpy(depth)[None] * 10
    out = inp.refine(t, holes, z)
    changed = (out - t).abs().amax(1)[0] > 1e-6
    assert changed[30:60, 40:46].any()
    near = torch.nn.functional.max_pool2d(holes[:, None].float(), 9, 1, 4)[0, 0] > 0
    assert not changed[~near].any()  # nothing outside the (feathered) gaps is touched


def test_renderer_uses_inpainter(scene):
    img, depth = scene
    t = torch.from_numpy(img).permute(2, 0, 1)[None].float() / 255
    calls = []

    class Spy:
        def refine(self, view, holes, z):
            calls.append((view.shape, holes.sum().item(), z.shape))
            return view

    StereoRenderer(StereoSettings(strength=10, convergence=0)).render(t, torch.from_numpy(depth)[None], inpainter=Spy())
    assert calls and calls[0][1] > 0  # gaps were handed over


def test_auto_model_falls_back_when_a_repository_is_unavailable(monkeypatch):
    import pytest

    from parallaxa.stereo import inpaint

    tried, logs = [], []

    def fake_load(repo, dtype):
        tried.append(repo)
        if "v1-5" in repo:
            raise OSError("401 Repository Not Found")
        return inpaint.tiny_pipeline()

    monkeypatch.setattr(inpaint, "_load_one", fake_load)
    pipe = inpaint.load_pipeline("auto", torch.device("cpu"), torch.float32, log=logs.append)
    assert pipe is not None and tried == [inpaint.INPAINT_MODELS["sd15"], inpaint.INPAINT_MODELS["sdxl"]]
    assert any("unavailable" in m for m in logs) and any("using" in m for m in logs)

    monkeypatch.setattr(inpaint, "_load_one", lambda repo, dtype: (_ for _ in ()).throw(OSError("offline")))
    with pytest.raises(RuntimeError, match="--fill basic"):
        inpaint.load_pipeline("auto", torch.device("cpu"), torch.float32, log=logs.append)
