import torch

from parallaxa.config import StereoSettings
from parallaxa.stereo.dibr import StereoRenderer, disparity_px, fill_holes, forward_warp


def _tensors(scene):
    img, depth = scene
    t = torch.from_numpy(img).permute(2, 0, 1)[None].float() / 255
    return t, torch.from_numpy(depth)[None]


def test_disparity_sign_and_convergence():
    s = StereoSettings(strength=10.0, convergence=0.5)
    c = torch.tensor([[[0.0, 0.5, 1.0]]]).expand(1, 1, 3).contiguous()
    d = disparity_px(c, s)
    w = 3
    assert torch.allclose(d, torch.tensor([[[-0.05 * w, 0.0, 0.05 * w]]]))


def test_integer_shift_moves_pixels_exactly():
    img = torch.arange(10.0).view(1, 1, 1, 10).expand(1, 3, 1, 10).contiguous()
    shift = torch.full((1, 1, 10), 2.0)
    out, holes, _ = forward_warp(img, shift, torch.zeros(1, 1, 10))
    assert holes[0, 0, :2].all() and not holes[0, 0, 2:].any()
    assert torch.allclose(out[0, 0, 0, 2:], torch.arange(8.0))


def test_near_object_occludes_far():
    # Far row moves 0 px, the near pixel at x=3 moves +2 onto x=5 and must win there.
    img = torch.zeros(1, 3, 1, 10)
    img[..., 3] = 1.0
    shift = torch.zeros(1, 1, 10)
    shift[..., 3] = 2.0
    z = shift.clone()
    out, holes, _ = forward_warp(img, shift, z)
    assert out[0, 0, 0, 5] == 1.0
    assert holes[0, 0, 3]  # disocclusion where the near pixel used to be


def test_holes_filled_from_background_side():
    # Hole between a near (z=5, colour 1) and a far (z=0, colour 0.2) region must take the far colour.
    img = torch.zeros(1, 3, 1, 8)
    img[..., :3] = 1.0
    img[..., 5:] = 0.2
    holes = torch.zeros(1, 1, 8, dtype=torch.bool)
    holes[..., 3:5] = True
    z = torch.zeros(1, 1, 8)
    z[..., :3] = 5.0
    out = fill_holes(img, holes, z, iterations=4)
    assert torch.allclose(out[0, :, 0, 3:5], torch.full((3, 2), 0.2))


def test_render_symmetric_parallax(scene):
    img, depth = _tensors(scene)
    s = StereoSettings(strength=10.0, convergence=0.0, edge_dilate=0, depth_blur=0)
    left, right = StereoRenderer(s).render(img, depth)
    # Square (closeness 1) gets 10% of 128 = 12.8 px total parallax: +6.4 in left, -6.4 in right.
    lum_l = left[0].mean(0)[48]
    lum_r = right[0].mean(0)[48]
    bright_l = torch.nonzero(lum_l > 0.95).flatten()
    bright_r = torch.nonzero(lum_r > 0.95).flatten()
    assert abs(bright_l.float().mean() - (63.5 + 6.4)) < 1.0
    assert abs(bright_r.float().mean() - (63.5 - 6.4)) < 1.0
    # No unfilled black holes, and no bright foreground smeared into the disocclusions.
    assert left.min() >= 0 and right.max() <= 1
    assert (lum_l[48:54] < 0.9).all()
    assert (lum_r[74:80] < 0.9).all()


def test_eye_mode_right_keeps_original_left(scene):
    img, depth = _tensors(scene)
    left, right = StereoRenderer(StereoSettings(eye_mode="right")).render(img, depth)
    assert torch.equal(left, img)
    assert not torch.equal(right, img)


def _reference_fill(img, holes, zbuf, iterations, z_tol=1.0):
    """The original full-frame diffusion, kept to check the sparse version gives the same result."""
    import torch.nn.functional as F

    out = fill_holes(img, holes, zbuf, iterations=0, z_tol=z_tol)
    b, c, h, w = img.shape
    # Rebuild z exactly as fill_holes does for the row fill.
    from parallaxa.stereo.dibr import _nearest_valid

    left, right = _nearest_valid(~holes)
    lz = torch.gather(zbuf, 2, left.clamp(0, w - 1))
    rz = torch.gather(zbuf, 2, right.clamp(0, w - 1))
    use_left = (left >= 0) & ((right >= w) | (lz <= rz))
    z = torch.where(holes, torch.where(use_left, lz, rz), zbuf)
    for _ in range(iterations):
        p_img = F.pad(out, (1, 1, 1, 1), mode="replicate")
        p_z = F.pad(z.unsqueeze(1), (1, 1, 1, 1), mode="replicate")
        acc, wsum = torch.zeros_like(out), torch.zeros_like(z.unsqueeze(1))
        for dy in (0, 1, 2):
            for dx in (0, 1, 2):
                wt = (p_z[:, :, dy : dy + h, dx : dx + w] <= z.unsqueeze(1) + z_tol).float() * (2.0 if dy == dx == 1 else 1.0)
                acc += p_img[:, :, dy : dy + h, dx : dx + w] * wt
                wsum += wt
        out = torch.where(holes.unsqueeze(1), acc / wsum, out)
    return out


def test_sparse_fill_matches_full_frame_reference():
    torch.manual_seed(1)
    img = torch.rand(2, 3, 20, 30)
    holes = torch.rand(2, 20, 30) < 0.25
    zbuf = torch.rand(2, 20, 30) * 5
    fast = fill_holes(img, holes, zbuf, iterations=6)
    ref = _reference_fill(img, holes, zbuf, iterations=6)
    assert torch.allclose(fast, ref, atol=1e-5)
