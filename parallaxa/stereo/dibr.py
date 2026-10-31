"""Depth-image-based rendering: synthesise left/right views from one image plus depth.

Pipeline per eye:
  1. disparity = strength * width * (closeness - convergence), after dilating near objects so
     their silhouettes carry foreground depth (avoids "rubber sheet" smearing at edges);
  2. forward-warp every source pixel horizontally with a two-tap (sub-pixel) splat and a
     z-buffer, so nearer surfaces occlude farther ones;
  3. fill the disoccluded holes from the *farther* neighbouring side (holes are background
     that was hidden behind the foreground), then smooth the fill with depth-aware diffusion that
     never pulls colour from nearer surfaces.

Everything is vectorised torch, so it runs on the GPU alongside the depth model.
"""

from __future__ import annotations

import math
from contextlib import nullcontext

import torch
import torch.nn.functional as F

from ..config import StereoSettings


def _gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur of a (B, 1, H, W) tensor."""
    if sigma <= 0.3:
        return x
    r = max(1, int(math.ceil(sigma * 2.5)))
    t = torch.arange(-r, r + 1, device=x.device, dtype=x.dtype)
    k = torch.exp(-0.5 * (t / sigma) ** 2)
    k = k / k.sum()
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="replicate"), k.view(1, 1, 1, -1))
    return F.conv2d(F.pad(x, (0, 0, r, r), mode="replicate"), k.view(1, 1, -1, 1))


def prepare_closeness(closeness: torch.Tensor, settings: StereoSettings) -> torch.Tensor:
    """Edge dilation + light smoothing of a (B, H, W) [0, 1] closeness map."""
    w = closeness.shape[-1]
    x = closeness.unsqueeze(1)
    k = int(round(settings.edge_dilate * w))
    if k >= 1:
        x = F.max_pool2d(x, kernel_size=2 * k + 1, stride=1, padding=k)
    x = _gaussian_blur(x, settings.depth_blur * w)
    return x[:, 0]


def disparity_px(closeness: torch.Tensor, settings: StereoSettings) -> torch.Tensor:
    """Signed screen disparity in pixels. Positive = in front of the screen (crossed parallax)."""
    w = closeness.shape[-1]
    return (settings.strength / 100.0) * w * (closeness - settings.convergence)


def forward_warp(img: torch.Tensor, shift: torch.Tensor, z: torch.Tensor, z_tol: float = 1.0):
    """Splat each pixel of img (B, C, H, W) to x + shift (B, H, W), nearest-wins by z (larger = nearer).

    Returns (warped image, hole mask (B, H, W) bool, target z-buffer (B, H, W)).
    """
    b, c, h, w = img.shape
    dev = img.device
    xs = torch.arange(w, device=dev, dtype=torch.float32).view(1, 1, w)
    tx = xs + shift.float()
    x0 = torch.floor(tx)
    frac = tx - x0
    x0 = x0.long()
    row = (torch.arange(b * h, device=dev).view(b, h, 1) * w)  # flat row offsets

    taps = []
    for xi, wt in ((x0, 1.0 - frac), (x0 + 1, frac)):
        valid = (xi >= 0) & (xi < w) & (wt > 1e-4)
        idx = (row + xi.clamp(0, w - 1)).flatten()
        taps.append((idx, wt.flatten(), valid.flatten()))

    zf = z.float().flatten()
    zbuf = torch.full((b * h * w,), -1e9, device=dev)
    for idx, _, valid in taps:
        zbuf.scatter_reduce_(0, idx[valid], zf[valid], reduce="amax")

    colour = img.float().permute(0, 2, 3, 1).reshape(-1, c)
    acc = torch.zeros((b * h * w, c), device=dev)
    wsum = torch.zeros((b * h * w,), device=dev)
    for idx, wt, valid in taps:
        keep = valid & (zf >= zbuf[idx] - z_tol)
        weight = wt * keep
        acc.index_add_(0, idx, colour * weight.unsqueeze(1))
        wsum.index_add_(0, idx, weight)

    holes = wsum < 1e-3
    out = acc / wsum.clamp_min(1e-6).unsqueeze(1)
    out = out.view(b, h, w, c).permute(0, 3, 1, 2)
    return out, holes.view(b, h, w), zbuf.view(b, h, w)


def _nearest_valid(valid: torch.Tensor):
    """Indices of the nearest valid pixel to the left and right along each row (-1 / W if none)."""
    w = valid.shape[-1]
    idx = torch.arange(w, device=valid.device).expand_as(valid)
    left = torch.where(valid, idx, torch.full_like(idx, -1)).cummax(dim=-1).values
    right = torch.where(valid, idx, torch.full_like(idx, w)).flip(-1).cummin(dim=-1).values.flip(-1)
    return left, right


def fill_holes(img: torch.Tensor, holes: torch.Tensor, zbuf: torch.Tensor, iterations: int = 8, z_tol: float = 1.0) -> torch.Tensor:
    """Fill disocclusions with background content.

    img (B, C, H, W), holes (B, H, W) bool, zbuf (B, H, W) target closeness (valid where ~holes).
    """
    if not holes.any():
        return img
    b, c, h, w = img.shape
    valid = ~holes
    left, right = _nearest_valid(valid)
    has_l, has_r = left >= 0, right < w
    lz = torch.gather(zbuf, 2, left.clamp(0, w - 1))
    rz = torch.gather(zbuf, 2, right.clamp(0, w - 1))
    # Take the farther (smaller z) side; fall back to whichever side exists.
    use_left = has_l & (~has_r | (lz <= rz))
    src = torch.where(use_left, left, right).clamp(0, w - 1)
    fill_z = torch.where(use_left, lz, rz)
    z = torch.where(holes, fill_z, zbuf)

    idx = src.unsqueeze(1).expand(b, c, h, w)
    out = torch.where(holes.unsqueeze(1), torch.gather(img, 3, idx), img)

    if iterations <= 0:
        return out

    # Depth-aware diffusion inside the holes: average 3x3 neighbours that are not nearer than the
    # hole's background. Removes the streaky look of plain horizontal stretching. Only hole pixels
    # change and z is fixed, so neighbour indices and weights are computed once and each pass is a
    # small gather over the hole pixels rather than a full-frame operation.
    hole_idx = holes.flatten().nonzero().squeeze(1)
    bi, rem = hole_idx // (h * w), hole_idx % (h * w)
    y, x = rem // w, rem % w
    offs = torch.tensor([(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)], device=img.device)
    ny = (y.unsqueeze(1) + offs[:, 0]).clamp_(0, h - 1)  # clamping = replicate padding at borders
    nx = (x.unsqueeze(1) + offs[:, 1]).clamp_(0, w - 1)
    nbr = bi.unsqueeze(1) * (h * w) + ny * w + nx  # (N, 9)
    zf = z.flatten()
    wt = (zf[nbr] <= zf[hole_idx].unsqueeze(1) + z_tol).to(out.dtype)
    wt[:, 4] *= 2.0  # centre pixel
    wt = (wt / wt.sum(1, keepdim=True)).unsqueeze(2)

    col = out.permute(0, 2, 3, 1).reshape(-1, c).contiguous()
    for _ in range(iterations):
        col[hole_idx] = (col[nbr] * wt).sum(1)
    return col.view(b, h, w, c).permute(0, 3, 1, 2)


class StereoRenderer:
    def __init__(self, settings: StereoSettings):
        self.s = settings

    @torch.inference_mode()
    def render(self, img: torch.Tensor, closeness: torch.Tensor, timer=None) -> tuple[torch.Tensor, torch.Tensor]:
        """img (B, 3, H, W) float [0, 1]; closeness (B, H, W) normalised [0, 1].

        Returns (left, right), each (B, 3, H, W) float [0, 1]. `timer(name)` is an optional context
        manager factory used to time the sub-stages.
        """
        s = self.s
        tm = timer or (lambda _name: nullcontext())
        with tm("render: prepare depth"):
            near = prepare_closeness(closeness.float(), s)
            disp = disparity_px(near, s)
        # z-key = disparity in pixels, so z_tolerance is measured in pixels of parallax.
        z = disp

        def synth(src: torch.Tensor, shift: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
            with tm("render: warp"):
                warped, holes, zbuf = forward_warp(src, shift, z, s.z_tolerance)
            with tm("render: fill holes"):
                return fill_holes(warped, holes, zbuf, s.fill_iterations, s.z_tolerance).clamp_(0, 1)

        if s.eye_mode == "right":
            return img.float(), synth(img, -disp, z)
        if s.eye_mode == "left":
            return synth(img, disp, z), img.float()
        # Both eyes in one batched pass (fewer, larger GPU kernels).
        both = synth(torch.cat([img, img]), torch.cat([disp * 0.5, -disp * 0.5]), torch.cat([z, z]))
        return both[: img.shape[0]], both[img.shape[0] :]
