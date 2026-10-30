"""Pack a left/right view pair into the output layouts."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

# Least-squares anaglyph matrices for red/cyan glasses (Eric Dubois, 2001), applied to RGB.
_DUBOIS_L = torch.tensor([[0.437, 0.449, 0.164], [-0.062, -0.062, -0.024], [-0.048, -0.050, -0.017]])
_DUBOIS_R = torch.tensor([[-0.011, -0.032, -0.007], [0.377, 0.761, 0.009], [-0.026, -0.093, 1.234]])
_LUMA = torch.tensor([0.299, 0.587, 0.114])

LAYOUT_SUFFIX = {
    "sbs": "_3D_SBS",
    "hsbs": "_3D_HSBS",
    "tb": "_3D_TB",
    "htb": "_3D_HTB",
    "anaglyph": "_3D_ANAGLYPH",
    "rgbd": "_RGBD",
    "depth": "_DEPTH",
    "mpo": "_3D",
}


def _resize(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    return F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False, antialias=True)


def anaglyph(left: torch.Tensor, right: torch.Tensor, mode: str = "dubois") -> torch.Tensor:
    """left/right (B, 3, H, W) in [0, 1] -> red/cyan anaglyph (B, 3, H, W)."""
    if mode == "dubois":
        ml, mr = _DUBOIS_L.to(left), _DUBOIS_R.to(left)
        out = torch.einsum("ij,bjhw->bihw", ml, left) + torch.einsum("ij,bjhw->bihw", mr, right)
        return out.clamp(0, 1)
    if mode == "gray":
        lum = _LUMA.to(left).view(1, 3, 1, 1)
        gl, gr = (left * lum).sum(1, keepdim=True), (right * lum).sum(1, keepdim=True)
        return torch.cat([gl, gr, gr], 1)
    if mode == "color":
        return torch.cat([left[:, :1], right[:, 1:]], 1)
    if mode == "half-color":
        lum = _LUMA.to(left).view(1, 3, 1, 1)
        return torch.cat([(left * lum).sum(1, keepdim=True), right[:, 1:]], 1)
    raise ValueError(f"Unknown anaglyph mode {mode!r} (dubois, half-color, color, gray)")


def depth_to_rgb(closeness: torch.Tensor) -> torch.Tensor:
    """(B, H, W) [0, 1] -> (B, 3, H, W) greyscale, white = near."""
    return closeness.clamp(0, 1).unsqueeze(1).expand(-1, 3, -1, -1)


def compose(fmt: str, left: torch.Tensor, right: torch.Tensor, source: torch.Tensor | None = None,
            closeness: torch.Tensor | None = None, anaglyph_mode: str = "dubois") -> torch.Tensor:
    """Build one output layout. All inputs (B, 3, H, W) float [0, 1]; returns (B, 3, H', W')."""
    b, _, h, w = left.shape
    if fmt == "sbs":
        return torch.cat([left, right], dim=3)
    if fmt == "hsbs":
        hw = w // 2
        return torch.cat([_resize(left, h, hw), _resize(right, h, w - hw)], dim=3)
    if fmt == "tb":
        return torch.cat([left, right], dim=2)
    if fmt == "htb":
        hh = h // 2
        return torch.cat([_resize(left, hh, w), _resize(right, h - hh, w)], dim=2)
    if fmt == "anaglyph":
        return anaglyph(left, right, anaglyph_mode)
    if fmt == "rgbd":
        assert source is not None and closeness is not None
        return torch.cat([source, depth_to_rgb(closeness).to(source)], dim=3)
    if fmt == "depth":
        assert closeness is not None
        return depth_to_rgb(closeness)
    raise ValueError(f"Unknown output format {fmt!r}")


def to_uint8(x: torch.Tensor) -> np.ndarray:
    """(B, 3, H, W) float [0, 1] -> (B, H, W, 3) uint8 numpy (on CPU)."""
    return (x.clamp(0, 1) * 255.0 + 0.5).to(torch.uint8).permute(0, 2, 3, 1).contiguous().cpu().numpy()
