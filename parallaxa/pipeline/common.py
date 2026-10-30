"""Shared engine: device, cached depth models, renderer, file discovery."""

from __future__ import annotations

from pathlib import Path

import torch

from ..config import PHOTO_EXTS, VIDEO_EXTS, JobSettings
from ..depth import DepthEstimator, load_depth_model
from ..device import default_dtype, resolve_device
from ..stereo import StereoRenderer


class Engine:
    """Holds the device and lazily-loaded depth models so a batch reuses them."""

    def __init__(self, job: JobSettings, log=print):
        self.job = job
        self.device = resolve_device(job.device)
        self.dtype = default_dtype(self.device, job.fp16)
        self.renderer = StereoRenderer(job.stereo)
        self.log = log
        self._models: dict[str, DepthEstimator] = {}

    def depth_model(self, name: str, depth_res: int) -> DepthEstimator:
        if name not in self._models:
            self.log(f"Loading depth model {name} on {self.device} ...")
            self._models[name] = load_depth_model(name, self.device, self.dtype, depth_res)
        model = self._models[name]
        if hasattr(model, "depth_res"):
            model.depth_res = depth_res
        return model

    def to_tensor(self, frames_u8) -> torch.Tensor:
        """(B, H, W, 3) uint8 numpy -> (B, 3, H, W) float [0, 1] on the device."""
        t = torch.from_numpy(frames_u8)
        if self.device.type == "cuda":
            t = t.pin_memory().to(self.device, non_blocking=True)
        return t.permute(0, 3, 1, 2).float().div_(255.0)


def even_size(w: int, h: int, target_h: int = 0) -> tuple[int, int]:
    """Per-eye frame size, optionally rescaled to target_h, rounded to even numbers (needed for yuv420)."""
    if target_h and target_h != h:
        w = w * target_h / h
        h = target_h
    return max(2, int(round(w / 2)) * 2), max(2, int(round(h / 2)) * 2)


def collect_inputs(paths: list[str], recursive: bool = False) -> list[tuple[Path, str]]:
    """Expand files/folders into [(path, 'photo'|'video')], sorted, skipping our own outputs."""
    found: list[tuple[Path, str]] = []
    for p in map(Path, paths):
        if p.is_dir():
            files = sorted(p.rglob("*") if recursive else p.iterdir())
        elif p.exists():
            files = [p]
        else:
            raise FileNotFoundError(p)
        for f in files:
            if not f.is_file() or any(tag in f.stem for tag in ("_3D_", "_RGBD", "_DEPTH", ".partial")) or f.stem.endswith("_3D"):
                continue
            ext = f.suffix.lower()
            if ext in PHOTO_EXTS:
                found.append((f, "photo"))
            elif ext in VIDEO_EXTS:
                found.append((f, "video"))
    return found
