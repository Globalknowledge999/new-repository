"""Depth model registry."""

from __future__ import annotations

import torch

from .base import DepthEstimator, DepthNormalizer

MODELS = {
    "da2-small": "Depth Anything V2 Small (Apache-2.0, ~25M params, fastest)",
    "da2-base": "Depth Anything V2 Base (CC-BY-NC-4.0, ~98M params)",
    "da2-large": "Depth Anything V2 Large (CC-BY-NC-4.0, ~335M params, best detail)",
    "vda-small": "Video Depth Anything Small (Apache-2.0, ~29M params): temporally stable video depth",
    "vda-base": "Video Depth Anything Base (CC-BY-NC-4.0, ~113M params)",
    "vda-large": "Video Depth Anything Large (CC-BY-NC-4.0, ~382M params; needs >12 GB VRAM at 518 px)",
    "debug-gradient": "No model: depth = vertical gradient (bottom near). Offline smoke tests only.",
    "debug-random": "Tiny randomly-initialised network. Offline pipeline tests only.",
    "debug-vda": "Randomly-initialised Video Depth Anything Small. Offline pipeline tests only.",
}


class GradientDepth(DepthEstimator):
    name = "debug-gradient"
    licence = "n/a"

    def predict(self, frames: torch.Tensor) -> torch.Tensor:
        b, _, h, w = frames.shape
        ramp = torch.linspace(0, 1, h, device=frames.device).view(1, h, 1)
        return ramp.expand(b, h, w).clone()


def load_depth_model(name: str, device: torch.device, dtype: torch.dtype, depth_res: int = 518) -> DepthEstimator:
    from .depth_anything_v2 import DepthAnythingV2, tiny_random_model

    if name.startswith("vda-"):
        from .video_depth_anything import VideoDepthAnything

        return VideoDepthAnything(name.split("-", 1)[1], device, dtype, depth_res)
    if name == "debug-vda":
        import warnings

        from .video_depth_anything import VideoDepthAnything

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from ..third_party.video_depth_anything.model import VideoDepthAnything as Net
        torch.manual_seed(0)
        est = VideoDepthAnything("small", device, dtype, depth_res, network=Net("vits"))
        est.name, est.licence = "debug-vda", "n/a"
        return est
    if name.startswith("da2-"):
        return DepthAnythingV2(name.split("-", 1)[1], device, dtype, depth_res)
    if name == "debug-gradient":
        return GradientDepth(device, dtype)
    if name == "debug-random":
        est = DepthAnythingV2("small", device, torch.float32, depth_res, model=tiny_random_model())
        est.name, est.licence = "debug-random", "n/a"
        return est
    if "/" in name:  # any Hugging Face depth-estimation model id with a DPT-style head
        est = DepthAnythingV2("custom", device, dtype, depth_res, model_id=name)
        est.name = name
        return est
    raise ValueError(f"Unknown depth model '{name}'. Choose from: {', '.join(MODELS)}")


def depth_stream(est: DepthEstimator):
    """push()/flush() interface for any estimator (temporal models stream natively)."""
    from .base import PerFrameStream

    return est.stream() if getattr(est, "temporal", False) else PerFrameStream(est)


__all__ = ["DepthEstimator", "DepthNormalizer", "MODELS", "depth_stream", "load_depth_model"]
