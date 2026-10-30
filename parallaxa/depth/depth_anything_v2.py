"""Depth Anything V2 (Yang et al., 2024) via Hugging Face transformers.

Weights are downloaded once into the Hugging Face cache. Licences: Small = Apache-2.0;
Base / Large = CC-BY-NC-4.0 (non-commercial). See MODELS.md.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .base import DepthEstimator

HF_IDS = {
    "small": "depth-anything/Depth-Anything-V2-Small-hf",
    "base": "depth-anything/Depth-Anything-V2-Base-hf",
    "large": "depth-anything/Depth-Anything-V2-Large-hf",
}
LICENCES = {"small": "Apache-2.0", "base": "CC-BY-NC-4.0", "large": "CC-BY-NC-4.0"}

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
_PATCH = 14


def inference_size(h: int, w: int, short_side: int) -> tuple[int, int]:
    """Scale so the short side is ~short_side, both sides multiples of the ViT patch size (14)."""
    scale = short_side / min(h, w)
    nh = max(_PATCH, int(round(h * scale / _PATCH)) * _PATCH)
    nw = max(_PATCH, int(round(w * scale / _PATCH)) * _PATCH)
    return nh, nw


class DepthAnythingV2(DepthEstimator):
    def __init__(self, variant: str, device: torch.device, dtype: torch.dtype, depth_res: int = 518,
                 model_id: str | None = None, model: torch.nn.Module | None = None):
        super().__init__(device, dtype)
        self.variant = variant
        self.name = f"da2-{variant}"
        self.licence = LICENCES.get(variant, "see model card")
        self.depth_res = depth_res
        if model is None:
            from transformers import AutoModelForDepthEstimation

            model = AutoModelForDepthEstimation.from_pretrained(model_id or HF_IDS[variant])
        self.model = model.to(device=device, dtype=dtype).eval()
        self._mean = _MEAN.to(device, dtype)
        self._std = _STD.to(device, dtype)

    @torch.inference_mode()
    def predict(self, frames: torch.Tensor) -> torch.Tensor:
        b, _, h, w = frames.shape
        ih, iw = inference_size(h, w, self.depth_res)
        x = frames.to(self.dtype)
        x = F.interpolate(x, size=(ih, iw), mode="bicubic", align_corners=False, antialias=(ih < h)).clamp_(0, 1)
        x = (x - self._mean) / self._std
        pred = self.model(pixel_values=x).predicted_depth  # (B, ih, iw), relative inverse depth
        pred = F.interpolate(pred.unsqueeze(1).float(), size=(h, w), mode="bilinear", align_corners=False)
        return pred[:, 0]


def tiny_random_model() -> torch.nn.Module:
    """A tiny randomly-initialised Depth Anything network (no download) for tests and offline dry runs."""
    from transformers import DepthAnythingConfig, DepthAnythingForDepthEstimation, Dinov2Config

    backbone = Dinov2Config(hidden_size=32, num_hidden_layers=4, num_attention_heads=2, intermediate_size=64,
                            patch_size=14, image_size=518, out_features=["stage1", "stage2", "stage3", "stage4"],
                            reshape_hidden_states=False)
    cfg = DepthAnythingConfig(backbone_config=backbone, neck_hidden_sizes=[8, 16, 32, 32], fusion_hidden_size=16,
                              head_hidden_size=8, reassemble_hidden_size=32, patch_size=14)
    torch.manual_seed(0)
    return DepthAnythingForDepthEstimation(cfg)
