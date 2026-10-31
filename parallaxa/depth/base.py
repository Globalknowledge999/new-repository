"""Depth estimator interface and depth-map normalisation."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class DepthEstimator(ABC):
    """Predicts relative *closeness* (inverse depth: larger = nearer) for a batch of frames."""

    name: str = "base"
    licence: str = ""

    def __init__(self, device: torch.device, dtype: torch.dtype):
        self.device, self.dtype = device, dtype

    @abstractmethod
    def predict(self, frames: torch.Tensor) -> torch.Tensor:
        """frames: (B, 3, H, W) float in [0, 1] on self.device. Returns (B, H, W) float32."""


class DepthNormalizer:
    """Map raw relative depth to [0, 1] (1 = nearest) using robust percentiles.

    For video, the percentile range is smoothed across frames so the 3D effect does not pump
    when a bright/near object enters the shot.
    """

    def __init__(self, low_pct: float = 2.0, high_pct: float = 98.0, temporal_alpha: float = 0.0, gamma: float = 1.0):
        self.low_q, self.high_q = low_pct / 100.0, high_pct / 100.0
        self.alpha, self.gamma = temporal_alpha, gamma
        self._lo: float | None = None
        self._hi: float | None = None

    def reset(self) -> None:
        self._lo = self._hi = None

    def __call__(self, depth: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(depth, dtype=torch.float32)
        for i, d in enumerate(depth.float()):
            sample = d.flatten()
            if sample.numel() > 200_000:  # percentiles of a strided subsample are accurate and much cheaper
                sample = sample[:: sample.numel() // 200_000 + 1]
            lo, hi = torch.quantile(sample, torch.tensor([self.low_q, self.high_q], device=d.device)).tolist()
            if self.alpha > 0 and self._lo is not None:
                lo = self.alpha * self._lo + (1 - self.alpha) * lo
                hi = self.alpha * self._hi + (1 - self.alpha) * hi
            self._lo, self._hi = lo, hi
            n = ((d - lo) / max(hi - lo, 1e-6)).clamp_(0, 1)
            if self.gamma != 1.0:
                n = n.pow_(self.gamma)
            out[i] = n
        return out
