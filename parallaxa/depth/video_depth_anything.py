"""Video Depth Anything (Chen et al., CVPR 2025) with streaming, constant-memory inference.

The network sees 32-frame windows. Consecutive windows start 22 frames apart, so they overlap.
Each window after the first is built from:

    position 0      the first frame of the video      (global scale anchor)
    position 1      the frame at window start - 10    (local scale anchor, seen by the previous window)
    positions 2..9  the last 8 frames of the previous window (overlap)
    positions 10..31  22 new frames

The relative depth of each window is mapped onto the previous one with a least-squares scale and
shift fitted on the two anchor frames. The 8 overlapping frames are cross-faded from the
old window's prediction to the new one. This is the schedule the published checkpoints were
trained and evaluated with. Here it is implemented as a stream: frames go in, and depth maps come
out as soon as no later window can change them (at most ~40 frames of latency). Memory therefore
does not grow with video length.

Weights: Small = Apache-2.0; Base / Large = CC-BY-NC-4.0.
"""

from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn.functional as F

from .base import DepthEstimator
from .depth_anything_v2 import _MEAN, _STD, inference_size

WINDOW = 32
OVERLAP = 10
INTERP = 8
STEP = WINDOW - OVERLAP  # 22 new frames per window
KEY_GLOBAL, KEY_LOCAL = 0, 12  # positions in the previous window reused as anchors

HF_WEIGHTS = {
    "small": ("depth-anything/Video-Depth-Anything-Small", "video_depth_anything_vits.pth", "vits"),
    "base": ("depth-anything/Video-Depth-Anything-Base", "video_depth_anything_vitb.pth", "vitb"),
    "large": ("depth-anything/Video-Depth-Anything-Large", "video_depth_anything_vitl.pth", "vitl"),
}
LICENCES = {"small": "Apache-2.0", "base": "CC-BY-NC-4.0", "large": "CC-BY-NC-4.0"}


def load_vda_network(variant: str) -> torch.nn.Module:
    import warnings

    from huggingface_hub import hf_hub_download

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # upstream warns when the optional xFormers package is absent
        from ..third_party.video_depth_anything.model import VideoDepthAnything

    repo, filename, encoder = HF_WEIGHTS[variant]
    net = VideoDepthAnything(encoder)
    state = torch.load(hf_hub_download(repo, filename), map_location="cpu", weights_only=True)
    net.load_state_dict(state, strict=True)
    return net


def fit_scale_shift(pred: torch.Tensor, target: torch.Tensor) -> tuple[float, float]:
    """Least-squares s, t minimising |s * pred + t - target|^2 over all pixels."""
    p, q = pred.double().flatten(), target.double().flatten()
    n = p.numel()
    a00, a01, b0, b1 = (p * p).sum(), p.sum(), (p * q).sum(), q.sum()
    det = a00 * n - a01 * a01
    if det.abs() < 1e-12:
        return 1.0, 0.0
    return float((n * b0 - a01 * b1) / det), float((a00 * b1 - a01 * b0) / det)


class VideoDepthStream:
    """Feed frames with push(); receive finished depth maps (network resolution) in order."""

    def __init__(self, network, device: torch.device, dtype: torch.dtype, depth_res: int = 518):
        self.net, self.device, self.depth_res = network, device, depth_res
        self.autocast = device.type == "cuda" and dtype == torch.float16
        self._mean, self._std = _MEAN.to(device), _STD.to(device)
        self.reset()

    def reset(self) -> None:
        self.size: tuple[int, int] | None = None
        self.inputs: list[torch.Tensor] = []  # preprocessed frames not yet consumed, from index self.buf_start
        self.buf_start = 0
        self.received = 0
        self.emitted = 0
        self.next_window = 0  # start frame of the next window to run
        self.anchor_input: torch.Tensor | None = None  # frame 0
        self.local_input: torch.Tensor | None = None  # previous window's position 12
        self.ref_global: torch.Tensor | None = None  # aligned depth of frame 0
        self.ref_local: torch.Tensor | None = None
        self.tail: torch.Tensor | None = None  # aligned depth of the last 8 frames, awaiting cross-fade
        self.resumed = False

    # -- preprocessing ------------------------------------------------------------------------
    def _prep(self, frames: torch.Tensor) -> torch.Tensor:
        h, w = frames.shape[-2:]
        if self.size is None:
            res = self.depth_res
            ratio = max(h, w) / min(h, w)
            if ratio > 1.78:  # very wide (e.g. 2.39:1 cinema): keep the token count of a 16:9 frame
                res = max(14, int(round(res * 1.777 / ratio / 14)) * 14)
            self.size = inference_size(h, w, res)
        ih, iw = self.size
        x = F.interpolate(frames.float(), size=(ih, iw), mode="bicubic", align_corners=False, antialias=ih < h).clamp_(0, 1)
        return (x - self._mean) / self._std

    @torch.inference_mode()
    def _run(self, window: list[torch.Tensor]) -> torch.Tensor:
        x = torch.stack(window).unsqueeze(0)  # (1, 32, 3, h, w)
        ctx = torch.autocast("cuda", dtype=torch.float16) if self.autocast else nullcontext()
        with ctx:
            depth = self.net(x)
        return depth[0].float()  # (32, h, w)

    # -- streaming ----------------------------------------------------------------------------
    def push(self, frames: torch.Tensor) -> torch.Tensor:
        """frames (B, 3, H, W) float [0, 1]. Returns finished depth maps (n, h, w), n >= 0."""
        for f in self._prep(frames):
            self.inputs.append(f)
        self.received += frames.shape[0]
        out = []
        while self._have(self.next_window + WINDOW - 1):
            out.append(self._step(pad_to=None))
        return self._finish(out)

    def flush(self) -> torch.Tensor:
        """End of video: run the remaining (padded) windows and return all outstanding depth."""
        out = []
        while self.next_window < self.received:
            out.append(self._step(pad_to=self.received))
        if self.tail is not None:
            emitted_until = self.next_window - STEP + WINDOW - INTERP  # first frame index held in tail
            keep = max(0, min(INTERP, self.received - emitted_until))
            out.append(self.tail[:keep])
            self.tail = None
        return self._finish(out)

    # -- resume ---------------------------------------------------------------------------------
    def state(self) -> dict:
        """What a resumed run needs to keep the same depth scale: the global anchor frame and its depth."""
        if self.anchor_input is None:
            return {}
        return {"anchor_input": self.anchor_input.cpu(), "ref_global": self.ref_global.cpu()}

    def seed(self, state: dict) -> None:
        """Resume mid-video. The first window then carries the saved anchor at position 0 and is
        scale-aligned to it alone; later windows follow the normal schedule. Callers should discard
        the first ~2 windows of output (warm-up), whose position-to-frame mapping is approximate."""
        if state:
            self.anchor_input = state["anchor_input"].to(self.device)
            self.ref_global = state["ref_global"].to(self.device)
            self.resumed = True

    def _have(self, idx: int) -> bool:
        return idx < self.received

    def _frame(self, idx: int, pad_to: int | None) -> torch.Tensor:
        if pad_to is not None:
            idx = min(idx, pad_to - 1)  # repeat the last frame past the end, as upstream does
        return self.inputs[idx - self.buf_start]

    def _step(self, pad_to: int | None) -> torch.Tensor:
        s = self.next_window
        first = self.anchor_input is None
        if self.resumed and self.tail is None:
            window = [self.anchor_input] + [self._frame(s + i, pad_to) for i in range(1, WINDOW)]
            depth = self._run(window)
            sc, sh = fit_scale_shift(depth[0], self.ref_global)
            aligned = (depth * sc + sh).clamp_(min=0)
            emit = torch.cat([aligned[1:2], aligned[1 : WINDOW - INTERP]])  # frame s ~ frame s+1 (warm-up only)
            self.ref_local, self.local_input, self.tail = aligned[KEY_LOCAL], window[KEY_LOCAL], aligned[WINDOW - INTERP :]
            self._advance(s)
            return emit
        if first:
            window = [self._frame(s + i, pad_to) for i in range(WINDOW)]
        else:
            window = [self.anchor_input, self.local_input] + [self._frame(s + i, pad_to) for i in range(2, WINDOW)]
        depth = self._run(window)

        if first:
            aligned = depth
            self.anchor_input = window[KEY_GLOBAL]
            self.ref_global = aligned[KEY_GLOBAL]
            emit = aligned[: WINDOW - INTERP]
        else:
            sc, sh = fit_scale_shift(torch.stack([depth[0], depth[1]]), torch.stack([self.ref_global, self.ref_local]))
            aligned = (depth * sc + sh).clamp_(min=0)
            w = torch.linspace(0, 1, INTERP, device=depth.device).view(-1, 1, 1)
            blended = self.tail * (1 - w) + aligned[2:OVERLAP] * w
            emit = torch.cat([blended, aligned[OVERLAP : WINDOW - INTERP]])
        self.ref_local = aligned[KEY_LOCAL]
        self.local_input = window[KEY_LOCAL]
        self.tail = aligned[WINDOW - INTERP :]

        self._advance(s)
        return emit

    def _advance(self, s: int) -> None:
        # Frames before the next window's overlap are no longer needed (always keep the newest frame:
        # a final, mostly padded window repeats it).
        self.next_window = s + STEP
        drop = min(self.next_window + 2, self.received - 1) - self.buf_start
        if drop > 0:
            del self.inputs[:drop]
            self.buf_start += drop

    def _finish(self, parts: list[torch.Tensor]) -> torch.Tensor:
        parts = [p for p in parts if p.numel()]
        out = torch.cat(parts) if parts else torch.empty(0, device=self.device)
        out = out[: self.received - self.emitted]  # padded windows never produce frames past the end
        self.emitted += out.shape[0]
        return out


class VideoDepthAnything(DepthEstimator):
    """Temporal depth estimator. Use stream() for video; predict() on single images is not supported."""

    temporal = True

    def __init__(self, variant: str, device: torch.device, dtype: torch.dtype, depth_res: int = 518,
                 network: torch.nn.Module | None = None):
        super().__init__(device, dtype)
        self.variant = variant
        self.name = f"vda-{variant}"
        self.licence = LICENCES.get(variant, "n/a")
        self.depth_res = depth_res
        # Weights stay float32; fp16 is applied with autocast (the head keeps its last conv in fp32).
        self.net = (network if network is not None else load_vda_network(variant)).to(device).eval()

    def stream(self) -> VideoDepthStream:
        return VideoDepthStream(self.net, self.device, self.dtype, self.depth_res)

    def predict(self, frames: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Video Depth Anything works on video streams; use a da2-* model for photos.")
