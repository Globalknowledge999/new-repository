"""Live preview: depth is estimated once per image; re-rendering for new 3D settings takes milliseconds."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PySide6.QtCore import QObject, Signal, Slot

from ..config import PROFILES, StereoSettings
from ..depth import DepthNormalizer, load_depth_model
from ..device import default_dtype, resolve_device
from ..io.image_io import load_image
from ..io.video_io import _NO_WINDOW, ffmpeg_bin, probe
from ..stereo import StereoRenderer
from ..stereo.formats import anaglyph, to_uint8
from .options import ConvertOptions

MODES = {"anaglyph": "Red/cyan anaglyph", "wiggle": "Wobble (no glasses)", "cross": "Cross-eye (R | L)",
         "sbs": "Side by side (L | R)", "depth": "Depth map"}


def photo_model_for(opts: ConvertOptions, kind: str) -> tuple[str, int]:
    """The same model/resolution a conversion with these options would use (video models fall back for stills)."""
    profile = opts.profile if opts.profile != "auto" else ("photo-quality" if kind == "photo" else "fast")
    prof = PROFILES[profile]
    model = opts.model or prof["model"]
    if model.startswith("vda-"):
        model = "da2-" + model.split("-", 1)[1]  # preview a single frame with the same-size image model
    elif model == "debug-vda":
        model = "debug-random"
    return model, prof["depth_res"]


def video_frame(path: Path, at: float = 0.1) -> np.ndarray:
    """One RGB frame from `at` (fraction) of the way into a video."""
    info = probe(path)
    t = max(0.0, info.duration * at)
    cmd = [ffmpeg_bin(), "-nostdin", "-hide_banner", "-loglevel", "error", "-ss", f"{t:.3f}", "-i", str(path),
           "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    out = subprocess.run(cmd, capture_output=True, creationflags=_NO_WINDOW)
    w, h = info.width, info.height
    if out.returncode != 0 or len(out.stdout) < w * h * 3:
        raise RuntimeError(f"could not read a frame from {path.name}: {out.stderr.decode(errors='replace').strip()}")
    return np.frombuffer(out.stdout[: w * h * 3], np.uint8).reshape(h, w, 3).copy()


class PreviewEngine:
    """Synchronous core (testable without threads)."""

    def __init__(self, device: str = "auto", max_width: int = 1280):
        self.device = resolve_device(device)
        self.dtype = default_dtype(self.device)
        self.max_width = max_width
        self._models: dict[str, object] = {}
        self._inpainters: dict[str, object] = {}
        self._key = None
        self.img: torch.Tensor | None = None
        self.raw_depth: torch.Tensor | None = None

    def wants_ai(self, opts: ConvertOptions, kind: str) -> bool:
        return opts.fill == "ai" or (opts.fill == "auto" and kind == "photo")

    def inpainter(self, opts: ConvertOptions):
        key = f"inpaint:{opts.fill_model}"
        if key not in self._inpainters:
            from ..stereo.inpaint import load_inpainter

            self._inpainters.clear()
            self._inpainters[key] = load_inpainter(opts.fill_model, self.device, self.dtype)
        return self._inpainters[key]

    def load(self, path: Path, kind: str, opts: ConvertOptions) -> None:
        model, res = photo_model_for(opts, kind)
        key = (str(path), model, res)
        if key == self._key:
            return
        rgb = load_image(path)[0] if kind == "photo" else video_frame(path)
        h, w = rgb.shape[:2]
        t = torch.from_numpy(rgb).to(self.device).permute(2, 0, 1)[None].float() / 255
        if w > self.max_width:
            nh = int(round(h * self.max_width / w / 2)) * 2
            t = F.interpolate(t, size=(nh, self.max_width), mode="area")
        if model not in self._models:
            self._models.clear()  # keep only one preview model in VRAM
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            self._models[model] = load_depth_model(model, self.device, self.dtype, res)
        est = self._models[model]
        if hasattr(est, "depth_res"):
            est.depth_res = res
        with torch.inference_mode():
            self.raw_depth = est.predict(t)
        self.img, self._key = t, key

    @torch.inference_mode()
    def render(self, opts: ConvertOptions, mode: str, ai: bool = False) -> list[np.ndarray]:
        """Returns one image, or two (left, right) for the wobble mode. ai=True adds generative gap filling."""
        assert self.img is not None and self.raw_depth is not None
        closeness = DepthNormalizer()(self.raw_depth)
        if mode == "depth":
            return [to_uint8(closeness.unsqueeze(1).expand(-1, 3, -1, -1))[0]]
        stereo = StereoSettings(strength=opts.strength, convergence=opts.convergence, eye_mode=opts.eye_mode)
        left, right = StereoRenderer(stereo).render(self.img, closeness, inpainter=self.inpainter(opts) if ai else None)
        if mode == "anaglyph":
            return [to_uint8(anaglyph(left, right, opts.anaglyph))[0]]
        if mode == "sbs":
            return [to_uint8(torch.cat([left, right], 3))[0]]
        if mode == "cross":
            return [to_uint8(torch.cat([right, left], 3))[0]]
        return [to_uint8(left)[0], to_uint8(right)[0]]  # wiggle


class PreviewWorker(QObject):
    """Runs PreviewEngine on a background thread. Requests are coalesced: only the newest one runs."""

    rendered = Signal(list, str)  # images, caption
    status = Signal(str)
    failed = Signal(str)

    def __init__(self, device: str = "auto"):
        super().__init__()
        self.device = device
        self.engine: PreviewEngine | None = None
        self._pending = None

    @Slot(object)
    def request(self, req) -> None:
        """req = (path, kind, options, mode). Called through a queued connection."""
        self._pending = req
        # Defer so a burst of slider moves collapses into one render.
        from PySide6.QtCore import QTimer

        QTimer.singleShot(0, self._run)

    def _run(self) -> None:
        req, self._pending = self._pending, None
        if req is None:
            return
        path, kind, opts, mode = req
        try:
            if self.engine is None:
                self.status.emit("Starting preview engine...")
                self.engine = PreviewEngine(self.device)
            model, _ = photo_model_for(opts, kind)
            if self.engine._key is None or self.engine._key[:2] != (str(path), model):
                self.status.emit(f"Estimating depth with {model} (the first time also loads the model)...")
            self.engine.load(Path(path), kind, opts)
            if self._pending is not None:  # newer request arrived while estimating depth
                return self._run()
            caption = f"{Path(path).name}  |  {MODES[mode]}  |  strength {opts.strength:g}%  convergence {opts.convergence:g}"
            ai = mode != "depth" and self.engine.wants_ai(opts, kind)
            self.rendered.emit(self.engine.render(opts, mode), caption + ("  |  AI gap filling..." if ai else ""))
            if ai:
                # Show the instant basic result first, then the AI-filled one unless a newer request came in.
                from PySide6.QtCore import QCoreApplication

                QCoreApplication.processEvents()
                if self._pending is not None:
                    return self._run()
                if "inpaint:" + opts.fill_model not in self.engine._inpainters:
                    self.status.emit("Loading the AI gap-filling model (the first time downloads 2-7 GB)...")
                self.rendered.emit(self.engine.render(opts, mode, ai=True), caption + "  |  AI gap filling")
        except Exception as e:  # show, don't crash the GUI
            self.failed.emit(f"Preview failed: {e}")
