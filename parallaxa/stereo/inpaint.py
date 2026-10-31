"""Generative gap filling: repaint disocclusions with a Stable Diffusion inpainting model.

Disocclusions are background that was hidden behind a foreground object, so a naive inpainter
tends to grow the foreground into the gap. To avoid that, pixels near a gap that are *nearer* than
the gap's background are masked too, so the model only sees background as context, the way
layered-depth inpainting methods do. Only the gap pixels are written back.

Work happens at a reduced resolution (long side `work_long_side`) in overlapping 512 px tiles. Only
tiles that contain gaps are processed, and the result is upsampled and feathered into the full
resolution view. A fixed seed keeps the result repeatable (and steadier from frame to frame on video).

Weights (plain safetensors data from Hugging Face, downloaded on first use): by default Stable
Diffusion 1.5 inpainting (CreativeML OpenRAIL-M), falling back to SDXL inpainting or SD 2 inpainting
(both Open RAIL++-M) if a repository is unavailable. All are fine for personal use.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

INPAINT_MODELS = {
    "sd15": "stable-diffusion-v1-5/stable-diffusion-inpainting",  # ~2 GB fp16, 512 px tiles
    "sdxl": "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",  # ~7 GB fp16, 1024 px tiles, slower, finer
    "sd2": "stabilityai/stable-diffusion-2-inpainting",  # may no longer be public
}
AUTO_ORDER = ["sd15", "sdxl", "sd2"]  # "auto": the first one that downloads
PROMPT = "seamless natural continuation of the surrounding background, photo, high detail"
NEGATIVE = "person, people, face, animal, object, text, watermark, logo, frame, border, blurry, artifacts, distorted"


def _round8(x: float) -> int:
    return max(8, int(round(x / 8)) * 8)


def paint_masks(holes: torch.Tensor, z: torch.Tensor, z_tol: float = 1.0, grow_px: int = 2,
                context_frac: float = 0.03) -> tuple[torch.Tensor, torch.Tensor]:
    """holes, z: (B, H, W). Returns (paint, hide), both (B, H, W) bool.

    paint: the gaps, grown slightly to cover the soft edge pixels of the forward warp.
    hide:  foreground pixels near a gap, nearer than that gap's background. They are masked from
           the model's view (but not written back) so it continues the background.
    """
    w = holes.shape[-1]
    h_f = holes[:, None].float()
    paint = F.max_pool2d(h_f, 2 * grow_px + 1, 1, grow_px) > 0
    r = max(2, int(context_frac * w))
    ry = max(1, r // 4)  # gaps are mostly vertical strips: look further sideways than up/down
    hole_z = torch.where(holes, z, torch.full_like(z, -1e9))[:, None]
    bg_z = F.max_pool2d(hole_z, (2 * ry + 1, 2 * r + 1), 1, (ry, r))
    hide = (bg_z > -1e8) & (z[:, None] > bg_z + z_tol) & ~paint
    return paint[:, 0], hide[:, 0]


def tile_origins(size: int, tile: int, overlap: int) -> list[int]:
    if size <= tile:
        return [0]
    n = math.ceil((size - overlap) / (tile - overlap))
    step = (size - tile) / (n - 1)
    return [int(round(i * step)) for i in range(n)]


def _feather(tile: int, overlap: int, device) -> torch.Tensor:
    ramp = torch.ones(tile, device=device)
    if overlap > 0:
        r = torch.linspace(0, 1, overlap + 2, device=device)[1:-1]
        ramp[:overlap], ramp[-overlap:] = r, r.flip(0)
    return (ramp[:, None] * ramp[None, :])[None, None]


def _load_one(repo: str, dtype: torch.dtype):
    import warnings

    from diffusers import AutoPipelineForInpainting
    from diffusers.utils import logging as dlogging

    kw = dict(torch_dtype=dtype, use_safetensors=True)
    if "xl" not in repo.lower():
        kw.update(safety_checker=None, requires_safety_checker=False)
    level = dlogging.get_verbosity()
    dlogging.set_verbosity_error()  # hide noisy "couldn't connect / not expected kwarg" messages
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                return AutoPipelineForInpainting.from_pretrained(repo, variant="fp16" if dtype == torch.float16 else None, **kw)
            except (OSError, ValueError) as e:  # no fp16 variant files: load the full-precision ones
                if "variant" not in str(e).lower() and "fp16" not in str(e).lower():
                    raise
                return AutoPipelineForInpainting.from_pretrained(repo, **kw)
    finally:
        dlogging.set_verbosity(level)


def load_pipeline(model: str, device: torch.device, dtype: torch.dtype, log=print):
    from diffusers import DPMSolverMultistepScheduler

    names = AUTO_ORDER if model == "auto" else [model]
    errors = []
    for name in names:
        repo = INPAINT_MODELS.get(name, name)
        try:
            pipe = _load_one(repo, dtype)
        except Exception as e:  # repository removed / gated / offline: try the next one
            errors.append(f"{repo}: {str(e).splitlines()[0][:160]}")
            if len(names) > 1:
                log(f"  {repo} unavailable, trying the next model...")
            continue
        log(f"  using {repo}")
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
        pipe.set_progress_bar_config(disable=True)
        return pipe.to(device)
    raise RuntimeError("Could not load an AI gap-filling model:\n  " + "\n  ".join(errors) +
                       "\nCheck your internet connection, choose another with --fill-model (sd15, sdxl, sd2 or a Hugging Face id),"
                       " or use --fill basic.")


def tiny_pipeline():
    """Randomly-initialised miniature inpainting pipeline (no download) for tests and offline dry runs."""
    import warnings

    from diffusers import AutoencoderKL, DPMSolverMultistepScheduler, StableDiffusionInpaintPipeline, UNet2DConditionModel
    from transformers import CLIPTextConfig, CLIPTextModel

    torch.manual_seed(0)
    unet = UNet2DConditionModel(block_out_channels=(32, 64), layers_per_block=1, sample_size=32, in_channels=9, out_channels=4,
                                down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
                                up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"), cross_attention_dim=32)
    vae = AutoencoderKL(block_out_channels=[32, 64], in_channels=3, out_channels=3, latent_channels=4,
                        down_block_types=["DownEncoderBlock2D"] * 2, up_block_types=["UpDecoderBlock2D"] * 2)
    te = CLIPTextModel(CLIPTextConfig(bos_token_id=0, eos_token_id=2, hidden_size=32, intermediate_size=37, num_attention_heads=4,
                                      num_hidden_layers=2, pad_token_id=1, vocab_size=1000))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe = StableDiffusionInpaintPipeline(vae=vae, text_encoder=te, tokenizer=None, unet=unet,
                                              scheduler=DPMSolverMultistepScheduler(steps_offset=1), safety_checker=None,
                                              feature_extractor=None, requires_safety_checker=False)
    pipe.set_progress_bar_config(disable=True)
    return pipe


class DiffusionInpainter:
    def __init__(self, device: torch.device, dtype: torch.dtype, model: str = "auto", steps: int = 14,
                 work_long_side: int = 1024, tile: int = 512, overlap: int = 96, guidance: float = 4.0,
                 seed: int = 1234, batch: int = 2, pipe=None, log=print):
        self.device, self.dtype = device, dtype
        self.model, self.steps, self.work_long_side = model, steps, work_long_side
        self.tile, self.overlap, self.guidance, self.seed, self.batch = tile, overlap, guidance, seed, batch
        self.pipe = pipe if pipe is not None else load_pipeline(model, device, dtype, log)
        self.xl = "XL" in type(self.pipe).__name__
        if self.xl:  # SDXL works at 1024 px; at 512 it degrades badly
            self.tile, self.overlap, self.work_long_side, self.batch = 1024, 160, max(work_long_side, 1536), 1
        self._embeds: tuple[torch.Tensor, torch.Tensor] | None = None
        self.stats = {"tiles": 0, "seconds": 0.0}  # of the last refine() call

    def _prompt_embeds(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._embeds is None:
            if self.pipe.tokenizer is not None:
                pos, neg = self.pipe.encode_prompt(PROMPT, self.device, 1, True, NEGATIVE)
            else:  # test pipeline without a tokenizer
                ids = torch.zeros(1, 77, dtype=torch.long, device=self.pipe.text_encoder.device)
                pos = neg = self.pipe.text_encoder(ids)[0]
            self._embeds = (pos, neg)
        return self._embeds

    @torch.inference_mode()
    def refine(self, img: torch.Tensor, holes: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """img (B, 3, H, W) [0, 1] with gaps already basic-filled; holes (B, H, W); z (B, H, W) depth incl. gaps."""
        import time

        t0 = time.perf_counter()
        self.stats = {"tiles": 0, "seconds": 0.0}
        out = img.clone()
        for b in range(img.shape[0]):
            if holes[b].any():
                out[b : b + 1] = self._one(img[b : b + 1].float(), holes[b : b + 1], z[b : b + 1].float())
        self.stats["seconds"] = time.perf_counter() - t0
        return out

    def _one(self, img: torch.Tensor, holes: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        _, _, h, w = img.shape
        scale = min(1.0, self.work_long_side / max(h, w))
        wh, ww = _round8(h * scale), _round8(w * scale)
        img_w = F.interpolate(img, size=(wh, ww), mode="area")
        holes_w = F.interpolate(holes[:, None].float(), size=(wh, ww), mode="area")[:, 0] > 0
        z_w = F.interpolate(z[:, None], size=(wh, ww), mode="area")[:, 0]
        paint_w, hide_w = paint_masks(holes_w, z_w)
        model_mask = (paint_w | hide_w).float()[:, None]

        # Pad up to at least one tile (edge-replicated, so the context stays natural).
        t = self.tile
        ph, pw = max(0, t - wh), max(0, t - ww)
        if ph or pw:
            img_w = F.pad(img_w, (0, pw, 0, ph), mode="replicate")
            model_mask = F.pad(model_mask, (0, pw, 0, ph))
            paint_w = F.pad(paint_w, (0, pw, 0, ph))
        H, W = img_w.shape[-2:]
        boxes = [(y, x) for y in tile_origins(H, t, self.overlap) for x in tile_origins(W, t, self.overlap)
                 if paint_w[0, y : y + t, x : x + t].any()]
        self.stats["tiles"] += len(boxes)
        canvas = torch.zeros_like(img_w)
        weight = torch.zeros_like(img_w[:, :1])
        feather = _feather(t, self.overlap, img.device)
        pos, neg = self._prompt_embeds() if self.pipe.tokenizer is None else (None, None)
        for i in range(0, len(boxes), self.batch):
            chunk = boxes[i : i + self.batch]
            tiles = torch.cat([img_w[:, :, y : y + t, x : x + t] for y, x in chunk])
            masks = torch.cat([model_mask[:, :, y : y + t, x : x + t] for y, x in chunk])
            gen = torch.Generator(device=self.pipe.device).manual_seed(self.seed)
            if self.pipe.tokenizer is not None:  # real pipelines (SD 1.5/2/XL) encode the prompt themselves
                prompts = dict(prompt=[PROMPT] * len(chunk), negative_prompt=[NEGATIVE] * len(chunk))
            else:
                prompts = dict(prompt_embeds=pos.repeat(len(chunk), 1, 1), negative_prompt_embeds=neg.repeat(len(chunk), 1, 1))
            res = self.pipe(**prompts, image=tiles.to(self.dtype), mask_image=masks.to(self.dtype), height=t, width=t,
                            strength=0.99 if self.xl else 1.0,
                            num_inference_steps=self.steps, guidance_scale=self.guidance, generator=gen,
                            output_type="pt").images.float().to(img.device)
            for (y, x), r in zip(chunk, res):
                canvas[:, :, y : y + t, x : x + t] += r[None] * feather
                weight[:, :, y : y + t, x : x + t] += feather
        painted = (canvas / weight.clamp_min(1e-6))[:, :, :wh, :ww]
        painted = F.interpolate(painted, size=(h, w), mode="bicubic", align_corners=False).clamp_(0, 1)

        # Feathered composite: fully replace the gaps, blend a couple of pixels around them.
        alpha = F.max_pool2d(holes[:, None].float(), 5, 1, 2)
        alpha = F.avg_pool2d(alpha, 5, 1, 2).maximum(holes[:, None].float())
        return img * (1 - alpha) + painted * alpha


def load_inpainter(name: str, device: torch.device, dtype: torch.dtype, steps: int = 14, log=print) -> DiffusionInpainter:
    if name == "debug-tiny":
        return DiffusionInpainter(device, torch.float32, name, steps=min(steps, 2), tile=64, overlap=16,
                                  work_long_side=128, pipe=tiny_pipeline().to(device))
    if device.type == "cpu":
        dtype = torch.float32  # half precision diffusion isn't supported on CPU
    return DiffusionInpainter(device, dtype, name, steps=steps, log=log)
