"""Photo conversion: one image -> depth -> stereo pair -> requested output files."""

from __future__ import annotations

import time
from pathlib import Path

import torch.nn.functional as F

from ..config import PHOTO_FORMATS
from ..depth import DepthNormalizer
from ..io.image_io import load_image, save_depth_png16, save_image, save_mpo
from ..stereo.formats import LAYOUT_SUFFIX, compose, to_uint8
from .common import Engine, even_size


def output_path(src: Path, out_dir: Path, fmt: str, image_ext: str) -> Path:
    ext = {"mpo": ".mpo", "depth": ".png", "rgbd": ".png"}.get(fmt, image_ext)
    return out_dir / f"{src.stem}{LAYOUT_SUFFIX[fmt]}{ext}"


def convert_photo(engine: Engine, src: Path, out_dir: Path, model: str, depth_res: int) -> list[Path]:
    job = engine.job
    out = job.output
    formats = [f for f in out.formats if f in PHOTO_FORMATS]
    targets = {f: output_path(src, out_dir, f, out.image_ext) for f in formats}
    if not job.overwrite and targets and all(p.exists() for p in targets.values()):
        engine.log(f"skip (exists): {src.name}")
        return []

    t0 = time.perf_counter()
    rgb, meta = load_image(src)
    img = engine.to_tensor(rgb[None])
    h, w = rgb.shape[:2]
    ew, eh = even_size(w, h, out.height)
    if (ew, eh) != (w, h):
        img = F.interpolate(img, size=(eh, ew), mode="bicubic", align_corners=False, antialias=eh < h).clamp_(0, 1)

    est = engine.depth_model(model, depth_res)
    ds = job.depth
    closeness = DepthNormalizer(ds.low_pct, ds.high_pct, 0.0, job.stereo.depth_gamma)(est.predict(img))
    left, right = engine.renderer.render(img, closeness)

    written = []
    icc = meta.get("icc_profile")
    for fmt, path in targets.items():
        if path.exists() and not job.overwrite:
            continue
        if fmt == "mpo":
            l8, r8 = to_uint8(left)[0], to_uint8(right)[0]
            save_mpo(l8, r8, path, out.jpeg_quality)
        elif fmt == "depth":
            save_depth_png16(closeness[0].cpu().numpy(), path)
        else:
            arr = to_uint8(compose(fmt, left, right, img, closeness, out.anaglyph))[0]
            save_image(arr, path, out.jpeg_quality, icc)
        written.append(path)
    engine.log(f"done  {src.name} -> {', '.join(p.name for p in written)}  ({time.perf_counter() - t0:.2f}s)")
    return written
