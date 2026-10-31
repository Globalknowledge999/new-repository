"""Command-line interface.

    parallaxa convert photo.jpg holiday_folder/ film.mkv -o out/ --format sbs,anaglyph
    parallaxa models
    parallaxa info
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from . import __version__
from .config import PHOTO_FORMATS, PROFILES, VIDEO_FORMATS, JobSettings


def _formats(s: str) -> tuple[str, ...]:
    fmts = tuple(f.strip().lower() for f in s.split(",") if f.strip())
    allowed = set(PHOTO_FORMATS) | set(VIDEO_FORMATS)
    bad = [f for f in fmts if f not in allowed]
    if bad:
        raise argparse.ArgumentTypeError(f"unknown format(s) {bad}; choose from {sorted(allowed)}")
    return fmts


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="parallaxa", description="Convert 2D photos and videos to stereoscopic 3D on your GPU.")
    p.add_argument("--version", action="version", version=f"parallaxa {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    c = sub.add_parser("convert", help="convert photos/videos (files or folders)")
    c.add_argument("inputs", nargs="+", help="image/video files or folders")
    c.add_argument("-o", "--output", default=None, help="output folder (default: '3d' next to each input)")
    c.add_argument("-r", "--recursive", action="store_true", help="include sub-folders")
    c.add_argument("-f", "--format", type=_formats, default=("sbs",),
                   help="comma list: sbs (default), hsbs, tb, htb, anaglyph, rgbd; photos also mpo, depth")
    c.add_argument("--profile", choices=["auto", *PROFILES], default="auto",
                   help="auto = Large model for both, higher depth resolution for photos")
    c.add_argument("--model", default=None, help="override depth model (see 'parallaxa models')")
    c.add_argument("--depth-res", type=int, default=None, help="depth inference short side in px (multiple of 14)")

    s = c.add_argument_group("3D")
    s.add_argument("--strength", type=float, default=3.0, help="total parallax, %% of eye width (default 3.0)")
    s.add_argument("--convergence", type=float, default=0.6,
                   help="0..1 depth placed at screen level; lower = more pop-out (default 0.6)")
    s.add_argument("--eye-mode", choices=["symmetric", "right", "left"], default="symmetric",
                   help="synthesise both eyes, or keep the original as the left/right eye")
    s.add_argument("--edge-dilate", type=float, default=0.002, help="foreground edge growth, fraction of width")
    s.add_argument("--fill-iterations", type=int, default=8, help="hole-fill smoothing passes")
    s.add_argument("--depth-gamma", type=float, default=1.0)
    s.add_argument("--temporal-alpha", type=float, default=0.9, help="video: depth-range smoothing 0..0.99")

    o = c.add_argument_group("output")
    o.add_argument("--height", type=int, default=0, help="per-eye output height (e.g. 1080, 2160); 0 = source")
    o.add_argument("--anaglyph", choices=["dubois", "half-color", "color", "gray"], default="dubois")
    o.add_argument("--image-ext", choices=[".jpg", ".png", ".webp"], default=".jpg")
    o.add_argument("--jpeg-quality", type=int, default=95)
    o.add_argument("--codec", choices=["auto", "h264", "hevc"], default="auto")
    o.add_argument("--encoder", choices=["auto", "nvenc", "cpu"], default="auto")
    o.add_argument("--crf", type=int, default=18, help="video quality (lower = better/larger)")
    o.add_argument("--preset", default="", help="encoder preset (x264/x265: medium.. ; nvenc: p1..p7)")
    o.add_argument("--container", choices=["auto", "mp4", "mkv", "mov"], default="auto")

    r = c.add_argument_group("runtime")
    r.add_argument("--device", default="auto", help="auto | cuda | cuda:N | cpu")
    r.add_argument("--fp32", action="store_true", help="disable half precision on GPU")
    r.add_argument("--batch-size", type=int, default=4, help="video frames per GPU batch")
    r.add_argument("--overwrite", action="store_true")
    r.add_argument("--progress", choices=["text", "json", "none"], default="text")
    r.add_argument("--timing", action="store_true", help="video: print time spent per pipeline stage")

    sub.add_parser("models", help="list depth models and their licences")
    sub.add_parser("info", help="show GPU / FFmpeg / encoder status")
    return p


def _job_from_args(a) -> JobSettings:
    job = JobSettings(device=a.device, fp16=False if a.fp32 else None, batch_size=a.batch_size, overwrite=a.overwrite)
    st = job.stereo
    st.strength, st.convergence, st.eye_mode = a.strength, a.convergence, a.eye_mode
    st.edge_dilate, st.fill_iterations, st.depth_gamma = a.edge_dilate, a.fill_iterations, a.depth_gamma
    job.depth.temporal_alpha = a.temporal_alpha
    o = job.output
    o.formats, o.height, o.anaglyph, o.image_ext, o.jpeg_quality = a.format, a.height, a.anaglyph, a.image_ext, a.jpeg_quality
    o.codec, o.encoder, o.crf, o.preset, o.container = a.codec, a.encoder, a.crf, a.preset, a.container
    return job


def _model_for(kind: str, a) -> tuple[str, int]:
    profile = a.profile if a.profile != "auto" else ("photo-quality" if kind == "photo" else "quality")
    prof = PROFILES[profile]
    return a.model or prof["model"], a.depth_res or prof["depth_res"]


def cmd_convert(a) -> int:
    from .pipeline.common import Engine, collect_inputs
    from .pipeline.photo import convert_photo
    from .pipeline.video import convert_video

    job = _job_from_args(a)
    log = (lambda m: print(json.dumps({"event": "log", "message": m}), flush=True)) if a.progress == "json" else \
          (lambda m: print(m, file=sys.stderr, flush=True))
    items = collect_inputs(a.inputs, a.recursive)
    if not items:
        log("No supported photos or videos found.")
        return 1
    engine = Engine(job, log)
    from .device import describe

    log(f"parallaxa {__version__} | device {describe(engine.device)} | {len(items)} file(s)")
    failures = 0
    t0 = time.perf_counter()
    photos = [p for p, k in items if k == "photo"]
    videos = [p for p, k in items if k == "video"]
    for i, src in enumerate(photos + videos, 1):
        kind = "photo" if src in photos else "video"
        out_dir = Path(a.output) if a.output else src.parent / "3d"
        model, res = _model_for(kind, a)
        log(f"[{i}/{len(items)}] {src}")
        try:
            if kind == "photo":
                convert_photo(engine, src, out_dir, model, res)
            else:
                convert_video(engine, src, out_dir, model, res, a.progress, a.timing)
        except KeyboardInterrupt:
            log("Cancelled.")
            return 130
        except Exception as e:  # keep going through a batch
            failures += 1
            log(f"FAILED {src.name}: {e}")
    log(f"Finished {len(items) - failures}/{len(items)} in {time.perf_counter() - t0:.1f}s")
    return 1 if failures else 0


def cmd_models(_a) -> int:
    from .depth import MODELS

    for k, v in MODELS.items():
        print(f"  {k:16s} {v}")
    print("\nProfiles:")
    for k, v in PROFILES.items():
        print(f"  {k:16s} model={v['model']} depth_res={v['depth_res']}")
    return 0


def cmd_info(_a) -> int:
    import torch

    from .io.video_io import encoder_check, ffmpeg_bin

    print(f"parallaxa {__version__}, python {sys.version.split()[0]}, torch {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}" + (f" ({torch.version.cuda})" if torch.cuda.is_available() else ""))
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            pr = torch.cuda.get_device_properties(i)
            print(f"  cuda:{i} {pr.name} {pr.total_memory / 2**30:.1f} GB")
    try:
        print(f"ffmpeg: {ffmpeg_bin()}")
        for enc in ("hevc_nvenc", "h264_nvenc", "libx265", "libx264"):
            ok, err = encoder_check(enc)
            print(f"  {enc:11s} {'ok' if ok else 'not available'}")
            if not ok and err:
                for line in err.splitlines()[:4]:
                    print(f"      {line}")
    except RuntimeError as e:
        print(e)
    return 0


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    return {"convert": cmd_convert, "models": cmd_models, "info": cmd_info}[a.command](a)


if __name__ == "__main__":
    sys.exit(main())
