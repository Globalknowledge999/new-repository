"""Video conversion: stream frames through depth -> stereo -> one encoder per output layout.

Frames are decoded, processed and encoded in small batches on overlapping threads, so memory
use is constant regardless of film length.
"""

from __future__ import annotations

import time
from pathlib import Path

from ..config import VIDEO_FORMATS
from ..depth import DepthNormalizer
from ..io.video_io import FrameReader, FrameWriter, VideoInfo, container_for, probe
from ..progress import Progress, fmt_time
from ..stereo.formats import LAYOUT_SUFFIX, compose, to_uint8
from .common import Engine, even_size


def layout_size(fmt: str, ew: int, eh: int) -> tuple[int, int]:
    return {"sbs": (2 * ew, eh), "tb": (ew, 2 * eh), "rgbd": (2 * ew, eh)}.get(fmt, (ew, eh))


def output_path(info: VideoInfo, out_dir: Path, fmt: str, container: str) -> Path:
    return out_dir / f"{info.path.stem}{LAYOUT_SUFFIX[fmt]}.{container}"


def convert_video(engine: Engine, src: Path, out_dir: Path, model: str, depth_res: int, progress_mode: str = "text") -> list[Path]:
    job = engine.job
    out = job.output
    info = probe(src)
    container = container_for(info, out.container)
    formats = [f for f in out.formats if f in VIDEO_FORMATS]
    if not formats:
        engine.log(f"skip: none of {out.formats} apply to video ({src.name})")
        return []
    targets = {f: output_path(info, out_dir, f, container) for f in formats}
    if not job.overwrite:
        targets = {f: p for f, p in targets.items() if not p.exists()}
        if not targets:
            engine.log(f"skip (exists): {src.name}")
            return []
    if info.is_hdr:
        engine.log("warning: HDR source; this version processes in SDR (8-bit), so the output will look washed out.")

    ew, eh = even_size(info.width, info.height, out.height)
    engine.log(f"{src.name}: {info.width}x{info.height} @ {float(info.fps):.3f} fps, ~{info.frame_count} frames, "
               f"audio={','.join(info.audio_codecs) or 'none'} -> eye {ew}x{eh}, {', '.join(targets)}")

    est = engine.depth_model(model, depth_res)
    ds = job.depth
    normalizer = DepthNormalizer(ds.low_pct, ds.high_pct, ds.temporal_alpha, job.stereo.depth_gamma)

    partial = {f: p.with_name(p.stem + ".partial" + p.suffix) for f, p in targets.items()}
    writers: dict[str, FrameWriter] = {}
    reader = None
    t0 = time.perf_counter()
    try:
        for f, p in partial.items():
            w, h = layout_size(f, ew, eh)
            writers[f] = FrameWriter(p, info, w, h, f, out.codec, out.encoder, out.crf, out.preset)
        engine.log(f"encoder: {next(iter(writers.values())).encoder}")
        reader = FrameReader(info, ew, eh, batch=job.batch_size)
        prog = Progress(info.frame_count, src.name, progress_mode)
        for batch in reader:
            img = engine.to_tensor(batch)
            closeness = normalizer(est.predict(img))
            left, right = engine.renderer.render(img, closeness)
            for f, wr in writers.items():
                wr.write(to_uint8(compose(f, left, right, img, closeness, out.anaglyph)))
            prog.update(len(batch))
        prog.total = prog.total or prog.done
        prog.finish()
        for wr in writers.values():
            wr.close()
    except BaseException:
        for wr in writers.values():
            wr.abort()
        if reader:
            reader.close()
        for p in partial.values():
            p.unlink(missing_ok=True)
        raise

    for f, p in partial.items():
        p.replace(targets[f])
    engine.log(f"done  {src.name} in {fmt_time(time.perf_counter() - t0)} -> {', '.join(p.name for p in targets.values())}")
    return list(targets.values())
