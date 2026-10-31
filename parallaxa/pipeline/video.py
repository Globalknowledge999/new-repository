"""Video conversion: stream frames through depth -> stereo -> encoders, in resumable segments.

Frames are decoded, processed and encoded on overlapping threads, in small batches. Memory use is
flat for any film length. Output is written as segments (10 minutes by default) into a work folder
next to the output. When every segment is finished, they are joined without re-encoding and the
source's audio, subtitles and chapters are added. If a run stops (crash, Ctrl+C, reboot), running
the same command again continues from the last finished segment.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..config import VIDEO_FORMATS
from ..depth import DepthNormalizer, depth_stream
from ..io.video_io import FrameReader, SegmentWriter, VideoInfo, choose_encoder, container_for, mux_segments, probe
from ..progress import Progress, fmt_time
from ..stereo.formats import LAYOUT_SUFFIX, compose, to_uint8
from .common import Engine, even_size

WORK_SUFFIX = ".parallaxa-work"
# Frames decoded before the resume point so a temporal model can rebuild its context; their
# output is discarded.
WARMUP_TEMPORAL = 64


def layout_size(fmt: str, ew: int, eh: int) -> tuple[int, int]:
    return {"sbs": (2 * ew, eh), "tb": (ew, 2 * eh), "rgbd": (2 * ew, eh)}.get(fmt, (ew, eh))


def output_path(info: VideoInfo, out_dir: Path, fmt: str, container: str) -> Path:
    return out_dir / f"{info.path.stem}{LAYOUT_SUFFIX[fmt]}.{container}"


class StageTimer:
    """Wall-clock time per pipeline stage. Synchronises CUDA around each stage, so enable only to diagnose."""

    def __init__(self, device: torch.device, enabled: bool):
        self.enabled, self.cuda = enabled, device.type == "cuda"
        self.totals: dict[str, float] = {}

    @contextmanager
    def __call__(self, name: str):
        if not self.enabled:
            yield
            return
        if self.cuda:
            torch.cuda.synchronize()
        t = time.perf_counter()
        yield
        if self.cuda:
            torch.cuda.synchronize()
        self.totals[name] = self.totals.get(name, 0.0) + time.perf_counter() - t

    def report(self, frames: int) -> str:
        total = sum(self.totals.values()) or 1e-9
        lines = [f"timing over {frames} frames ({frames / total:.2f} frames/s):"]
        for name, sec in sorted(self.totals.items(), key=lambda kv: -kv[1]):
            lines.append(f"  {name:30s} {1000 * sec / max(frames, 1):8.1f} ms/frame  {100 * sec / total:5.1f}%")
        return "\n".join(lines)


class WorkDir:
    """Finished segments plus a small state file, so an interrupted conversion can resume."""

    def __init__(self, path: Path, fingerprint: str):
        self.path, self.fingerprint = path, fingerprint
        self.state_file = path / "state.json"

    def segment(self, fmt: str, index: int) -> Path:
        return self.path / f"{fmt}_{index:05d}.mp4"

    def load(self) -> dict:
        try:
            state = json.loads(self.state_file.read_text())
        except (OSError, ValueError):
            return {}
        return state if state.get("fingerprint") == self.fingerprint else {}

    def save(self, done: int, normalizer: list, stream_state: dict) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        if stream_state and not (self.path / "stream.pt").exists():
            torch.save(stream_state, self.path / "stream.pt")
        tmp = self.state_file.with_suffix(".tmp")
        tmp.write_text(json.dumps({"fingerprint": self.fingerprint, "done_segments": done, "normalizer": normalizer}))
        tmp.replace(self.state_file)

    def stream_state(self) -> dict:
        p = self.path / "stream.pt"
        return torch.load(p, map_location="cpu", weights_only=True) if p.exists() else {}

    def clear(self) -> None:
        shutil.rmtree(self.path, ignore_errors=True)


def _fingerprint(info: VideoInfo, job, model: str, depth_res: int, formats: list[str], seg_len: int, eye: tuple[int, int]) -> str:
    st = info.path.stat()
    key = {"src": str(info.path.resolve()), "size": st.st_size, "mtime": int(st.st_mtime), "model": model,
           "depth_res": depth_res, "formats": formats, "seg_len": seg_len, "eye": eye,
           "stereo": asdict(job.stereo), "depth": asdict(job.depth), "fill": [job.fill, job.fill_model, job.fill_steps],
           "out": {k: v for k, v in asdict(job.output).items() if k not in ("formats", "container")}}
    return hashlib.sha1(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()


def convert_video(engine: Engine, src: Path, out_dir: Path, model: str, depth_res: int, progress_mode: str = "text",
                  timing: bool = False, segment_minutes: float = 10.0) -> list[Path]:
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
    formats = list(targets)
    if info.is_hdr:
        engine.log("warning: HDR source; this version processes in SDR (8-bit), so the output will look washed out.")

    ew, eh = even_size(info.width, info.height, out.height)
    engine.log(f"{src.name}: {info.width}x{info.height} @ {float(info.fps):.3f} fps, ~{info.frame_count} frames, "
               f"audio={','.join(info.audio_codecs) or 'none'} -> eye {ew}x{eh}, {', '.join(formats)}")

    if engine.device.type == "cuda":
        # Thousands of frames share one size, so cuDNN's one-off autotuning pays off. (Photos skip it:
        # each run is a new process and would pay the tuning cost every time without benefit.)
        torch.backends.cudnn.benchmark = True
    est = engine.depth_model(model, depth_res)
    inpainter = engine.inpainter("video")
    if inpainter is not None:
        engine.log("note: AI gap filling on video takes seconds per frame and may flicker; it suits short clips.")
    temporal = getattr(est, "temporal", False)
    stream = depth_stream(est)
    ds = job.depth
    normalizer = DepthNormalizer(ds.low_pct, ds.high_pct, ds.temporal_alpha, job.stereo.depth_gamma)

    seg_len = max(1, int(round(segment_minutes * 60 * float(info.fps))))
    work = WorkDir(out_dir / f"{src.stem}{WORK_SUFFIX}", _fingerprint(info, job, model, depth_res, formats, seg_len, (ew, eh)))
    if job.overwrite:
        work.clear()
    state = work.load()
    if not state and work.path.exists():
        work.clear()  # settings changed since the interrupted run: start again
    done_segments = int(state.get("done_segments", 0))
    start = done_segments * seg_len
    first = 0
    if start:
        normalizer.restore(state["normalizer"])
        first = max(0, start - WARMUP_TEMPORAL) if temporal else start
        if first:
            stream.seed(work.stream_state())
        engine.log(f"resuming at frame {start} ({done_segments} segment(s) already done)")

    codec, encoder_name = choose_encoder(out.codec, out.encoder, max(layout_size(f, ew, eh)[0] for f in formats))
    engine.log(f"encoder: {encoder_name}")
    writers: dict[str, SegmentWriter] = {}
    reader = None
    seg_index = done_segments
    t0 = time.perf_counter()
    tm = StageTimer(engine.device, enabled=timing)
    prog = Progress(info.frame_count, src.name, progress_mode, initial=start)
    pending: deque[np.ndarray] = deque()  # decoded frames waiting for their depth
    pos = first  # absolute index of the next frame to receive depth

    def open_segment(index: int) -> None:
        for f in formats:
            w, h = layout_size(f, ew, eh)
            p = work.segment(f, index)
            # All layouts share one codec so the joined files are consistent.
            writers[f] = SegmentWriter(p.with_suffix(".partial.mp4"), info, w, h, f, codec, out.encoder, out.crf, out.preset)

    def close_segment(index: int) -> None:
        for f, wr in writers.items():
            wr.close()
            wr.out_path.replace(work.segment(f, index))
        writers.clear()
        work.save(index + 1, normalizer.state(), stream.state())

    def render_and_write(frames_u8: np.ndarray, depth: torch.Tensor) -> None:
        nonlocal pos, seg_index
        with tm("upload to GPU"):
            img = engine.to_tensor(frames_u8)
        if depth.shape[-2:] != img.shape[-2:]:
            depth = F.interpolate(depth[:, None].float(), size=img.shape[-2:], mode="bilinear", align_corners=True)[:, 0]
        with tm("depth normalise"):
            closeness = normalizer(depth)
        left, right = engine.renderer.render(img, closeness, tm if timing else None, inpainter=inpainter)
        if not writers:
            open_segment(seg_index)
        for f, wr in writers.items():
            with tm("pack + download"):
                packed = to_uint8(compose(f, left, right, img, closeness, out.anaglyph))
            with tm("encode (waiting for ffmpeg)"):
                wr.write(packed)
        pos += len(frames_u8)
        prog.update(len(frames_u8))
        if pos == (seg_index + 1) * seg_len:
            with tm("encode (waiting for ffmpeg)"):
                close_segment(seg_index)
            seg_index += 1

    def consume(depth: torch.Tensor) -> None:
        """Pair finished depth maps with their frames; render in batches that never cross a segment boundary."""
        nonlocal pos
        i = 0
        while i < depth.shape[0]:
            if pos < start:  # warm-up after resume: depth only, no output
                skip = min(start - pos, depth.shape[0] - i)
                for _ in range(skip):
                    pending.popleft()
                pos += skip
                i += skip
                continue
            n = min(job.batch_size, depth.shape[0] - i, (seg_index + 1) * seg_len - pos)
            frames_u8 = np.stack([pending.popleft() for _ in range(n)])
            render_and_write(frames_u8, depth[i : i + n])
            i += n

    try:
        reader = FrameReader(info, ew, eh, batch=job.batch_size, start_frame=first)
        frames = iter(reader)
        while True:
            with tm("decode (waiting for ffmpeg)"):
                batch = next(frames, None)
            if batch is None:
                break
            pending.extend(batch)
            with tm("depth model"):
                depth = stream.push(engine.to_tensor(batch))
            consume(depth)
        with tm("depth model"):
            depth = stream.flush()
        consume(depth)
        if writers:
            with tm("encode (waiting for ffmpeg)"):
                close_segment(seg_index)
            seg_index += 1
        if seg_index == 0:
            raise RuntimeError("no frames could be decoded from this video")
        prog.total = pos
        prog.finish()
    except BaseException:
        for wr in writers.values():
            wr.abort()
            wr.out_path.unlink(missing_ok=True)
        if reader:
            reader.close()
        if progress_mode == "text":
            sys.stderr.write("\n")  # end the progress line
        kept = work.load().get("done_segments", 0) if work.path.exists() else 0
        if kept:
            engine.log(f"stopped at frame {pos}. Frames 0-{kept * seg_len - 1} are saved ({kept} segment(s) in {work.path.name}); "
                       f"the final video is only created when the whole file is done. Run the same command again to resume.")
        raise

    if timing:
        engine.log(tm.report(pos - start))
    with tm("join + add audio"):
        for f, target in targets.items():
            segments = [work.segment(f, i) for i in range(seg_index)]
            partial = target.with_name(target.stem + ".partial" + target.suffix)
            mux_segments(segments, info, partial, f, codec)
            partial.replace(target)
    work.clear()
    engine.log(f"done  {src.name} in {fmt_time(time.perf_counter() - t0)} -> {', '.join(p.name for p in targets.values())}")
    return list(targets.values())
