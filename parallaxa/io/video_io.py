"""FFmpeg-based video probing, decoding to raw RGB frames and encoding with audio passthrough."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from queue import Queue

import numpy as np

# Audio codecs each container accepts as a straight copy.
_AUDIO_COPY_OK = {
    "mp4": {"aac", "mp3", "ac3", "eac3", "alac", "opus", "flac", "mp2"},
    "mov": {"aac", "mp3", "ac3", "eac3", "alac", "pcm_s16le", "pcm_s24le", "pcm_s32le", "pcm_f32le", "pcm_s16be", "pcm_s24be"},
    "mkv": None,  # Matroska takes anything
}

# Matroska StereoMode tag and x264 frame-packing SEI per output layout.
_MKV_STEREO_MODE = {"sbs": "left_right", "hsbs": "left_right", "tb": "top_bottom", "htb": "top_bottom", "anaglyph": "anaglyph_cyan_red"}
_X264_FRAME_PACKING = {"sbs": 3, "hsbs": 3, "tb": 4, "htb": 4}

_NO_WINDOW = 0x08000000 if os.name == "nt" else 0  # CREATE_NO_WINDOW: no console pop-ups on Windows


def ffmpeg_bin(name: str = "ffmpeg") -> str:
    """Find ffmpeg/ffprobe: $PARALLAXA_FFMPEG_DIR, a bundled ./ffmpeg/bin, then PATH."""
    exe = name + (".exe" if os.name == "nt" else "")
    candidates = []
    if env := os.environ.get("PARALLAXA_FFMPEG_DIR"):
        candidates.append(Path(env) / exe)
    candidates.append(Path(__file__).resolve().parents[2] / "ffmpeg" / "bin" / exe)
    for c in candidates:
        if c.is_file():
            return str(c)
    found = shutil.which(name)
    if not found:
        raise RuntimeError(f"{name} not found. Install FFmpeg and add it to PATH (see README).")
    return found


@dataclass
class VideoInfo:
    path: Path
    width: int  # display width after rotation
    height: int
    fps: Fraction
    frame_count: int  # best estimate; 0 if unknown
    duration: float
    pix_fmt: str
    color_space: str | None
    color_primaries: str | None
    color_transfer: str | None
    color_range: str | None
    audio_codecs: list[str]
    subtitle_codecs: list[str]

    @property
    def is_hdr(self) -> bool:
        return self.color_transfer in ("smpte2084", "arib-std-b67")

    @property
    def matrix(self) -> str:
        """Colour matrix used for YUV<->RGB conversion (guessed from size if untagged)."""
        cs = self.color_space or ""
        if cs.startswith("bt2020"):
            return "bt2020"
        if cs in ("bt709", "smpte240m", "fcc"):
            return cs
        if cs in ("bt470bg", "smpte170m"):
            return "bt601"
        return "bt709" if self.height >= 720 else "bt601"


def _fraction(s: str | None) -> Fraction:
    try:
        f = Fraction(s or "0")
        return f if f > 0 else Fraction(0)
    except (ValueError, ZeroDivisionError):
        return Fraction(0)


def probe(path: str | Path) -> VideoInfo:
    path = Path(path)
    cmd = [ffmpeg_bin("ffprobe"), "-v", "error", "-print_format", "json", "-show_streams", "-show_format", str(path)]
    out = subprocess.run(cmd, capture_output=True, text=True, creationflags=_NO_WINDOW)
    if out.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {path}: {out.stderr.strip()}")
    data = json.loads(out.stdout)
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video" and not s.get("disposition", {}).get("attached_pic")), None)
    if video is None:
        raise RuntimeError(f"No video stream in {path}")

    w, h = int(video["width"]), int(video["height"])
    rotation = 0
    for sd in video.get("side_data_list", []) or []:
        if "rotation" in sd:
            rotation = int(sd["rotation"])
    rotation = rotation or int(video.get("tags", {}).get("rotate", 0) or 0)
    if abs(rotation) % 180 == 90:  # ffmpeg auto-rotates on decode
        w, h = h, w

    # Prefer the nominal rate unless the stream is clearly variable-rate.
    r, avg = _fraction(video.get("r_frame_rate")), _fraction(video.get("avg_frame_rate"))
    fps = r if r and (not avg or abs(float(r - avg)) / float(r) < 0.01) else (avg or r or Fraction(30))
    if fps > 240:
        fps = avg or Fraction(30)

    duration = float(video.get("duration") or data.get("format", {}).get("duration") or 0.0)
    frames = int(video.get("nb_frames") or 0)
    if not frames and duration:
        frames = int(round(duration * float(fps)))

    return VideoInfo(
        path=path,
        width=w,
        height=h,
        fps=fps,
        frame_count=frames,
        duration=duration,
        pix_fmt=video.get("pix_fmt", ""),
        color_space=video.get("color_space"),
        color_primaries=video.get("color_primaries"),
        color_transfer=video.get("color_transfer"),
        color_range=video.get("color_range"),
        audio_codecs=[s.get("codec_name", "") for s in streams if s.get("codec_type") == "audio"],
        subtitle_codecs=[s.get("codec_name", "") for s in streams if s.get("codec_type") == "subtitle"],
    )


class FrameReader:
    """Decode a video to RGB24 frames of a given size, read in batches on a background thread."""

    def __init__(self, info: VideoInfo, width: int, height: int, batch: int = 4, hwaccel: bool = False, queue_size: int = 4):
        self.width, self.height, self.batch = width, height, batch
        vf = f"scale={width}:{height}:flags=lanczos:in_color_matrix={info.matrix}:in_range={'pc' if info.color_range == 'pc' else 'tv'},format=rgb24"
        cmd = [ffmpeg_bin(), "-nostdin", "-hide_banner", "-loglevel", "error"]
        if hwaccel:
            cmd += ["-hwaccel", "auto"]
        cmd += ["-i", str(info.path), "-map", "0:v:0", "-vf", vf, "-fps_mode", "cfr", "-r", str(info.fps), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
        self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=_NO_WINDOW)
        self._queue: Queue = Queue(maxsize=queue_size)
        self._error: str | None = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        frame_bytes = self.width * self.height * 3
        assert self._proc.stdout is not None
        try:
            while True:
                frames = []
                for _ in range(self.batch):
                    buf = self._proc.stdout.read(frame_bytes)
                    if len(buf) < frame_bytes:
                        break
                    frames.append(np.frombuffer(buf, np.uint8).reshape(self.height, self.width, 3))
                if frames:
                    self._queue.put(np.stack(frames))
                if len(frames) < self.batch:
                    break
        finally:
            self._proc.wait()
            if self._proc.returncode not in (0, None):
                err = self._proc.stderr.read().decode(errors="replace") if self._proc.stderr else ""
                self._error = f"ffmpeg decode failed ({self._proc.returncode}): {err.strip()}"
            self._queue.put(None)

    def __iter__(self):
        while (item := self._queue.get()) is not None:
            yield item
        if self._error:
            raise RuntimeError(self._error)

    def close(self) -> None:
        if self._proc.poll() is None:
            self._proc.kill()
        # Drain so the reader thread can finish.
        while self._thread.is_alive():
            try:
                self._queue.get(timeout=0.1)
            except Exception:
                pass


@lru_cache(maxsize=None)
def encoder_available(name: str) -> bool:
    """True if ffmpeg can actually open this encoder (e.g. NVENC needs an NVIDIA GPU + driver)."""
    cmd = [ffmpeg_bin(), "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "color=black:s=256x256:d=0.1",
           "-frames:v", "1", "-c:v", name, "-f", "null", "-"]
    try:
        return subprocess.run(cmd, capture_output=True, timeout=30, creationflags=_NO_WINDOW).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def choose_encoder(codec: str, encoder: str, width: int) -> tuple[str, str]:
    """Pick (codec, ffmpeg encoder name). H.264 is limited to 4096 px wide, so wide SBS uses HEVC."""
    use_nvenc = encoder == "nvenc" or (encoder == "auto" and encoder_available("hevc_nvenc"))
    if codec == "auto":
        codec = "hevc" if (use_nvenc or width > 4096) else "h264"
    if codec == "h264" and width > 4096:
        print(f"[parallaxa] {width}px is wider than H.264 allows; using HEVC instead.")
        codec = "hevc"
    if use_nvenc:
        name = f"{codec}_nvenc"
        if encoder == "nvenc" and not encoder_available(name):
            raise RuntimeError(f"{name} is not usable here (needs an NVIDIA GPU and a recent driver).")
        if encoder_available(name):
            return codec, name
    return codec, {"h264": "libx264", "hevc": "libx265"}[codec]


def container_for(info: VideoInfo, container: str) -> str:
    if container != "auto":
        return container
    ext = info.path.suffix.lower().lstrip(".")
    return {"mkv": "mkv", "webm": "mkv", "mov": "mov"}.get(ext, "mp4")


class FrameWriter:
    """Encode RGB24 frames via ffmpeg, copying audio (and subtitles/chapters where possible) from the source."""

    def __init__(
        self,
        out_path: Path,
        info: VideoInfo,
        width: int,
        height: int,
        layout: str,
        codec: str = "auto",
        encoder: str = "auto",
        crf: int = 18,
        preset: str = "",
        queue_size: int = 4,
    ):
        self.out_path = out_path
        self.frame_shape = (height, width, 3)
        ext = out_path.suffix.lower().lstrip(".")
        container = "mkv" if ext in ("mkv", "webm") else ("mov" if ext == "mov" else "mp4")
        codec, enc = choose_encoder(codec, encoder, width)
        self.codec, self.encoder = codec, enc

        matrix = info.matrix
        cmd = [
            ffmpeg_bin(), "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-framerate", str(info.fps), "-i", "-",
            "-i", str(info.path),
            "-map", "0:v:0", "-map", "1:a?",
        ]
        if container == "mkv":
            cmd += ["-map", "1:s?", "-c:s", "copy"]
        cmd += ["-map_metadata", "1", "-map_chapters", "1"]

        # Colour: convert back with the same matrix used for decoding and tag the stream.
        out_matrix = "bt2020" if matrix == "bt2020" else matrix
        cmd += ["-vf", f"scale=out_color_matrix={out_matrix}:out_range=tv:flags=bicubic,format=yuv420p"]
        tags = {"bt709": ("bt709", "bt709", "bt709"), "bt601": ("smpte170m", "smpte170m", "smpte170m"), "bt2020": ("bt2020nc", "bt2020", "bt709")}
        cs, prim, trc = tags.get(out_matrix, tags["bt709"])
        cmd += ["-colorspace", cs, "-color_primaries", info.color_primaries or prim, "-color_trc", trc if info.is_hdr else (info.color_transfer or trc), "-color_range", "tv"]

        cmd += ["-c:v", enc]
        if enc.endswith("_nvenc"):
            cmd += ["-preset", preset or "p5", "-tune", "hq", "-rc", "vbr", "-cq", str(crf + 1), "-b:v", "0", "-spatial-aq", "1"]
        elif enc == "libx264":
            cmd += ["-preset", preset or "medium", "-crf", str(crf)]
            if layout in _X264_FRAME_PACKING:
                cmd += ["-x264-params", f"frame-packing={_X264_FRAME_PACKING[layout]}"]
        else:  # libx265
            cmd += ["-preset", preset or "medium", "-crf", str(crf + 2), "-x265-params", "log-level=error"]
        if codec == "hevc" and container in ("mp4", "mov"):
            cmd += ["-tag:v", "hvc1"]  # plays on Apple devices / Vision Pro
        if container == "mkv" and layout in _MKV_STEREO_MODE:
            cmd += ["-metadata:s:v:0", f"stereo_mode={_MKV_STEREO_MODE[layout]}"]

        ok = _AUDIO_COPY_OK[container]
        if ok is None or all(c in ok for c in info.audio_codecs):
            cmd += ["-c:a", "copy"]
        else:
            cmd += ["-c:a", "aac", "-b:a", "320k"]
        if container in ("mp4", "mov"):
            cmd += ["-movflags", "+faststart"]
        cmd += [str(out_path)]

        out_path.parent.mkdir(parents=True, exist_ok=True)
        self.cmd = cmd
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=_NO_WINDOW)
        self._queue: Queue = Queue(maxsize=queue_size)
        self._error: str | None = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        assert self._proc.stdin is not None
        try:
            while (frames := self._queue.get()) is not None:
                if self._error is None:
                    self._proc.stdin.write(np.ascontiguousarray(frames).tobytes())
        except (BrokenPipeError, OSError) as e:
            self._error = f"ffmpeg encoder stopped: {e}"
            while self._queue.get() is not None:
                pass

    def write(self, frames: np.ndarray) -> None:
        """frames: (N, H, W, 3) uint8."""
        if self._error:
            raise RuntimeError(self._error + self._stderr())
        assert frames.shape[1:] == self.frame_shape, (frames.shape, self.frame_shape)
        self._queue.put(frames)

    def _stderr(self) -> str:
        try:
            return "\n" + self._proc.stderr.read().decode(errors="replace").strip() if self._proc.stderr else ""
        except Exception:
            return ""

    def close(self) -> None:
        self._queue.put(None)
        self._thread.join()
        if self._proc.stdin:
            try:
                self._proc.stdin.close()
            except OSError:
                pass
        self._proc.wait()
        if self._proc.returncode != 0 or self._error:
            raise RuntimeError(f"ffmpeg encode failed ({self._proc.returncode}): {self._error or ''}{self._stderr()}")

    def abort(self) -> None:
        if self._proc.poll() is None:
            self._proc.kill()
        self._queue.put(None)
        self._thread.join(timeout=5)
        self._proc.wait()
