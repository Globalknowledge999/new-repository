"""Settings shared by the photo and video pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field

PHOTO_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif", ".bmp", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".ts", ".m2ts", ".mts", ".wmv", ".flv", ".mpg", ".mpeg"}

PHOTO_FORMATS = ("sbs", "hsbs", "tb", "htb", "anaglyph", "mpo", "depth", "rgbd")
VIDEO_FORMATS = ("sbs", "hsbs", "tb", "htb", "anaglyph", "rgbd")

# Named quality profiles: depth model + depth inference resolution (short side, pixels).
PROFILES = {
    "fast": {"model": "da2-small", "depth_res": 392},
    "balanced": {"model": "da2-base", "depth_res": 518},
    "quality": {"model": "da2-large", "depth_res": 518},
    "photo-quality": {"model": "da2-large", "depth_res": 700},
}


@dataclass
class StereoSettings:
    # Total parallax range (nearest minus farthest) as a percentage of the eye-view width.
    strength: float = 3.0
    # Depth level (0 = farthest, 1 = nearest) placed at the screen plane. Content nearer than this
    # pops out of the screen; content farther sits behind it.
    convergence: float = 0.6
    # 'symmetric' synthesises both eyes (half the shift each, smaller holes);
    # 'right' keeps the original as the left eye and synthesises only the right eye.
    eye_mode: str = "symmetric"
    # Grow near objects by this fraction of the width before warping, so their edges carry
    # their own depth instead of a smeared mix with the background.
    edge_dilate: float = 0.002
    # Light blur on the disparity map (fraction of the width) to avoid jagged warps.
    depth_blur: float = 0.001
    # Surfaces closer than this many pixels of disparity are blended rather than occluded.
    z_tolerance: float = 1.0
    # Depth-aware smoothing passes over filled holes (0 = plain background stretch).
    fill_iterations: int = 8
    # Gamma applied to normalised depth (<1 adds separation in the distance, >1 up close).
    depth_gamma: float = 1.0


@dataclass
class DepthSettings:
    model: str = "da2-large"
    depth_res: int = 518
    # Robust normalisation range (percentiles of the raw relative-depth map).
    low_pct: float = 2.0
    high_pct: float = 98.0
    # Video only: exponential smoothing of the normalisation range between frames (0 = none).
    temporal_alpha: float = 0.9


@dataclass
class OutputSettings:
    formats: tuple[str, ...] = ("sbs",)
    # Per-eye output height: 0 keeps the source resolution.
    height: int = 0
    anaglyph: str = "dubois"
    jpeg_quality: int = 95
    image_ext: str = ".jpg"  # for photo outputs that are not MPO/depth
    # Video
    codec: str = "auto"  # auto | h264 | hevc
    encoder: str = "auto"  # auto | nvenc | cpu
    crf: int = 18
    preset: str = ""
    container: str = "auto"  # auto | mp4 | mkv | mov


@dataclass
class JobSettings:
    device: str = "auto"
    fp16: bool | None = None
    batch_size: int = 4
    overwrite: bool = False
    stereo: StereoSettings = field(default_factory=StereoSettings)
    depth: DepthSettings = field(default_factory=DepthSettings)
    output: OutputSettings = field(default_factory=OutputSettings)
