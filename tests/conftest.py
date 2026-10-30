import shutil
import subprocess

import numpy as np
import pytest


@pytest.fixture
def scene():
    """Synthetic 96x128 image: textured far background with a bright near square in the middle."""
    h, w = 96, 128
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.zeros((h, w, 3), np.uint8)
    img[..., 0] = (xx * 2) % 256
    img[..., 1] = 60
    img[..., 2] = (yy * 2) % 256
    depth = np.zeros((h, w), np.float32)
    img[32:64, 48:80] = (250, 250, 250)
    depth[32:64, 48:80] = 1.0
    return img, depth


@pytest.fixture
def sample_video(tmp_path):
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not installed")
    path = tmp_path / "clip.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc2=size=160x90:rate=24:duration=1.5",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=1.5", "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-shortest", str(path)],
        check=True,
    )
    return path
