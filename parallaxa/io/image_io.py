"""Photo loading/saving, including HEIC input and stereo MPO output."""

from __future__ import annotations

import io
import struct
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

try:  # HEIC/HEIF support (iPhone photos)
    import pillow_heif

    pillow_heif.register_heif_opener()
    HEIF_AVAILABLE = True
except ImportError:  # pragma: no cover
    HEIF_AVAILABLE = False


def load_image(path: str | Path) -> tuple[np.ndarray, dict]:
    """Load an image as an RGB uint8 array (H, W, 3), honouring EXIF orientation.

    Returns the array and a dict with metadata worth carrying to the output (ICC profile).
    """
    path = Path(path)
    if path.suffix.lower() in {".heic", ".heif"} and not HEIF_AVAILABLE:
        raise RuntimeError("HEIC input needs the 'pillow-heif' package.")
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        meta = {"icc_profile": im.info.get("icc_profile")}
        if im.mode in ("I;16", "I;16B", "I", "F"):
            arr = np.asarray(im, dtype=np.float32)
            arr = arr / max(float(arr.max()), 1.0) * 255.0
            im = Image.fromarray(arr.astype(np.uint8))
        rgb = im.convert("RGB")
        return np.asarray(rgb).copy(), meta


def save_image(arr: np.ndarray, path: str | Path, quality: int = 95, icc_profile: bytes | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    im = Image.fromarray(arr)
    ext = path.suffix.lower()
    kw: dict = {}
    if icc_profile and im.mode == "RGB":
        kw["icc_profile"] = icc_profile
    if ext in (".jpg", ".jpeg"):
        im.save(path, quality=quality, subsampling=0 if quality >= 90 else 2, optimize=True, **kw)
    elif ext == ".webp":
        im.save(path, quality=quality, method=5, **kw)
    elif ext in (".heic", ".heif"):
        im.save(path, quality=quality, **kw)
    else:
        im.save(path, **kw)


def save_depth_png16(depth01: np.ndarray, path: str | Path) -> None:
    """Save a [0,1] depth map (1 = near) as a 16-bit greyscale PNG."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    d16 = np.clip(depth01 * 65535.0 + 0.5, 0, 65535).astype(np.uint16)
    Image.fromarray(d16).save(path)


# --------------------------------------------------------------------------------------------
# MPO (CIPA DC-007 "Multi-Picture Format") stereo writer.
#
# Each view is a complete JPEG. The first one carries an APP2 "MPF" segment holding the MP Index
# IFD (where every image lives in the file) and its own MP Attribute IFD; the second carries just
# its attribute IFD. Both are tagged as "Multi-Frame Image: Disparity" so 3D viewers (Fujifilm
# W1/W3 style, Nintendo 3DS, StereoPhoto Maker) read them as left/right views.
# --------------------------------------------------------------------------------------------

_TYPE_LONG = 4
_TYPE_UNDEFINED = 7
_MP_TYPE_DISPARITY = 0x020002
_FLAG_REPRESENTATIVE = 1 << 29


def _ifd(entries: list[tuple[int, int, int, bytes]], next_ifd: int) -> bytes:
    """Big-endian IFD where every value fits in 4 bytes or is already an offset."""
    out = struct.pack(">H", len(entries))
    for tag, typ, count, value in entries:
        out += struct.pack(">HHI", tag, typ, count) + value.ljust(4, b"\0")
    return out + struct.pack(">I", next_ifd)


def _attr_entries(index: int) -> list[tuple[int, int, int, bytes]]:
    return [
        (0xB000, _TYPE_UNDEFINED, 4, b"0100"),  # MPFVersion
        (0xB101, _TYPE_LONG, 1, struct.pack(">I", index)),  # MPIndividualNum (1 = left)
        (0xB201, _TYPE_LONG, 1, struct.pack(">I", 1)),  # BaseViewpointNum
    ]


def _app2(payload: bytes) -> bytes:
    body = b"MPF\0" + payload
    return b"\xff\xe2" + struct.pack(">H", len(body) + 2) + body


def _insert_after_app_headers(jpeg: bytes, segment: bytes) -> tuple[bytes, int]:
    """Insert a segment after SOI and any leading APP0/APP1 segments; return (data, position)."""
    assert jpeg[:2] == b"\xff\xd8", "not a JPEG"
    pos = 2
    while jpeg[pos] == 0xFF and jpeg[pos + 1] in (0xE0, 0xE1):
        (length,) = struct.unpack(">H", jpeg[pos + 2 : pos + 4])
        pos += 2 + length
    return jpeg[:pos] + segment + jpeg[pos:], pos


def _encode_jpeg(arr: np.ndarray, quality: int) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=quality, subsampling=0 if quality >= 90 else 2)
    return buf.getvalue()


def save_mpo(left: np.ndarray, right: np.ndarray, path: str | Path, quality: int = 95) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    j1, j2 = _encode_jpeg(left, quality), _encode_jpeg(right, quality)

    # Image 2: attribute IFD only.
    tiff2 = b"MM\0\x2a" + struct.pack(">I", 8) + _ifd(_attr_entries(2), 0)
    j2, _ = _insert_after_app_headers(j2, _app2(tiff2))

    # Image 1: index IFD -> MP entries -> attribute IFD. Layout offsets are relative to the
    # TIFF header ("MM") inside the APP2 segment.
    n = 2
    index_ifd_len = 2 + 3 * 12 + 4
    entries_off = 8 + index_ifd_len
    attr_off = entries_off + 16 * n

    def build_tiff1(size1: int, offset2: int) -> bytes:
        index = _ifd(
            [
                (0xB000, _TYPE_UNDEFINED, 4, b"0100"),
                (0xB001, _TYPE_LONG, 1, struct.pack(">I", n)),
                (0xB002, _TYPE_UNDEFINED, 16 * n, struct.pack(">I", entries_off)),
            ],
            attr_off,
        )
        mp_entries = struct.pack(">IIIHH", _FLAG_REPRESENTATIVE | _MP_TYPE_DISPARITY, size1, 0, 0, 0)
        mp_entries += struct.pack(">IIIHH", _MP_TYPE_DISPARITY, len(j2), offset2, 0, 0)
        return b"MM\0\x2a" + struct.pack(">I", 8) + index + mp_entries + _ifd(_attr_entries(1), 0)

    placeholder = build_tiff1(0, 0)
    j1_full, seg_pos = _insert_after_app_headers(j1, _app2(placeholder))
    tiff_pos = seg_pos + 4 + 4  # marker + length + "MPF\0"
    size1 = len(j1_full)
    offset2 = size1 - tiff_pos  # image 2 starts right after image 1
    final = _app2(build_tiff1(size1, offset2))
    assert len(final) == len(_app2(placeholder))
    j1_full, _ = _insert_after_app_headers(j1, final)
    path.write_bytes(j1_full + j2)
