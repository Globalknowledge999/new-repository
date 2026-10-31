"""Conversion options shown in the GUI, presets, and where they are stored."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path


@dataclass
class ConvertOptions:
    formats: list[str] = field(default_factory=lambda: ["sbs"])
    strength: float = 3.0
    convergence: float = 0.6
    eye_mode: str = "symmetric"
    profile: str = "auto"
    height: int = 0  # per-eye output height; 0 = source
    anaglyph: str = "dubois"
    image_ext: str = ".jpg"
    codec: str = "auto"
    container: str = "auto"
    crf: int = 18
    segment_minutes: float = 10.0
    output_dir: str = ""  # "" = a '3d' folder next to each input
    # Advanced / testing: override the profile's model or device.
    model: str = ""
    device: str = "auto"

    def to_args(self) -> list[str]:
        """Command-line arguments for `parallaxa convert` (the GUI runs conversions through the CLI)."""
        args = ["-f", ",".join(self.formats), "--strength", f"{self.strength:g}", "--convergence", f"{self.convergence:g}",
                "--eye-mode", self.eye_mode, "--profile", self.profile, "--height", str(self.height),
                "--anaglyph", self.anaglyph, "--image-ext", self.image_ext, "--codec", self.codec,
                "--container", self.container, "--crf", str(self.crf), "--segment-minutes", f"{self.segment_minutes:g}",
                "--device", self.device]
        if self.output_dir:
            args += ["-o", self.output_dir]
        if self.model:
            args += ["--model", self.model]
        return args

    def preset_dict(self) -> dict:
        d = asdict(self)
        for k in ("output_dir", "model", "device"):  # machine-specific, not part of a look
            d.pop(k)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ConvertOptions":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


BUILTIN_PRESETS: dict[str, dict] = {
    "VR headset (full SBS)": {"formats": ["sbs"]},
    "3D TV (half SBS)": {"formats": ["hsbs"]},
    "3D TV (half top-bottom)": {"formats": ["htb"]},
    "Red/cyan glasses": {"formats": ["anaglyph"]},
    "Gentle (behind the screen)": {"formats": ["sbs"], "strength": 2.5, "convergence": 0.9},
    "Strong pop-out": {"formats": ["sbs"], "strength": 4.0, "convergence": 0.35},
    "Depth maps (RGB-D + depth)": {"formats": ["rgbd", "depth"]},
}


def config_dir() -> Path:
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return base / "Parallaxa"


class PresetStore:
    """Built-in presets plus the user's own, saved as JSON next to the last-used settings."""

    def __init__(self, directory: Path | None = None):
        self.dir = directory or config_dir()
        self.file = self.dir / "presets.json"
        self.last_file = self.dir / "last_settings.json"

    def _user(self) -> dict[str, dict]:
        try:
            return json.loads(self.file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def names(self) -> list[str]:
        return list(BUILTIN_PRESETS) + sorted(self._user())

    def is_builtin(self, name: str) -> bool:
        return name in BUILTIN_PRESETS

    def get(self, name: str, base: ConvertOptions | None = None) -> ConvertOptions:
        """The preset applied on top of defaults, keeping the current output folder / model / device."""
        values = BUILTIN_PRESETS.get(name) or self._user().get(name, {})
        opts = ConvertOptions.from_dict(values)
        if base:
            opts.output_dir, opts.model, opts.device = base.output_dir, base.model, base.device
        return opts

    def save(self, name: str, opts: ConvertOptions) -> None:
        if self.is_builtin(name):
            raise ValueError(f"'{name}' is a built-in preset; choose another name")
        user = self._user()
        user[name] = opts.preset_dict()
        self._write(self.file, user)

    def delete(self, name: str) -> None:
        user = self._user()
        if user.pop(name, None) is not None:
            self._write(self.file, user)

    def load_last(self) -> ConvertOptions:
        try:
            return ConvertOptions.from_dict(json.loads(self.last_file.read_text(encoding="utf-8")))
        except (OSError, ValueError, TypeError):
            return ConvertOptions()

    def save_last(self, opts: ConvertOptions) -> None:
        self._write(self.last_file, asdict(opts))

    def _write(self, path: Path, data: dict) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp.replace(path)
