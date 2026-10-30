"""Compute device selection (cuda / cpu)."""

from __future__ import annotations

import torch


def resolve_device(name: str = "auto") -> torch.device:
    """Return a torch.device for 'auto', 'cuda', 'cuda:N' or 'cpu'."""
    name = (name or "auto").lower()
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but PyTorch cannot see a GPU. Install the CUDA build of "
            "PyTorch (see README) or pass --device cpu."
        )
    return torch.device(name)


def default_dtype(device: torch.device, fp16: bool | None = None) -> torch.dtype:
    """Half precision on GPU (much faster on RTX cards), float32 on CPU."""
    if fp16 is None:
        fp16 = device.type == "cuda"
    return torch.float16 if fp16 and device.type == "cuda" else torch.float32


def describe(device: torch.device) -> str:
    if device.type == "cuda":
        idx = device.index or 0
        props = torch.cuda.get_device_properties(idx)
        return f"cuda:{idx} {props.name} ({props.total_memory / 2**30:.1f} GB)"
    return "cpu"
