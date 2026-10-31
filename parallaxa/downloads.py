"""Fetch model weights ahead of time (installer and first launch of the app)."""

from __future__ import annotations

import json
import sys
import time

# name -> (description, approximate download size in MB)
DOWNLOADS = {
    "da2-large": ("Depth Anything V2 Large: photo depth", 1300),
    "da2-small": ("Depth Anything V2 Small: video depth", 100),
    "vda-small": ("Video Depth Anything Small: steady video depth (optional)", 120),
    "sd15": ("Stable Diffusion 1.5 inpainting: AI gap filling (optional)", 2100),
}
ESSENTIAL = ["da2-large", "da2-small"]


def _da2_repo(name: str) -> str:
    from .depth.depth_anything_v2 import HF_IDS

    return HF_IDS[name.split("-", 1)[1]]


def is_cached(name: str) -> bool:
    """True if the model's main weights are already in the Hugging Face cache."""
    from huggingface_hub import try_to_load_from_cache

    if name.startswith("da2-"):
        repo, filename = _da2_repo(name), "model.safetensors"
    elif name.startswith("vda-"):
        from .depth.video_depth_anything import HF_WEIGHTS

        repo, filename, _ = HF_WEIGHTS[name.split("-", 1)[1]]
    elif name == "sd15":
        from .stereo.inpaint import INPAINT_MODELS

        repo, filename = INPAINT_MODELS["sd15"], "unet/diffusion_pytorch_model.fp16.safetensors"
    else:
        raise ValueError(name)
    return isinstance(try_to_load_from_cache(repo, filename), str)


def missing(names: list[str] | None = None) -> list[str]:
    try:
        return [n for n in (names or ESSENTIAL) if not is_cached(n)]
    except Exception:  # cache unreadable: let the download step sort it out
        return list(names or ESSENTIAL)


def fetch(name: str) -> None:
    from huggingface_hub import hf_hub_download, snapshot_download

    if name.startswith("da2-"):
        snapshot_download(_da2_repo(name), allow_patterns=["*.json", "*.safetensors", "*.txt"])
    elif name.startswith("vda-"):
        from .depth.video_depth_anything import HF_WEIGHTS

        repo, filename, _ = HF_WEIGHTS[name.split("-", 1)[1]]
        hf_hub_download(repo, filename)
    elif name == "sd15":
        from diffusers import DiffusionPipeline

        from .stereo.inpaint import INPAINT_MODELS

        DiffusionPipeline.download(INPAINT_MODELS["sd15"], variant="fp16", use_safetensors=True)


def run(names: list[str], as_json: bool = False, force: bool = False) -> int:
    def emit(event: str, **kw) -> None:
        if as_json:
            print(json.dumps({"event": event, **kw}), flush=True)
        elif event == "start":
            print(f"Downloading {kw['name']}: {kw['description']} (~{kw['mb']} MB) ...", file=sys.stderr, flush=True)
        elif event == "done":
            print(f"  ready ({kw['seconds']:.0f}s)", file=sys.stderr, flush=True)
        elif event == "failed":
            print(f"  FAILED: {kw['error']}", file=sys.stderr, flush=True)

    todo = names if force else missing(names)
    failures = 0
    for i, name in enumerate(todo, 1):
        desc, mb = DOWNLOADS[name]
        emit("start", name=name, description=desc, mb=mb, index=i, total=len(todo))
        t = time.perf_counter()
        try:
            fetch(name)
            emit("done", name=name, seconds=time.perf_counter() - t)
        except Exception as e:
            failures += 1
            emit("failed", name=name, error=str(e).splitlines()[0][:300])
    if not todo:
        emit("log", message="All requested models are already downloaded.") if as_json else \
            print("All requested models are already downloaded.", file=sys.stderr)
    return 1 if failures else 0
