"""`parallaxa bench`: isolate where GPU time goes (raw GPU speed, precision, preprocessing, model)."""

from __future__ import annotations

import time

import torch
import torch.nn.functional as F

from .depth import load_depth_model
from .depth.depth_anything_v2 import inference_size
from .device import default_dtype, describe, resolve_device


def _time(fn, device: torch.device, runs: int) -> float:
    """Median seconds per call after warm-up."""
    for _ in range(3):
        fn()
    times = []
    for _ in range(runs):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter()
        fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t)
    return sorted(times)[len(times) // 2]


def run(model_name: str, device_name: str, height: int, width: int, depth_res: int, batch: int, runs: int) -> None:
    device = resolve_device(device_name)
    print(f"device: {describe(device)}   torch {torch.__version__}")
    if device.type == "cuda":
        print(f"cudnn {torch.backends.cudnn.version()}  "
              f"flash-sdp {torch.backends.cuda.flash_sdp_enabled()}  mem-efficient-sdp {torch.backends.cuda.mem_efficient_sdp_enabled()}")

    # 1. Raw GPU throughput.
    n = 4096 if device.type == "cuda" else 1024
    for dt in (torch.float16, torch.float32):
        if device.type == "cpu" and dt == torch.float16:
            continue
        a = torch.randn(n, n, device=device, dtype=dt)
        s = _time(lambda: a @ a, device, max(5, runs))
        print(f"matmul {str(dt).split('.')[-1]:8s} {2 * n**3 / s / 1e12:6.1f} TFLOPS")

    # 2. Depth model, broken down.
    frames = torch.rand(batch, 3, height, width, device=device)
    ih, iw = inference_size(height, width, depth_res)
    print(f"\nmodel {model_name}: {batch}x {width}x{height} frames -> network input {iw}x{ih}")
    results = []
    for dt in ([default_dtype(device), torch.float32] if device.type == "cuda" else [torch.float32]):
        est = load_depth_model(model_name, device, dt, depth_res)
        tag = str(dt).split(".")[-1]
        inner = getattr(est.model, "backbone", est.model)  # attention lives in the DINOv2 backbone
        attn = getattr(inner.config, "_attn_implementation", "?")
        x = torch.randn(batch, 3, ih, iw, device=device, dtype=dt)
        with torch.inference_mode():
            pre = _time(lambda: F.interpolate(frames.to(dt), size=(ih, iw), mode="bicubic", align_corners=False,
                                              antialias=ih < height), device, runs)
            fwd = _time(lambda: est.model(pixel_values=x).predicted_depth, device, runs)
            full = _time(lambda: est.predict(frames), device, runs)
        results.append((tag, attn, pre, fwd, full))
        del est
        if device.type == "cuda":
            torch.cuda.empty_cache()
    print(f"{'precision':10s} {'attention':10s} {'resize':>10s} {'network':>10s} {'full':>10s}   (ms per frame)")
    for tag, attn, pre, fwd, full in results:
        print(f"{tag:10s} {attn:10s} {1000 * pre / batch:10.1f} {1000 * fwd / batch:10.1f} {1000 * full / batch:10.1f}")
    if device.type == "cuda":
        print(f"\npeak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")
