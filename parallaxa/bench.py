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
    for _ in range(min(3, runs)):
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
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True  # same as real conversions
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
    est0 = load_depth_model(model_name, device, default_dtype(device), depth_res)
    if getattr(est0, "temporal", False):
        _bench_temporal(est0, frames, device, runs)
        return
    del est0
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


def _bench_temporal(est, frames: torch.Tensor, device: torch.device, runs: int) -> None:
    """Video Depth Anything: time whole 32-frame windows; each window yields 22 new frames."""
    from .depth.video_depth_anything import STEP, WINDOW

    stream = est.stream()
    x = stream._prep(frames[:1]).expand(WINDOW, -1, -1, -1).contiguous()
    window = list(x)
    win = _time(lambda: stream._run(window), device, max(3, runs // 3))
    print(f"window of {WINDOW} frames at {x.shape[-1]}x{x.shape[-2]}: {1000 * win:.0f} ms "
          f"-> {1000 * win / STEP:.1f} ms per output frame (fp16 autocast: {stream.autocast})")
    if device.type == "cuda":
        print(f"\npeak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")


def run_inpaint(model: str, device_name: str, steps: int, runs: int = 2) -> None:
    """Time the AI gap-filling model on 512 px tiles at batch sizes 1, 2 and 4."""
    from .stereo.inpaint import load_inpainter

    device = resolve_device(device_name)
    dtype = default_dtype(device)
    print(f"device: {describe(device)}   torch {torch.__version__}")
    inp = load_inpainter(model, device, dtype, steps)
    pipe = inp.pipe
    print(f"pipeline {type(pipe).__name__}: unet {pipe.unet.dtype} on {pipe.unet.device}, vae {pipe.vae.dtype}, "
          f"attention {type(next(iter(pipe.unet.attn_processors.values()))).__name__}, tile {inp.tile}px, {steps} steps")
    t = inp.tile
    for batch in (1, 2, 4):
        img = torch.rand(batch, 3, t, t, device=device, dtype=dtype)
        mask = torch.zeros(batch, 1, t, t, device=device, dtype=dtype)
        mask[..., t // 3 : 2 * t // 3, t // 2 - 16 : t // 2 + 16] = 1
        kw = dict(image=img, mask_image=mask, height=t, width=t, strength=1.0, num_inference_steps=steps,
                  guidance_scale=inp.guidance, output_type="pt")
        if pipe.tokenizer is not None:
            kw.update(prompt=["background"] * batch, negative_prompt=["object"] * batch)
        else:
            pos, neg = inp._prompt_embeds()
            kw.update(prompt_embeds=pos.repeat(batch, 1, 1), negative_prompt_embeds=neg.repeat(batch, 1, 1))
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        try:
            sec = _time(lambda: pipe(**kw), device, runs)
        except torch.OutOfMemoryError:
            print(f"batch {batch}: out of GPU memory")
            break
        vram = f", peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GB" if device.type == "cuda" else ""
        print(f"batch {batch}: {sec:.2f} s per call = {sec / batch:.2f} s per tile, {steps * batch / sec:.1f} tile-steps/s{vram}")
