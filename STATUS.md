# Project status / hand-over notes

Last updated: 2026-10-31

## Done
- Phase 1 (CLI converter) is committed and pushed. 22 tests pass on CPU with offline debug models.

## User's test PC (Windows 11)
- RTX 3060 12 GB, NVIDIA driver 591.59 (CUDA 13.1), Python 3.11.9 venv at `C:\Users\Simon\parallaxa\.venv`
- torch 2.11.0+cu128: `CUDA available: True`.
- FFmpeg 9.0.2 (winget Gyan.FFmpeg): NVENC fails with "requires nvenc API 13.1 / driver 610+".
  Fix in progress: update the driver to 610+, or downgrade to FFmpeg 8.x. CPU encoders work meanwhile.

## RTX 3060 measurements (720p clip, 492 frames, NVENC working)
- Depth per frame: Large@518 131-172 ms, Base@518 91 ms, Small@392 54 ms (bench: 28 ms)
- Render after sparse-fill optimisation: ~15 ms (was 57 ms)
- User saw no visible difference between Large, Base and Small on video, so video default -> Small
- GPU is healthy under load (1942 MHz, 61 C, ~120 W of 170 W); matmul bench 9 TFLOPS fp16 / 3 fp32
- Speed lever for Large: TensorRT/ONNX path (planned)

## Waiting on the user
- First real-model results: photo depth quality, time per photo, video fps, flicker, comfort
  of `--strength 3 --convergence 0.6`

## Next: phase 2
- Video Depth Anything (Small Apache-2.0 / Large CC-BY-NC) for temporally stable depth,
  with sliding windows that overlap and align scale/shift between chunks
- Chunked processing for long films; resume support
- Tune defaults from the user's feedback (strength, convergence, edge dilation)
