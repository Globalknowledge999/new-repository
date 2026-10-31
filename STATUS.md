# Project status / hand-over notes

Last updated: 2026-10-31

## Done
- Phase 1 (CLI converter) is committed and pushed. 22 tests pass on CPU with offline debug models.

## User's test PC (Windows 11)
- RTX 3060 12 GB, NVIDIA driver 591.59 (CUDA 13.1), Python 3.11.9 venv at `C:\Users\Simon\parallaxa\.venv`
- torch 2.11.0+cu128: `CUDA available: True`.
- FFmpeg 9.0.2 (winget Gyan.FFmpeg): NVENC fails with "requires nvenc API 13.1 / driver 610+".
  Fix in progress: update the driver to 610+, or downgrade to FFmpeg 8.x. CPU encoders work meanwhile.

## Waiting on the user
- `parallaxa info` showing `hevc_nvenc ok`
- First real-model results: photo depth quality, time per photo, video fps, flicker, comfort
  of `--strength 3 --convergence 0.6`

## Next: phase 2
- Video Depth Anything (Small Apache-2.0 / Large CC-BY-NC) for temporally stable depth,
  with sliding windows that overlap and align scale/shift between chunks
- Chunked processing for long films; resume support
- Tune defaults from the user's feedback (strength, convergence, edge dilation)
