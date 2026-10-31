# Project status / hand-over notes

Last updated: 2026-10-31 (phase 3)

## Done
- Phase 1 (CLI converter).
- Phase 2: Video Depth Anything (vendored network, our own streaming windowed inference that matches
  the upstream schedule exactly; tested against a port of the upstream algorithm), video default =
  vda-small@392; segmented output in <name>.parallaxa-work with resume (frame-exact via fps+trim
  filters; VDA warm-up + saved anchor); `bench` supports VDA. 40 tests pass offline.
- Phase 2 verified on the RTX 3060: real VDA-Small weights load and run (720p: 83 ms/frame depth,
  8.0 fps overall; 360p: ~13 fps). Ctrl+C + resume tested twice on a 3.4-min clip: joins clean,
  audio in sync.
- Open: render stages slowed from ~15 to ~34 ms/frame when running alongside VDA (cause unknown;
  suspect GPU memory pressure or uneven batch sizes). Low priority now that VDA is optional.
- User compared balanced (VDA) vs fast (per-frame DA-V2 Small): no visible difference, so the video
  default is back to `fast`; VDA stays available as `--profile balanced` / `quality`.

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

## Phase 3 (desktop app), done in the dev container
- PySide6 GUI: `parallaxa gui` / `parallaxa-gui.exe`. Queue on the left with settings below it,
  big live preview on the right (anaglyph / wobble / cross-eye / SBS / depth). Presets + last
  settings in %APPDATA%\Parallaxa. Jobs run as `parallaxa convert --progress json --control-stdin`
  child processes; Stop sends 'cancel' on stdin (clean, so segments are kept), and Start resumes.
- Fixed: JSON progress always reported eta_s=null (eta read twice).
- 49 tests pass, including offscreen GUI tests with real child processes and stop/resume.
- Not yet tried on Windows by the user.

## Next
- User to try the app on Windows (drag-drop, preview speed with DA-V2 Large, stop/resume)
- Then phase 4: optional generative high-quality fill
