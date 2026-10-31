# Project status / hand-over notes

Last updated: 2026-10-31 (phase 5)

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

## Phase 4 (AI gap filling), done in the dev container
- Stable Diffusion inpainting via diffusers (safetensors, fp16). Default `auto`: SD 1.5 inpainting, falling back to SDXL, then SD 2. SD 2 returned "Repository Not Found" on the user's PC on 2026-10-31. Masks hide foreground pixels next
  to each gap so the model continues the background; works on 512 px tiles at ~1024 px and only
  processes tiles with gaps; feathered composite. `--fill auto|basic|ai` (auto = ai for photos,
  basic for video). GUI: "Gap filling" setting; the preview shows basic instantly, then AI.
- LaMa rejected: TorchScript (executable) repackaged by a third party. StereoCrafter: >12 GB, gated.
- On the RTX 3060: SD 1.5 inpainting loads; ~1.7 s per 512 px tile; G1.jpg at strength 5 = 12 tiles
  in 20.5 s, 29.8 s total. An earlier 7 s per tile was caused by cuDNN autotuning, which is now on
  for video only and off during AI fill.
- User's verdict on G1.jpg at strength 5: AI and basic look the same, so basic is the default for
  photos too; AI fill stays available as `--fill ai` or in the app's Gap filling setting.

## Phase 5 (installer), done in the dev container
- installer/install.ps1: per-user install/update into %LOCALAPPDATA%\Programs\Parallaxa via uv 0.13.6
  (private Python 3.12), CUDA PyTorch matched to the driver (570+ cu128, 560+ cu126, else CPU),
  FFmpeg 8.0.1 essentials (driver 570+) or 7.1.1, with an NVENC test and fallback; `parallaxa download`
  of the depth models; Start menu/desktop shortcuts; HKCU Apps & features entry; uninstall.ps1.
- installer/parallaxa.iss: Inno Setup wrapper (Parallaxa-Setup-<ver>.exe, no admin rights needed).
- .github/workflows/windows-installer.yml: on windows-latest, runs install.ps1 (CPU torch), converts
  a video and a photo, runs the test suite, builds the setup, uploads it as an artifact (and to the
  release on v* tags).
- App: finds the installer's FFmpeg; `parallaxa download`; first-launch download offer and a
  "Download models…" toolbar action.
- All URLs verified to exist (uv 0.13.6, GyanD/codexffmpeg 8.0.1 and 7.1.1 essentials). The PowerShell
  itself could not be run here (no Windows/pwsh); the CI run is its first real test.

## Next
- User to try the app on Windows (drag-drop, preview speed with DA-V2 Large, stop/resume)
- Check the first Windows CI run; user to try Parallaxa-Setup.exe
