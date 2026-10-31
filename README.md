# Parallaxa

Parallaxa converts ordinary 2D photos and videos into stereoscopic 3D on your own GPU. Nothing is
uploaded anywhere. For each frame it estimates depth with a neural network, renders a second-eye
view by shifting pixels according to that depth, fills the gaps that open up behind foreground
objects, and packs the result into common 3D formats. The default output is **full side-by-side
(SBS)**.

> Personal, non-commercial project. The code is MIT. Some of the depth models it downloads are
> CC-BY-NC-4.0 (non-commercial); see [MODELS.md](MODELS.md).

## Status

| Phase | What | State |
|---|---|---|
| 1 | Command-line converter: photos + videos, all output formats, audio passthrough | done |
| 2 | Video Depth Anything for temporally stable video depth; resumable segmented processing for long films; speed work | done |
| 3 | Windows desktop app: drag-and-drop batch queue, live 3D preview, progress/ETA, stop/resume, presets | done |
| 4 | AI gap filling with Stable Diffusion inpainting (default for photos, optional for video) | **done (this release)** |
| 5 | One-click Windows installer (FFmpeg, CUDA PyTorch, models downloaded on first run) | planned |

---

## Setup on a Windows 11 PC with an RTX 3060 (12 GB)

You need about 10 GB of disk space: roughly 3 GB for PyTorch with CUDA, 1.5 GB for the models,
and room for output.

### 1. Prerequisites

1. **NVIDIA driver:** install a current Game Ready or Studio driver (version 570 or newer).
   Check it in a terminal with `nvidia-smi`.
2. **Python 3.12 (64-bit):** download it from <https://www.python.org/downloads/windows/> and tick
   *"Add python.exe to PATH"* during setup. Python 3.11 and 3.13 also work.
3. **Git:** <https://git-scm.com/download/win>, or `winget install Git.Git`.
4. **FFmpeg:** use a full build that includes NVENC. The simplest way to get one:

   ```bash
   winget install Gyan.FFmpeg
   ```

   Then open a **new** terminal and check that it works with `ffmpeg -version`. If you would
   rather not change PATH, unzip FFmpeg to `parallaxa\ffmpeg\` so that
   `parallaxa\ffmpeg\bin\ffmpeg.exe` exists, or set `PARALLAXA_FFMPEG_DIR` to its `bin` folder.

### 2. Get the code and create a virtual environment

Use PowerShell or Windows Terminal:

```bash
git clone <your-repo-url> parallaxa
```

```bash
cd parallaxa
```

```bash
py -3.12 -m venv .venv
```

```bash
.venv\Scripts\activate
```

(If PowerShell refuses to run the activate script, run
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.)

### 3. Install PyTorch with CUDA, then Parallaxa

Install the CUDA build of PyTorch **first**, from PyTorch's own index. The RTX 3060 (Ampere)
works with the CUDA 12.6 or 12.8 builds:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

```bash
pip install -e ".[dev]"
```

### 4. Check the installation

```bash
parallaxa info
```

You should see `CUDA available: True`, your `NVIDIA GeForce RTX 3060 12.0 GB`, and
`hevc_nvenc ok`. If CUDA shows `False`, the CPU-only PyTorch build got installed. Fix it with
`pip uninstall torch torchvision`, then repeat step 3.

### 5. First conversion

```bash
parallaxa convert "C:\Users\me\Pictures\holiday.jpg"
```

The first run downloads Depth Anything V2 Large (~1.3 GB) into the Hugging Face cache. The
result is saved as `holiday_3D_SBS.jpg` in a `3d` folder next to the input.

Optional: run the test suite. It uses offline debug models, so no download is needed:

```bash
pytest -q
```

---

## The desktop app

Open it from the activated venv:

```bash
parallaxa gui
```

Or double-click `parallaxa\.venv\Scripts\parallaxa-gui.exe` (no console window). Right-click it
→ *Send to* → *Desktop (create shortcut)* to get a desktop icon.

* **Add files:** drag photos, videos or whole folders onto the window (sub-folders are
  included), or use *Add files… / Add folder…*.
* **Preview:** click a file in the queue to see it in 3D. Videos show a frame from 10% of the way
  in. Change *View* to see it different ways: *Red/cyan anaglyph* (paper 3D glasses),
  *Wobble* (flips between the eyes, so you can judge depth with no glasses), *Cross-eye*
  (free-viewing), *Side by side*, or the *Depth map*. Depth is estimated once per file, so moving
  the **3D strength** and **Convergence** sliders updates the preview almost instantly.
* **Settings** apply to every file you start afterwards, and are remembered between sessions.
  **Presets** set them in one click (VR headset, 3D TV half-SBS, red/cyan, gentle, strong
  pop-out, depth maps). *Save as…* stores your own presets.
* **Start** converts the queue one file at a time and shows progress and time left. **Stop**
  ends the current file cleanly: a video keeps its finished segments, and pressing **Start**
  again resumes it. Double-click a finished file to open its output folder. Right-click for
  *Convert again* or *Remove*.
* Each conversion runs as a separate `parallaxa convert` process, so a crash or GPU
  out-of-memory error can't take the window down. The log at the bottom shows the exact
  command and its output.

## Command line

```bash
parallaxa convert INPUT [INPUT ...] [-o OUTPUT_DIR] [options]
```

INPUT can be any mix of files and folders. Folders are processed as a batch (add `-r` to include
sub-folders). Photos are processed first, then videos. Files that already exist in the output are
skipped unless you pass `--overwrite`, so you can stop a batch and restart it. Ctrl+C cancels
cleanly: a half-written video is deleted, never left behind as a broken file.

### Examples

```bash
parallaxa convert photos\ -o out\
```

```bash
parallaxa convert photo.heic -f sbs,anaglyph,mpo,depth
```

```bash
parallaxa convert film.mkv --height 1080 -f hsbs
```

```bash
parallaxa convert film.mp4 --strength 4 --convergence 0.5
```

```bash
parallaxa convert clips\ --profile fast --batch-size 8
```

### Output formats (`-f`, comma-separated)

| Key | Photo | Video | Layout | File suffix |
|---|---|---|---|---|
| `sbs` (default) | ✓ | ✓ | Full side-by-side, left eye on the left, 2× width | `_3D_SBS` |
| `hsbs` | ✓ | ✓ | Half SBS: each eye squeezed to half width (3D TVs, some players) | `_3D_HSBS` |
| `tb` | ✓ | ✓ | Full top-bottom (left eye on top), 2× height | `_3D_TB` |
| `htb` | ✓ | ✓ | Half top-bottom | `_3D_HTB` |
| `anaglyph` | ✓ | ✓ | Red/cyan, Dubois method (`--anaglyph half-color/color/gray` for other styles) | `_3D_ANAGLYPH` |
| `rgbd` | ✓ (PNG) | ✓ | Colour on the left, depth (white = near) on the right | `_RGBD` |
| `depth` | ✓ (16-bit PNG) | n/a | Depth map only | `_DEPTH` |
| `mpo` | ✓ | n/a | Stereo MPO (Fujifilm W3 / 3DS / StereoPhoto Maker style) | `_3D.mpo` |

VR players (Quest + Skybox/DeoVR/Heresphere, Apple Vision Pro players, Bigscreen) read full SBS
directly. You can select the layout in the player, and most also recognise `_SBS` / `_TB` in the
file name. MKV outputs also carry a Matroska `stereo_mode` tag. H.264 outputs carry a
frame-packing SEI.

### 3D controls

* `--strength` (default **3.0**) sets the total parallax between the nearest and farthest points,
  as a % of the eye-view width. 2 to 4 is comfortable on most screens and headsets. Above 5 gets
  strong and shows more artefacts at edges.
* `--convergence` (default **0.6**, range 0 to 1) picks which depth sits at screen level.
  Everything nearer than this pops *out* of the screen, everything farther sits *behind* it.
  `1.0` puts the whole scene behind the screen (the most comfortable option). `0.0` makes
  everything pop out.
* `--fill auto|basic|ai` controls gap filling. `ai` repaints the gaps beside foreground objects
  with a **Stable Diffusion inpainting** model, which gives real texture instead of stretched
  background. It's the default for photos (`auto`). It's off for video because it takes a few
  seconds per frame and can flicker. Turn it on with `--fill ai` for short clips.
  `--fill-model auto` (the default) uses SD 1.5 inpainting (~2 GB download). If that repository
  is unavailable, it falls back to SDXL inpainting (~7 GB, slower, finer) and then SD 2. Choose
  one directly with `sd15`, `sdxl`, `sd2` or any Hugging Face inpainting model id.
  `--fill-steps 14` sets the steps (fewer is faster, more is finer). On the RTX 3060 a photo at
  strength 5 takes ~30 s (12 tiles in ~20 s, ~1.7 s per tile), against ~7 s with basic filling.
  The time depends on how many gaps the picture has, which grows with strength. The log reports
  how many tiles were filled and how long it took. `parallaxa bench --inpaint` measures the model alone.
* `--eye-mode symmetric|right|left`. `symmetric` (default) synthesises both eyes with half the
  shift each, which halves the size of every filled gap. `right` keeps the untouched original as
  the left eye.
* `--depth-gamma`: values below 1 add separation in the background, values above 1 add it in the
  foreground.
* `--edge-dilate`, `--fill-iterations`: edge and gap-fill tuning (see *How it works*).

### Quality profiles / models

| `--profile` | Model | Depth resolution | Notes |
|---|---|---|---|
| `--profile` | Model | Depth resolution | Notes |
|---|---|---|---|
| `auto` (default) | photos: `photo-quality`; video: `fast` | | |
| `fast` | DA-V2 Small, per frame | 392 | **Default for video.** ~13 fps at 720p on an RTX 3060 |
| `balanced` | Video Depth Anything Small | 392 | Temporally stable depth (~8 fps at 720p). Try it if a video shimmers or "breathes" |
| `quality` | Video Depth Anything Small | 518 | The resolution VDA was trained at. Sharper, ~1.7× the depth cost |
| `photo-quality` | DA-V2 Large | 700 | **Default for photos** |

To override a profile, use `--model` and `--depth-res N` (a multiple of 14; higher means more
detail, slower and more VRAM). `parallaxa models` lists all the options:

* `vda-small` / `vda-base` / `vda-large`: Video Depth Anything, for video only. If you pass one
  for a photo, it uses the same-size DA-V2 model instead. **`vda-large` won't fit in 12 GB** at
  518 px (upstream measured 23.6 GB). `vda-base` (CC-BY-NC) is worth trying at 392 px.
* `da2-small` / `da2-base` / `da2-large`: Depth Anything V2, per frame.

To measure a model's speed and VRAM on your GPU, run `parallaxa bench --model vda-small --depth-res 392`.

### Video options

* `--height 1080|1440|2160|…` sets the per-eye output height. The default `0` keeps the source
  resolution, so full SBS of a 1080p film comes out 3840×1080.
* `--codec auto|h264|hevc` and `--encoder auto|nvenc|cpu`. On your RTX 3060, `auto` picks
  **HEVC via NVENC**. H.264 can't go wider than 4096 px, so full SBS from 4K sources switches
  to HEVC automatically.
* `--crf 18` sets quality (lower is better and makes larger files). With NVENC it maps to
  constant-quality `-cq 19`.
* `--container auto|mp4|mkv|mov`. `auto` keeps MKV/MOV sources in the same container and uses
  MP4 for everything else. Audio is **copied untouched**. It is re-encoded to 320 kbps AAC only
  when the container can't hold the original codec (e.g. PCM in MP4). MKV also keeps
  subtitles. Chapters and metadata are kept.
* `--batch-size 4` sets frames per GPU step. Try 8 on a 3060 at 1080p. Lower it if you run out
  of VRAM at 4K.
* `--progress json` writes machine-readable progress lines (the future GUI will use these).
* `--segment-minutes 10` sets the segment length. Video is written in segments of this length
  into a `<name>.parallaxa-work` folder next to the output. If a conversion stops (Ctrl+C, crash,
  power cut), **run the same command again** and it continues from the last finished segment.
  When all segments are done, they're joined without re-encoding, the audio is added, and the
  work folder is deleted. Changing any setting between runs starts the conversion over, so
  segments made with different settings never get mixed.
* `--timing` prints how many milliseconds each stage takes per frame. It's useful for finding
  what's slow.

---

## Things I'd like you to test on the RTX 3060

Please report back:

1. The output of `parallaxa info`.
2. **Photos:** convert 5 to 10 varied photos (people, landscapes, close-ups, a HEIC from a phone)
   with `-f sbs,anaglyph,depth`. How do the depth maps look? Are object edges clean, or do you
   see halos and stretching? Do faces look flat or rounded? Note the time per photo printed in
   the log.
3. **Video:** convert a 1 to 2 minute clip with the defaults (`fast`). Report the frames/s,
   whether audio stays in sync, and whether depth flickers or "breathes". If it does, compare
   with `--profile balanced` (Video Depth Anything).
4. **Resume:** start a longer video, press Ctrl+C after a segment or two, run the same command
   again, and check the result plays smoothly across the join.
5. Watch the result in your headset or player. Is `--strength 3 --convergence 0.6` comfortable,
   too weak, or too strong?

If something fails, run it again with `--batch-size 1` and paste the full error output.

---

## How it works

```
decode (FFmpeg, thread) ─► depth model (GPU) ─► normalise ─► stereo render (GPU) ─► pack layouts ─► encode (FFmpeg, thread per format)
                                                                                                    ▲ audio/subs/chapters copied from source
```

1. **Depth.** The network predicts relative inverse depth (bigger means nearer) at a reduced
   resolution (392 to 700 px on the short side, in multiples of the ViT's 14 px patch). Photos
   use Depth Anything V2 one image at a time. Video uses **Video Depth Anything**, which looks at
   32 frames together. Each window overlaps the previous one by 10 frames. Two anchor frames
   (the first frame of the video and one from the previous window) keep the depth scale
   consistent from window to window, and the 8 overlapping frames are cross-faded. This removes
   most of the shimmer that per-frame depth has. Parallaxa runs this as a stream, emitting
   each depth map as soon as it is final (at most ~2 windows of delay). The depth is then
   upsampled to the frame size and mapped to 0..1 using the 2nd and 98th percentiles, so a few
   outlier pixels can't compress the whole scene. That range is also smoothed between frames
   (`--temporal-alpha`) so the 3D strength doesn't pump.
2. **Disparity.** `disparity = strength% × width × (depth − convergence)`. Before this step, near
   objects are grown by a few pixels (a max filter). Depth networks blur depth across object
   silhouettes. Without the growth, the edge pixels of a person get "half" depth and stretch
   like rubber. With it, the silhouette moves cleanly with the person.
3. **Forward warp with a z-buffer.** Every pixel is splatted horizontally to its new position,
   split between the two nearest target pixels for sub-pixel accuracy. When several pixels land
   on the same spot, the nearest one wins. Surfaces within 1 px of parallax of each other are
   blended rather than occluded, which avoids crackling on smooth slopes.
4. **Hole filling.** The gaps that open next to foreground objects are background that was
   hidden behind them, so each gap is filled from whichever side is *farther away*, never from
   the foreground. Then a few depth-aware smoothing passes run inside the gap. They only average
   neighbours that are not nearer than the gap, which removes streaks without bleeding the
   foreground into the background.
   With **AI gap filling**, a Stable Diffusion inpainting model then repaints those gaps.
   Before it runs, foreground pixels next to each gap are hidden from it as well, so it can only
   continue the *background* (an unmasked inpainter tends to grow the person into the gap).
   Only the gap pixels are written back, feathered over 2 px. It works on 512 px tiles at about
   1024 px resolution and only processes tiles that contain gaps. The seed is fixed, so results
   are repeatable.
5. **Pack.** Views are packed into SBS, TB, half variants, anaglyph, RGB-D or MPO and encoded.
   The YUV↔RGB conversion uses the source's colour matrix (BT.709 for HD), so colours don't
   shift.

Video is streamed in small batches between three threads (decode → GPU → encode). Memory use
stays flat whatever the film length. Output is written in resumable segments (see
`--segment-minutes`). On resume, decoding restarts at the exact frame, using FFmpeg's `trim`
filter. Video Depth Anything gets a 64-frame warm-up plus the saved anchor frame, so the depth
scale stays continuous across the join.

### Project layout

```
parallaxa/
  cli.py                 command-line entry point (parallaxa convert / gui / models / info / bench)
  bench.py               GPU / model speed measurements
  config.py              settings dataclasses, quality profiles, file types
  device.py              cuda/cpu selection, fp16 on GPU
  progress.py            progress + ETA (text or JSON lines)
  depth/
    base.py              DepthEstimator interface, robust (temporal) normaliser
    depth_anything_v2.py Depth Anything V2 wrapper (+ tiny offline test network)
    video_depth_anything.py  streaming windowed inference for Video Depth Anything
    __init__.py          model registry
  third_party/video_depth_anything/   upstream network definition (Apache-2.0, see its LICENSE)
  stereo/
    dibr.py              disparity, z-buffered forward warp, hole filling, renderer
    formats.py           SBS / half / TB / anaglyph / RGB-D packing
    inpaint.py           AI gap filling (Stable Diffusion inpainting, foreground-excluded masks, tiling)
  io/
    image_io.py          JPG/PNG/WebP/HEIC load, save, 16-bit depth PNG, MPO writer
    video_io.py          ffprobe, threaded FFmpeg decode/encode, segment join + audio mux
  pipeline/
    common.py            engine (device + cached models), batch file discovery
    photo.py             photo conversion
    video.py             streaming, segmented, resumable video conversion
  gui/
    app.py               starts the desktop app
    main_window.py       queue, settings, preview, presets, log
    jobs.py              batch queue: one `parallaxa convert --progress json` child process per file
    preview.py           live preview (depth once, re-render on slider changes) on a worker thread
    options.py           GUI settings <-> CLI arguments, presets storage (%APPDATA%\Parallaxa)
tests/                   unit + end-to-end tests (run offline with debug models; GUI tests run offscreen)
```

### Troubleshooting

* **`hevc_nvenc not available`**: run `parallaxa info` to see FFmpeg's reason. If it says
  *"Driver does not support the required nvenc API version"*, your FFmpeg is newer than your GPU
  driver. FFmpeg 9.x needs NVIDIA driver **610+**; FFmpeg 8.x works with **570+**. Either update
  the driver or install FFmpeg 8.x (`winget install Gyan.FFmpeg --version <8.x>`, or put an 8.x
  `bin` folder at `parallaxa\ffmpeg\bin`, which takes priority over PATH). Until then, videos
  are encoded on the CPU (same quality, slower).
* **`.venv\Scripts\activate` not found in PowerShell**: create the venv first, then run
  `.\.venv\Scripts\Activate.ps1`.
* **`CUDA available: False`**: the CPU-only PyTorch got installed. Reinstall it inside the venv
  from the `cu128` index (setup step 3).

### Known limitations

* Video Depth Anything Large doesn't fit in 12 GB of VRAM. Small is the practical choice on an
  RTX 3060.
* A resumed conversion may show a very slight change in depth scale at the join. The saved
  anchor keeps this small.
* Processing is 8-bit SDR. HDR (PQ/HLG) sources are converted without tone mapping and will look
  washed out (you get a warning).
* Variable-frame-rate phone videos are converted to constant frame rate, which keeps audio in sync.
* AI gap filling is per frame, so on video it's slow and can flicker. Basic filling remains the
  video default. A true video-diffusion filler (StereoCrafter-style) needs more than 12 GB of VRAM
  and gated weights, so it isn't included.
