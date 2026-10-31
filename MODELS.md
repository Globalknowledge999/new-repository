# Models and licences

Parallaxa's own code is MIT-licensed. The neural networks it downloads have their **own**
licences. Several are **non-commercial only** (CC-BY-NC-4.0), which is fine for personal use.
Don't sell or commercially distribute output made with them without checking the terms.

Weights are not stored in this repository. They download from Hugging Face on first use into
`%USERPROFILE%\.cache\huggingface` (set `HF_HOME` to put them somewhere else).

## Used now

| Key | Model | Params | Licence (weights) | Used for |
|---|---|---|---|---|
| `da2-small` | [Depth Anything V2 Small](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf) | ~25M | Apache-2.0 | **default for video** (`fast` profile) |
| `da2-base` | [Depth Anything V2 Base](https://huggingface.co/depth-anything/Depth-Anything-V2-Base-hf) | ~98M | CC-BY-NC-4.0 | `balanced` profile |
| `da2-large` | [Depth Anything V2 Large](https://huggingface.co/depth-anything/Depth-Anything-V2-Large-hf) | ~335M | CC-BY-NC-4.0 | **default for photos** |

These are loaded through Hugging Face `transformers` (Apache-2.0). Paper: Yang et al.,
*Depth Anything V2*, NeurIPS 2024.

| Key | Model | Params | Licence (weights) | Used for |
|---|---|---|---|---|
| `vda-small` | [Video Depth Anything Small](https://huggingface.co/depth-anything/Video-Depth-Anything-Small) | ~29M | Apache-2.0 | `balanced` and `quality` profiles |
| `vda-base` | [Video Depth Anything Base](https://huggingface.co/depth-anything/Video-Depth-Anything-Base) | ~113M | CC-BY-NC-4.0 | optional |
| `vda-large` | [Video Depth Anything Large](https://huggingface.co/depth-anything/Video-Depth-Anything-Large) | ~382M | CC-BY-NC-4.0 | optional; needs well over 12 GB VRAM at 518 px |

Paper: Chen et al., *Video Depth Anything*, CVPR 2025. The network definition is vendored in
`parallaxa/third_party/video_depth_anything/` under its Apache-2.0 licence (code), with small,
marked modifications (fused attention, no easydict, quiet start-up), so the published
checkpoints load unchanged. The streaming inference around it is Parallaxa's own.

### AI gap filling

| Key | Model | Licence (weights) | Used for |
|---|---|---|---|
| `sd15` | [Stable Diffusion 1.5 inpainting](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-inpainting) | CreativeML OpenRAIL-M (use restrictions; personal use fine) | **default** (`--fill-model auto` tries it first) |
| `sdxl` | [SDXL 1.0 inpainting](https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1) | CreativeML Open RAIL++-M | first fallback; finer detail, ~7 GB, slower |
| `sd2` | [Stable Diffusion 2 inpainting](https://huggingface.co/stabilityai/stable-diffusion-2-inpainting) | CreativeML Open RAIL++-M | last fallback. Returned "Repository Not Found" on 2026-10-31, so it may no longer be public |

These are loaded with Hugging Face `diffusers` (Apache-2.0) as safetensors, which is plain data
with no executable code, in fp16 (SD 1.5: ~2 GB download, ~3 GB VRAM).

**Considered and not used:** LaMa as a TorchScript file. It's fast, but that format contains
executable code, and the copy available is repackaged by a third party. StereoCrafter: video
diffusion built on Stable Video Diffusion, with gated weights, and it needs well over 12 GB.

## Planned / possible

| Phase | Model | Licence | Why |
|---|---|---|---|
| later | [StereoCrafter](https://github.com/TencentARC/StereoCrafter) (Tencent ARC) | Tencent custom licence | Temporally consistent generative fill for video, if a bigger GPU is ever available |

Alternatives considered and not used by default:

* **Depth Pro** (Apple): sharp metric depth, but a custom research-only licence and slow per frame.
* **DepthCrafter** (Tencent): excellent temporal consistency, but diffusion-based and slow.
  Its licence is non-commercial too.
* **Marigold**: diffusion-based and slow for video. Its quality on photos is similar to DA-V2 Large.
* **MiDaS / DPT**: MIT, but clearly weaker than DA-V2.

## Debug models (no download)

* `debug-gradient`: depth is a vertical ramp. Useful to check FFmpeg/output wiring.
* `debug-tiny` (gap filling): a miniature randomly initialised inpainting pipeline, for offline tests.
* `debug-vda`: a randomly initialised Video Depth Anything Small network, for offline tests.
* `debug-random`: a tiny randomly initialised Depth-Anything-shaped network, so the real code
  path runs without any download. The output depth is meaningless.
