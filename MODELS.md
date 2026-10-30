# Models and licences

Parallaxa's own code is MIT-licensed. The neural networks it downloads have their **own**
licences. Several are **non-commercial only** (CC-BY-NC-4.0), which is fine for personal use.
Don't sell or commercially distribute output made with them without checking the terms.

Weights are not stored in this repository. They download from Hugging Face on first use into
`%USERPROFILE%\.cache\huggingface` (set `HF_HOME` to put them somewhere else).

## Used now (phase 1)

| Key | Model | Params | Licence (weights) | Used for |
|---|---|---|---|---|
| `da2-small` | [Depth Anything V2 Small](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf) | ~25M | Apache-2.0 | `fast` profile |
| `da2-base` | [Depth Anything V2 Base](https://huggingface.co/depth-anything/Depth-Anything-V2-Base-hf) | ~98M | CC-BY-NC-4.0 | `balanced` profile |
| `da2-large` | [Depth Anything V2 Large](https://huggingface.co/depth-anything/Depth-Anything-V2-Large-hf) | ~335M | CC-BY-NC-4.0 | **default for photos and (until phase 2) video** |

These are loaded through Hugging Face `transformers` (Apache-2.0). Paper: Yang et al.,
*Depth Anything V2*, NeurIPS 2024.

## Planned

| Phase | Model | Licence | Why |
|---|---|---|---|
| 2 | [Video Depth Anything](https://github.com/DepthAnything/Video-Depth-Anything) Small / Large (ByteDance, CVPR 2025) | Small: Apache-2.0; Base/Large: CC-BY-NC-4.0; code Apache-2.0 | Temporally consistent depth for video, built on the DA-V2 encoder plus a temporal head. It handles long videos with overlapping windows, which is exactly what phase 2's chunking needs. |
| 4 | [StereoCrafter](https://github.com/TencentARC/StereoCrafter) (Tencent ARC) | Tencent custom licence (research / non-commercial terms; read before use) | Generative inpainting of disocclusions for the high-quality mode. Heavy: it needs an SVD-based diffusion backbone, so expect it to be slow on 12 GB. |
| 4 (alt.) | A smaller image-inpainting model (e.g. LaMa, Apache-2.0) | Apache-2.0 | Cheaper "HQ fill" for photos if StereoCrafter is too slow on an RTX 3060. |

Alternatives considered and not used by default:

* **Depth Pro** (Apple): sharp metric depth, but a custom research-only licence and slow per frame.
* **DepthCrafter** (Tencent): excellent temporal consistency, but diffusion-based and slow.
  Its licence is non-commercial too.
* **Marigold**: diffusion-based and slow for video. Its quality on photos is similar to DA-V2 Large.
* **MiDaS / DPT**: MIT, but clearly weaker than DA-V2.

## Debug models (no download)

* `debug-gradient`: depth is a vertical ramp. Useful to check FFmpeg/output wiring.
* `debug-random`: a tiny randomly initialised Depth-Anything-shaped network, so the real code
  path runs without any download. The output depth is meaningless.
