"""Video Depth Anything network definition (ByteDance, Apache-2.0).

Vendored from https://github.com/DepthAnything/Video-Depth-Anything (commit 4f5ae23) so the
published checkpoints load unchanged. Only the model definition is included; Parallaxa's own
streaming inference lives in parallaxa/depth/video_depth_anything.py.

Modifications (marked "Parallaxa modification" in the source):
  * dinov2_layers/attention.py: fused scaled_dot_product_attention in Attention.forward.
  * dpt_temporal.py: plain dict instead of easydict.EasyDict.
  * four files: the "xFormers not available" message is silenced (xFormers is optional).
  * model.py: new file, a minimal VideoDepthAnything module (same parameter names as upstream).
See LICENSE in this folder.
"""
