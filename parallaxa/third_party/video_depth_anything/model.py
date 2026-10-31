# Copyright (2025) Bytedance Ltd. and/or its affiliates; modifications (c) the Parallaxa authors.
# Licensed under the Apache License, Version 2.0 (see LICENSE in this folder).
"""Minimal VideoDepthAnything module: DINOv2 encoder + temporal DPT head.

Parameter names (`pretrained.*`, `head.*`) match the upstream checkpoints so they load strictly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dinov2 import DINOv2
from .dpt_temporal import DPTHeadTemporal

CONFIGS = {
    "vits": {"features": 64, "out_channels": [48, 96, 192, 384], "layers": [2, 5, 8, 11]},
    "vitb": {"features": 128, "out_channels": [96, 192, 384, 768], "layers": [2, 5, 8, 11]},
    "vitl": {"features": 256, "out_channels": [256, 512, 1024, 1024], "layers": [4, 11, 17, 23]},
}


class VideoDepthAnything(nn.Module):
    def __init__(self, encoder: str = "vits", num_frames: int = 32):
        super().__init__()
        cfg = CONFIGS[encoder]
        self.layers = cfg["layers"]
        self.pretrained = DINOv2(model_name=encoder)
        self.head = DPTHeadTemporal(self.pretrained.embed_dim, cfg["features"], False,
                                    out_channels=cfg["out_channels"], use_clstoken=False,
                                    num_frames=num_frames, pe="ape")

    def forward(self, x: torch.Tensor, backbone_chunk: int = 8) -> torch.Tensor:
        """x: (B, T, 3, H, W) normalised, H and W multiples of 14 -> relative inverse depth (B, T, H, W)."""
        b, t, _, h, w = x.shape
        flat = x.flatten(0, 1)
        # The encoder treats frames independently, so run it in chunks to cap peak memory.
        chunks = [self.pretrained.get_intermediate_layers(flat[i:i + backbone_chunk], self.layers, return_class_token=True)
                  for i in range(0, flat.shape[0], backbone_chunk)]
        features = [tuple(torch.cat([c[level][j] for c in chunks]) for j in range(2)) for level in range(len(self.layers))]
        depth = self.head(features, h // 14, w // 14, t)[0]
        depth = F.interpolate(depth, size=(h, w), mode="bilinear", align_corners=True)
        return F.relu(depth).squeeze(1).unflatten(0, (b, t))
