"""Streaming Video Depth Anything schedule vs a direct port of the upstream offline algorithm."""

import numpy as np
import pytest
import torch

from parallaxa.depth.video_depth_anything import INTERP, OVERLAP, STEP, WINDOW, VideoDepthStream


class FakeNet(torch.nn.Module):
    """Deterministic stand-in: depth = f(frame) with a scale/offset that drifts per call, like a real
    relative-depth model whose scale is arbitrary per window."""

    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        a, b = 1.0 + 0.3 * self.calls, 0.1 * self.calls
        return torch.relu(x[:, :, 0] * a + b + x[:, :, 1] ** 2)


def upstream_reference(frames_in: list[torch.Tensor], net) -> list[torch.Tensor]:
    """Port of upstream infer_video_depth's windowing + alignment, minus resizing."""
    keyframes = [0, 12] + list(range(24, 32))
    n = len(frames_in)
    pad = (STEP - (n % STEP)) % STEP + (WINDOW - STEP)
    frames = frames_in + [frames_in[-1]] * pad
    depth_list, pre = [], None
    for s in range(0, n, STEP):
        cur = torch.stack(frames[s : s + WINDOW]).unsqueeze(0).clone()
        if pre is not None:
            cur[:, :OVERLAP] = pre[:, keyframes]
        depth_list += list(net(cur)[0])
        pre = cur
    aligned, ref = [], []
    align_len = OVERLAP - INTERP
    for fid in range(0, len(depth_list), WINDOW):
        if not aligned:
            aligned += depth_list[:WINDOW]
            ref = [depth_list[fid + k] for k in keyframes[:align_len]]
            continue
        cur = [depth_list[fid + i] for i in range(align_len)]
        p, q = torch.cat(cur).double().numpy().ravel(), torch.cat(ref).double().numpy().ravel()
        A = np.array([[p @ p, p.sum()], [p.sum(), p.size]])
        sc, sh = np.linalg.solve(A, np.array([p @ q, q.sum()]))
        post = [(depth_list[fid + i] * sc + sh).clamp(min=0) for i in range(align_len, OVERLAP)]
        pre_d = aligned[-INTERP:]
        w = np.linspace(0, 1, INTERP)
        aligned[-INTERP:] = [pre_d[i] * (1 - w[i]) + post[i] * w[i] for i in range(INTERP)]
        aligned += [(depth_list[fid + i] * sc + sh).clamp(min=0) for i in range(OVERLAP, WINDOW)]
        ref = ref[:1] + [(depth_list[fid + k] * sc + sh).clamp(min=0) for k in keyframes[1:align_len]]
    return aligned[:n]


@pytest.mark.parametrize("n,batch", [(1, 1), (7, 3), (31, 4), (32, 4), (33, 5), (54, 4), (55, 8), (100, 6), (200, 4), (199, 1)])
def test_stream_matches_upstream_schedule(n, batch):
    torch.manual_seed(n)
    h, w = 28, 42  # already multiples of 14, so preprocessing is a plain normalisation
    raw = [torch.rand(3, h, w) for _ in range(n)]

    stream = VideoDepthStream(FakeNet(), torch.device("cpu"), torch.float32, depth_res=h)
    outs = []
    for i in range(0, n, batch):
        outs.append(stream.push(torch.stack(raw[i : i + batch])))
    outs.append(stream.flush())
    got = torch.cat([o for o in outs if o.numel()])

    prepped = [stream._prep(f[None])[0] for f in raw]  # same normalisation the stream applied
    ref = torch.stack(upstream_reference(prepped, FakeNet()))
    assert got.shape == ref.shape == (n, h, w)
    assert torch.allclose(got.float(), ref.float(), atol=1e-4)


def test_stream_latency_is_bounded():
    stream = VideoDepthStream(FakeNet(), torch.device("cpu"), torch.float32, depth_res=28)
    emitted = 0
    for i in range(200):
        emitted += stream.push(torch.rand(1, 3, 28, 42)).shape[0]
        assert (i + 1) - emitted <= WINDOW + STEP  # frames held back never exceed ~two windows
        assert len(stream.inputs) <= WINDOW
    emitted += stream.flush().shape[0]
    assert emitted == 200
