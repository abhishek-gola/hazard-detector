"""VGGT as a measurement device.

We only need three things out of VGGT: per-frame metric-ish depth, a per-pixel
confidence, and the camera pose. So we build the trunk plus two heads and skip
the point head and the track head entirely -- that is roughly 300 M parameters
we never allocate, which matters on a 16 GB machine.

One thing to be careful about: VGGT's global attention runs across every frame
in the window, so a frame's depth depends on which other frames share its
window. Depths and poses are only comparable *within a single forward pass*.
The whole pipeline is built around that constraint -- see `pipeline.py`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from vggt.heads.camera_head import CameraHead
from vggt.heads.dpt_head import DPTHead
from vggt.models.aggregator import Aggregator
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

from .device import free_cache


@dataclass
class WindowGeometry:
    """Everything VGGT saw in one forward pass, all in one consistent frame.

    depth      (S, H, W)      float32, up to an unknown global scale
    conf       (S, H, W)      float32, higher is better; sky and glare score low
    extrinsic  (S, 3, 4)      OpenCV cam-from-world
    intrinsic  (S, 3, 3)      pixels
    """

    depth: np.ndarray
    conf: np.ndarray
    extrinsic: np.ndarray
    intrinsic: np.ndarray
    seconds: float

    def __len__(self) -> int:
        return self.depth.shape[0]


class VGGTCore(nn.Module):
    """Aggregator + camera head + depth head. No point head, no track head."""

    def __init__(self, img_size: int = 518, patch_size: int = 14, embed_dim: int = 1024):
        super().__init__()
        self.aggregator = Aggregator(
            img_size=img_size, patch_size=patch_size, embed_dim=embed_dim
        )
        self.camera_head = CameraHead(dim_in=2 * embed_dim)
        self.depth_head = DPTHead(
            dim_in=2 * embed_dim,
            output_dim=2,
            activation="exp",
            conf_activation="expp1",
        )

    @torch.no_grad()
    def forward(self, images: torch.Tensor, frames_chunk_size: int = 4):
        """images: (1, S, 3, H, W) in [0, 1]."""
        tokens, patch_start_idx = self.aggregator(images)

        # The camera head MUST see fp32. It refines the pose over four
        # iterations, and in bfloat16 that loop diverges to NaN every time --
        # silently, because the tokens going in are perfectly finite. The
        # symptom is a NaN focal length and an empty surprise map about three
        # stages downstream, which is a miserable thing to debug. Cast here.
        pose_enc = self.camera_head([t.float() for t in tokens])[-1]
        h, w = images.shape[-2:]
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc.float(), (h, w))

        # The DPT head is numerically touchy -- keep it in fp32 even when the
        # trunk ran in bf16, and chunk it so peak activation stays bounded.
        head_dtype = next(self.depth_head.parameters()).dtype
        depth, conf = self.depth_head(
            [t.to(head_dtype) for t in tokens],
            images=images.to(head_dtype),
            patch_start_idx=patch_start_idx,
            frames_chunk_size=frames_chunk_size,
        )
        return depth, conf, extrinsic, intrinsic


def load_backbone(
    weights: str | Path,
    device,
    trunk_dtype: torch.dtype = torch.bfloat16,
) -> VGGTCore:
    """Build VGGTCore and load facebook/VGGT-1B weights into it.

    The checkpoint carries point-head and track-head tensors we deliberately did
    not build; those come back as `unexpected` keys and are dropped. Anything
    reported as *missing* would be a real problem, so we fail loudly on that.
    """
    weights = Path(weights)
    if not weights.exists():
        raise FileNotFoundError(
            f"VGGT weights not found at {weights}\n"
            f"  run:  python scripts/fetch_weights.py"
        )

    model = VGGTCore()

    if weights.suffix == ".safetensors":
        from safetensors.torch import load_file

        state = load_file(str(weights))
    else:
        blob = torch.load(str(weights), map_location="cpu", weights_only=True)
        state = blob.get("model", blob) if isinstance(blob, dict) else blob

    wanted = set(model.state_dict().keys())
    filtered = {k: v for k, v in state.items() if k in wanted}
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    if missing:
        raise RuntimeError(
            f"checkpoint is missing {len(missing)} tensors the model needs, "
            f"first few: {missing[:5]}"
        )
    dropped = len(state) - len(filtered)

    model.eval()
    # Trunk in bf16 halves the resident weights; heads stay fp32.
    model.aggregator.to(device=device, dtype=trunk_dtype)
    model.camera_head.to(device=device, dtype=torch.float32)
    model.depth_head.to(device=device, dtype=torch.float32)
    for p in model.parameters():
        p.requires_grad_(False)

    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"[backbone] loaded {n_params / 1e9:.2f}B params "
        f"(trunk {trunk_dtype}, heads fp32); dropped {dropped} unused head tensors"
    )
    return model


@torch.no_grad()
def run_window(
    model: VGGTCore,
    frames: np.ndarray,
    device,
    trunk_dtype: torch.dtype = torch.bfloat16,
) -> WindowGeometry:
    """Run VGGT over one window of frames.

    frames: (S, H, W, 3) float32 in [0, 1].
    """
    t0 = time.time()
    images = torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous()
    images = images.unsqueeze(0).to(device=device, dtype=trunk_dtype)

    depth, conf, extrinsic, intrinsic = model(images)

    geom = WindowGeometry(
        depth=depth[0, ..., 0].float().cpu().numpy(),
        conf=conf[0].float().cpu().numpy(),
        extrinsic=extrinsic[0].float().cpu().numpy(),
        intrinsic=intrinsic[0].float().cpu().numpy(),
        seconds=time.time() - t0,
    )
    del images, depth, conf, extrinsic, intrinsic
    free_cache(device)
    return geom
