"""Frame loading, with the exact crop VGGT-World trained on.

224x448 is not negotiable if you ever want to swap in the paper's checkpoint --
it is the resolution both the KITTI and Cityscapes models were trained at. The
resize-then-centre-crop here is copied from `eval/kitti_val_short.py` so that
frames prepared by this repo are byte-comparable with theirs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

TARGET_H, TARGET_W = 224, 448
_EXTS = (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG")


def load_frame(path: str | Path, target_h: int = TARGET_H, target_w: int = TARGET_W) -> np.ndarray:
    """-> (H, W, 3) float32 in [0, 1]."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = max(target_h / h, target_w / w)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    img = img.resize((new_w, new_h), Image.BICUBIC)
    left, top = (new_w - target_w) // 2, (new_h - target_h) // 2
    img = img.crop((left, top, left + target_w, top + target_h))
    return np.asarray(img, dtype=np.float32) / 255.0


def find_frames(root: str | Path) -> list[Path]:
    """Accept a KITTI drive root, an `image_02/data` folder, or any flat dir."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(root)

    # A KITTI drive: descend to the left colour camera.
    for probe in (root / "image_02" / "data", root):
        if probe.is_dir():
            hits = sorted(p for p in probe.iterdir() if p.suffix in _EXTS)
            if hits:
                return hits

    hits = sorted(p for p in root.rglob("*") if p.suffix in _EXTS and "image_02" in p.parts)
    if hits:
        return hits
    raise FileNotFoundError(f"no images under {root}")


# KITTI native is 1242x375. Resizing to height 224 gives width 742, so the
# 448-wide centre crop keeps only 60% of the horizontal field of view -- a
# vehicle entering from the side is invisible until it is already well into the
# frame. FULL_FOV keeps all of it at the same vertical resolution; 742 = 53*14
# and 224 = 16*14, so both are exact multiples of VGGT's patch size.
CROP_FOV = (224, 448)   # what VGGT-World trained on; needed for its checkpoint
FULL_FOV = (224, 742)   # same scale, nothing thrown away


def load_sequence(
    root: str | Path,
    *,
    start: int = 0,
    count: int | None = None,
    stride: int = 2,
    target_h: int = TARGET_H,
    target_w: int = TARGET_W,
) -> tuple[np.ndarray, list[Path]]:
    """Load a strided run of frames.

    stride=2 is the default because that is what VGGT-World's own evaluation
    uses on KITTI -- 10 Hz capture sampled every other frame, so an effective
    5 Hz. A bigger gap between frames means more motion, which means a more
    demanding forecast and a stronger signal.
    """
    paths = find_frames(root)
    picked = paths[start::stride]
    if count is not None:
        picked = picked[:count]
    if len(picked) < 3:
        raise ValueError(f"need at least 3 frames, got {len(picked)} from {root}")
    frames = np.stack([load_frame(p, target_h, target_w) for p in picked], axis=0)
    return frames, picked
