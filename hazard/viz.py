"""Rendering: overlays, side-by-side panels, contact sheets, video.

Everything here writes BGR uint8 so it can go straight to cv2.imwrite.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .detect import Blob
from .surprise import SurpriseMap

# Kept deliberately unsaturated so the heat overlay reads on top of it.
_DEPTH_CMAP = cv2.COLORMAP_TURBO
_HEAT_CMAP = cv2.COLORMAP_INFERNO


def _to_bgr(frame_rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor((frame_rgb * 255).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


def colorize_depth(depth: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    """Percentile-normalised depth, so one far outlier cannot flatten the image."""
    d = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    m = valid if valid is not None else np.isfinite(d) & (d > 0)
    if not m.any():
        return np.zeros((*d.shape, 3), np.uint8)
    lo, hi = np.percentile(d[m], [5, 95])
    if hi <= lo:
        hi = lo + 1e-6
    norm = np.clip((d - lo) / (hi - lo), 0, 1)
    img = cv2.applyColorMap((norm * 255).astype(np.uint8), _DEPTH_CMAP)
    img[~m] = 0
    return img


def overlay_surprise(
    frame_rgb: np.ndarray,
    sm: SurpriseMap,
    blobs: list[Blob],
    *,
    threshold: float = 4.0,
    vmax: float = 12.0,
    alpha: float = 0.55,
    min_age: int = 2,
) -> np.ndarray:
    """The money shot: original frame, heat where the forecast failed, boxes on
    anything that failed consistently."""
    base = _to_bgr(frame_rgb)
    score = np.clip(sm.score, 0, vmax) / vmax
    heat = cv2.applyColorMap((score * 255).astype(np.uint8), _HEAT_CMAP)

    # Only tint where it is actually surprising -- a full-frame wash hides the
    # signal instead of showing it.
    mask = (sm.score > threshold * 0.6) & sm.valid
    out = base.copy()
    weight = (score * alpha)[..., None] * mask[..., None]
    out = (base * (1 - weight) + heat * weight).astype(np.uint8)

    for b in blobs:
        if b.age < min_age:
            continue
        x, y, w, h = b.box
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 255), 2)
        label = f"#{b.track_id} z={b.peak:.0f}"
        cv2.putText(out, label, (x, max(12, y - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (0, 255, 255), 1, cv2.LINE_AA)
    return out


def banner(img: np.ndarray, text: str, height: int = 22) -> np.ndarray:
    """Caption strip above an image."""
    strip = np.full((height, img.shape[1], 3), 24, np.uint8)
    cv2.putText(strip, text, (6, height - 7), cv2.FONT_HERSHEY_SIMPLEX,
                0.42, (235, 235, 235), 1, cv2.LINE_AA)
    return np.vstack([strip, img])


def panel(
    frame_rgb: np.ndarray,
    pred_depth: np.ndarray,
    obs_depth: np.ndarray,
    sm: SurpriseMap,
    blobs: list[Blob],
    frame_idx: int,
    *,
    threshold: float = 4.0,
) -> np.ndarray:
    """Four-up diagnostic: what we expected, what we saw, where they disagreed."""
    cells = [
        banner(_to_bgr(frame_rgb), f"frame {frame_idx}  (observed image)"),
        banner(colorize_depth(pred_depth, sm.valid), "forecast depth (from the past only)"),
        banner(colorize_depth(obs_depth, sm.valid), "VGGT depth (what actually happened)"),
        banner(
            overlay_surprise(frame_rgb, sm, blobs, threshold=threshold),
            f"surprise  peak z={sm.peak:.1f}  area={sm.area * 100:.2f}%",
        ),
    ]
    top = np.hstack(cells[:2])
    bot = np.hstack(cells[2:])
    return np.vstack([top, bot])


def contact_sheet(images: list[np.ndarray], cols: int = 2) -> np.ndarray:
    """Grid of the most surprising moments."""
    if not images:
        return np.zeros((64, 64, 3), np.uint8)
    h, w = images[0].shape[:2]
    rows = (len(images) + cols - 1) // cols
    sheet = np.full((rows * h, cols * w, 3), 18, np.uint8)
    for i, img in enumerate(images):
        r, c = divmod(i, cols)
        sheet[r * h:(r + 1) * h, c * w:(c + 1) * w] = cv2.resize(img, (w, h))
    return sheet


def write_video(frames: list[np.ndarray], path: Path, fps: float = 5.0) -> bool:
    if not frames:
        return False
    h, w = frames[0].shape[:2]
    path.parent.mkdir(parents=True, exist_ok=True)
    for tag in ("avc1", "mp4v"):
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*tag), fps, (w, h))
        if writer.isOpened():
            for f in frames:
                writer.write(f)
            writer.release()
            return True
    return False
