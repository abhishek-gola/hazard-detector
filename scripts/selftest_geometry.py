#!/usr/bin/env python3
"""Prove the detector works before spending a single GPU-second on it.

We build a synthetic street: a ground plane, a wall, and one box. The depth maps
are ray-traced analytically, so they are exact -- nothing here goes through
VGGT, and nothing goes through the same code path the forecaster uses. Then:

  A. the box stays put while the camera drives forward
     -> the static-world forecast should be almost perfect everywhere

  B. the box slides sideways into the lane
     -> the forecast should break, and break *only where the box is*

If B does not light up the box, the idea does not work and no amount of neural
network will save it. Run this first; it takes about a second.

    python scripts/selftest_geometry.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hazard.backbone import WindowGeometry  # noqa: E402
from hazard.forecast import ForecastContext, RigidFlowForecaster  # noqa: E402
from hazard.surprise import compute_surprise  # noqa: E402

H, W = 224, 448
GROUND_Y = 1.65  # camera is 1.65 m above the road; +y points down
FAR = 120.0


def intrinsics(h: int = H, w: int = W, fov_deg: float = 70.0) -> np.ndarray:
    f = (w / 2) / np.tan(np.radians(fov_deg) / 2)
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], dtype=np.float64)


def pose_forward(z: float) -> np.ndarray:
    """Camera looking down +z, translated z metres along the road. Returns the
    3x4 OpenCV cam-from-world extrinsic."""
    R = np.eye(3)
    t = np.array([0.0, 0.0, -z])  # X_cam = R X_world + t
    return np.hstack([R, t[:, None]])


def _rays(K: np.ndarray, cam_to_world: np.ndarray):
    v, u = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    d = np.stack(
        [(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u, float)],
        axis=-1,
    )
    origin = cam_to_world[:3, 3]
    world_dir = d @ cam_to_world[:3, :3].T
    return origin, world_dir


def _hit_plane(origin, direction, y0: float) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        s = (y0 - origin[1]) / direction[:, :, 1]
    s[~np.isfinite(s)] = np.inf
    s[s <= 0] = np.inf
    return s


def _hit_aabb(origin, direction, lo, hi) -> np.ndarray:
    s_lo = np.full(direction.shape[:2], -np.inf)
    s_hi = np.full(direction.shape[:2], np.inf)
    for ax in range(3):
        d = direction[:, :, ax]
        with np.errstate(divide="ignore", invalid="ignore"):
            t0 = (lo[ax] - origin[ax]) / d
            t1 = (hi[ax] - origin[ax]) / d
        t0, t1 = np.minimum(t0, t1), np.maximum(t0, t1)
        s_lo = np.maximum(s_lo, np.nan_to_num(t0, nan=-np.inf))
        s_hi = np.minimum(s_hi, np.nan_to_num(t1, nan=np.inf))
    out = np.where((s_hi >= np.maximum(s_lo, 0)) & (s_lo > 0), s_lo, np.inf)
    return out


def render(extrinsic: np.ndarray, K: np.ndarray, box_center_x: float) -> np.ndarray:
    """Exact depth map of {ground plane, back wall, one box}."""
    T = np.eye(4)
    T[:3, :4] = extrinsic
    C = np.linalg.inv(T)
    origin, direction = _rays(K, C)

    s = _hit_plane(origin, direction, GROUND_Y)
    s = np.minimum(s, _hit_plane(origin, direction, -6.0))  # ceiling-ish wall
    # back wall at z = 60
    with np.errstate(divide="ignore", invalid="ignore"):
        s_wall = (60.0 - origin[2]) / direction[:, :, 2]
    s_wall[~np.isfinite(s_wall) | (s_wall <= 0)] = np.inf
    s = np.minimum(s, s_wall)

    lo = np.array([box_center_x - 0.9, GROUND_Y - 1.6, 17.0])
    hi = np.array([box_center_x + 0.9, GROUND_Y, 21.0])
    s = np.minimum(s, _hit_aabb(origin, direction, lo, hi))

    return np.clip(np.where(np.isfinite(s), s, FAR), 0.1, FAR).astype(np.float32)


def build(box_x: list[float]) -> WindowGeometry:
    K = intrinsics()
    speed = 1.4  # metres between sampled frames, ~25 km/h at 5 Hz
    extr = np.stack([pose_forward(i * speed) for i in range(len(box_x))])
    depth = np.stack([render(extr[i], K, box_x[i]) for i in range(len(box_x))])
    return WindowGeometry(
        depth=depth,
        conf=np.ones_like(depth),
        extrinsic=extr.astype(np.float64),
        intrinsic=np.stack([K] * len(box_x)),
        seconds=0.0,
    )


def evaluate(name: str, box_x: list[float]) -> dict:
    geom = build(box_x)
    frames = np.zeros((len(box_x), H, W, 3), np.float32)
    fc = RigidFlowForecaster().forecast(
        ForecastContext(geom=geom, frames=frames, target=2)
    )
    sm = compute_surprise(fc, geom, 2, border=8, smooth_sigma=1.0)

    # Where is the box in the final frame?
    K = geom.intrinsic[2]
    lo = np.array([box_x[2] - 0.9, GROUND_Y - 1.6, 17.0])
    hi = np.array([box_x[2] + 0.9, GROUND_Y, 21.0])
    T = np.eye(4)
    T[:3, :4] = geom.extrinsic[2]
    origin, direction = _rays(K, np.linalg.inv(T))
    box_mask = np.isfinite(_hit_aabb(origin, direction, lo, hi))
    bg = sm.valid & ~box_mask
    on = sm.valid & box_mask

    res = {
        "name": name,
        "median_residual": sm.frame_median,
        "score_on_box": float(sm.score[on].mean()) if on.any() else 0.0,
        "score_off_box": float(sm.score[bg].mean()) if bg.any() else 0.0,
        "peak": sm.peak,
        "area_pct": sm.area * 100,
        "box_px": int(box_mask.sum()),
    }
    print(
        f"  {name:<28} median_resid={res['median_residual']:.4f}  "
        f"mean z on box={res['score_on_box']:6.2f}  off box={res['score_off_box']:5.2f}  "
        f"area>4={res['area_pct']:5.2f}%"
    )
    return res


def main() -> int:
    print("synthetic street, exact ray-traced depth, no neural network involved\n")

    print("A. static box, camera drives forward")
    static = evaluate("static scene", [0.0, 0.0, 0.0])

    print("\nB. box slides 1.1 m sideways into the lane over two frames")
    moving = evaluate("box cuts in", [-2.2, -1.1, 0.0])

    print("\n" + "-" * 68)
    ok = True

    if static["area_pct"] > 1.0:
        print(f"FAIL  static scene should be almost entirely unsurprising, "
              f"got {static['area_pct']:.2f}% above threshold")
        ok = False
    else:
        print(f"PASS  static scene stays quiet ({static['area_pct']:.2f}% flagged)")

    contrast = moving["score_on_box"] / max(moving["score_off_box"], 1e-3)
    if contrast < 3.0:
        print(f"FAIL  moving box only {contrast:.1f}x brighter than background")
        ok = False
    else:
        print(f"PASS  moving box is {contrast:.1f}x brighter than its background")

    if moving["score_on_box"] < static["score_on_box"] * 3:
        print("FAIL  motion did not raise the score on the box")
        ok = False
    else:
        print(f"PASS  motion raised the box score "
              f"{moving['score_on_box'] / max(static['score_on_box'], 1e-3):.1f}x "
              f"vs the same box standing still")

    print("-" * 68)
    print("the geometry is sound\n" if ok else "something is wrong with the warp\n")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
