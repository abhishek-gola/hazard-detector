"""Forecasters: given the past, guess the geometry of the next frame.

The detector does not care *how* the guess is made, only that the guess uses
nothing but past frames. Two implementations ship here:

  rigid     Treat the world as static, extrapolate the camera at constant
            velocity, and warp the previous depth map forward. Needs no weights
            beyond VGGT itself, and it is the baseline any learned forecaster
            has to beat.

  identity  Predict that nothing changes at all. Deliberately bad -- it exists
            so you can see how much of the signal is really coming from the
            geometry rather than from frame-to-frame noise.

  vggtworld The learned flow-matching forecaster from the paper. Needs the
            OneDrive checkpoint, which is not publicly fetchable, so this path
            is wired up but unverified. See README.

All of them return a `Forecast`: a predicted depth map plus a validity mask,
because a forward warp cannot say anything about pixels nothing landed on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np

from .backbone import WindowGeometry


@dataclass
class Forecast:
    depth: np.ndarray  # (H, W) predicted depth for the target frame
    valid: np.ndarray  # (H, W) bool -- False where the forecast has nothing to say
    conf: np.ndarray  # (H, W) confidence carried over from the source frame
    flow: np.ndarray | None = None  # (H, W) pixels this content moved to get here


@dataclass
class ForecastContext:
    """What a forecaster is allowed to look at."""

    geom: WindowGeometry  # depths/poses for the whole window, one scale
    frames: np.ndarray  # (S, H, W, 3) the raw window
    target: int  # index of the frame being predicted

    def past(self, k: int) -> int:
        """Index of the frame k steps before the target. `past(1)` is the last
        frame actually observed."""
        idx = self.target - k
        if idx < 0:
            raise IndexError(f"forecaster asked for frame {idx}, before the window")
        return idx


class Forecaster(Protocol):
    name: str
    n_context: int

    def forecast(self, ctx: ForecastContext) -> Forecast: ...


# --------------------------------------------------------------------------
# geometry helpers
# --------------------------------------------------------------------------


def _to_4x4(extrinsic_3x4: np.ndarray) -> np.ndarray:
    """OpenCV 3x4 cam-from-world -> 4x4."""
    out = np.eye(4, dtype=np.float64)
    out[:3, :4] = extrinsic_3x4
    return out


def _pixel_grid(h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    v, u = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    return u.astype(np.float64), v.astype(np.float64)


def _unproject(depth: np.ndarray, K: np.ndarray, cam_to_world: np.ndarray) -> np.ndarray:
    """(H, W) depth -> (H, W, 3) world points."""
    h, w = depth.shape
    u, v = _pixel_grid(h, w)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    x = (u - cx) / fx * depth
    y = (v - cy) / fy * depth
    z = depth
    cam = np.stack([x, y, z, np.ones_like(z)], axis=-1)  # (H, W, 4)
    world = cam.reshape(-1, 4) @ cam_to_world.T
    return world[:, :3].reshape(h, w, 3)


def _splat_forward(
    world_pts: np.ndarray,
    src_conf: np.ndarray,
    K: np.ndarray,
    world_to_cam: np.ndarray,
    shape: tuple[int, int],
) -> Forecast:
    """Project world points into a new camera and z-buffer them into an image.

    Forward warping, not backward: we know where each *source* pixel goes, not
    where each *target* pixel came from. Collisions are resolved by painter's
    algorithm -- sort far-to-near and let the nearest surface write last.
    """
    h, w = shape
    pts = world_pts.reshape(-1, 3)
    conf_flat = src_conf.reshape(-1)

    hom = np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=1)
    cam = hom @ world_to_cam.T  # (N, 4)
    z = cam[:, 2]

    in_front = z > 1e-6
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = fx * cam[:, 0] / z + cx
        v = fy * cam[:, 1] / z + cy

    # Points behind the camera divide by ~0 and land on NaN. Screen them out
    # before the integer cast, which would otherwise warn on every frame.
    finite = in_front & np.isfinite(u) & np.isfinite(v) & np.isfinite(z)
    u = np.where(finite, u, -1.0)
    v = np.where(finite, v, -1.0)

    ui = np.round(u).astype(np.int64)
    vi = np.round(v).astype(np.int64)
    ok = finite & (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)

    depth_buf = np.zeros(h * w, dtype=np.float32)
    conf_buf = np.zeros(h * w, dtype=np.float32)
    flow_buf = np.zeros(h * w, dtype=np.float32)
    hit = np.zeros(h * w, dtype=bool)

    # How far each pixel's content travelled. This is the missing term in the
    # noise model: a fractional error in the extrapolated pose produces a
    # reprojection error proportional to the displacement, so a pixel that moved
    # 40 px is inherently far less trustworthy than one that moved 2. It is what
    # makes the wide periphery of a forward-moving camera so noisy -- parallax
    # there is enormous -- and without it a full-field-of-view frame drowns in
    # false alarms at the edges.
    u_src, v_src = _pixel_grid(h, w)
    disp = np.hypot(u - u_src.reshape(-1), v - v_src.reshape(-1))

    if ok.any():
        idx = (vi[ok] * w + ui[ok]).astype(np.int64)
        zz = z[ok].astype(np.float32)
        cc = conf_flat[ok].astype(np.float32)
        dd = disp[ok].astype(np.float32)
        order = np.argsort(-zz)  # far first, so near overwrites
        idx, zz, cc, dd = idx[order], zz[order], cc[order], dd[order]
        depth_buf[idx] = zz
        conf_buf[idx] = cc
        flow_buf[idx] = dd
        hit[idx] = True

    return Forecast(
        depth=depth_buf.reshape(h, w),
        valid=hit.reshape(h, w),
        conf=conf_buf.reshape(h, w),
        flow=flow_buf.reshape(h, w),
    )


def _fill_disocclusions(fc: Forecast, max_radius: int = 2) -> Forecast:
    """Patch the thin holes a forward warp always leaves.

    When the camera moves, surfaces that were hidden behind a nearer object come
    into view, and nothing in the source frame maps onto them. Those slivers sit
    right along depth discontinuities. They should be filled with the *farther*
    neighbour, since what gets revealed is background -- so this is a grayscale
    dilation, not an erosion. Holes bigger than `max_radius` are left invalid:
    guessing there would manufacture surprise exactly where we least want it.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    depth = fc.depth.copy()
    conf = fc.conf.copy()
    valid = fc.valid.copy()

    for _ in range(max_radius):
        holes = ~valid
        if not holes.any():
            break
        far = cv2.dilate(depth, kernel)  # max filter == farther surface
        far_conf = cv2.dilate(conf, kernel)
        reachable = cv2.dilate(valid.astype(np.uint8), kernel).astype(bool) & holes
        depth[reachable] = far[reachable]
        conf[reachable] = far_conf[reachable]
        valid = valid | reachable

    return Forecast(depth=depth, valid=valid, conf=conf, flow=fc.flow)


# --------------------------------------------------------------------------
# forecasters
# --------------------------------------------------------------------------


class RigidFlowForecaster:
    """Static-world forecast: extrapolate the camera, warp the depth forward.

    This is the workhorse. It says: assume every surface you can see is bolted
    to the ground, and assume the car keeps doing what it was doing. Wherever
    that assumption holds, the prediction is near perfect. Wherever something
    moved under its own power, it breaks -- and it breaks locally, at the thing
    that moved. That residual is the whole detector.
    """

    name = "rigid"
    n_context = 2

    def __init__(self, fill_radius: int = 2):
        self.fill_radius = fill_radius

    def forecast(self, ctx: ForecastContext) -> Forecast:
        g = ctx.geom
        prev, prev2 = ctx.past(1), ctx.past(2)

        T_prev = _to_4x4(g.extrinsic[prev])  # world -> cam
        T_prev2 = _to_4x4(g.extrinsic[prev2])
        C_prev = np.linalg.inv(T_prev)  # cam -> world
        C_prev2 = np.linalg.inv(T_prev2)

        # Constant velocity in SE(3): whatever moved us from prev2 to prev,
        # apply it once more. Crude, but over a 0.2 s step it is close, and the
        # residual it leaves is global rather than local -- which the robust
        # normalisation in surprise.py divides straight back out.
        motion = C_prev @ np.linalg.inv(C_prev2)
        C_pred = motion @ C_prev
        T_pred = np.linalg.inv(C_pred)

        K = g.intrinsic[prev]
        world = _unproject(g.depth[prev].astype(np.float64), K, C_prev)
        fc = _splat_forward(world, g.conf[prev], K, T_pred, g.depth[prev].shape)
        return _fill_disocclusions(fc, self.fill_radius)


class IdentityForecaster:
    """Predict that the next frame looks exactly like the last one.

    An ablation, not a serious proposal. If the rigid forecaster's detections
    look like this one's, the pipeline is measuring ego-motion, not hazards.
    """

    name = "identity"
    n_context = 1

    def forecast(self, ctx: ForecastContext) -> Forecast:
        prev = ctx.past(1)
        d = ctx.geom.depth[prev].astype(np.float32)
        return Forecast(
            depth=d,
            valid=np.isfinite(d) & (d > 0),
            conf=ctx.geom.conf[prev].astype(np.float32),
        )


class VGGTWorldForecaster:
    """The paper's learned forecaster.

    Wired up against the released code path but NOT verified, because the
    checkpoint lives on a OneDrive folder share that needs an interactive login.
    Download `kitti_checkpoint.pt` by hand, drop it in `weights/`, install the
    extra deps (`pip install -r requirements-vggtworld.txt`) and pass
    `--forecaster vggtworld`.

    Unlike the rigid forecaster this one does not read `ctx.geom` at all -- it
    runs its own VGGT pass, forecasts in token space, and decodes depth through
    VGGT's frozen head, exactly as `demo/kitti_demo.py` does upstream.
    """

    name = "vggtworld"
    n_context = 2

    def __init__(self, ckpt: str, device, config_name: str = "default_kitti",
                 steps: int = 20):
        from hydra import compose, initialize_config_dir
        from hydra.utils import instantiate
        import torch
        from pathlib import Path

        cfg_dir = str((Path(__file__).resolve().parent.parent / "vggt_world_config"))
        with initialize_config_dir(version_base=None, config_dir=cfg_dir):
            cfg = compose(config_name=config_name)

        model = instantiate(cfg.model, _recursive_=False)
        blob = torch.load(ckpt, map_location="cpu")
        state = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[vggtworld] missing={len(missing)} unexpected={len(unexpected)}")

        model.to(device).eval()
        self.model = model
        self.device = device
        self.steps = steps

    def forecast(self, ctx: ForecastContext) -> Forecast:
        import torch

        prev2, prev = ctx.past(2), ctx.past(1)
        window = ctx.frames[[prev2, prev, ctx.target, ctx.target]]
        images = torch.from_numpy(window).permute(0, 3, 1, 2)[None]
        images = images.to(self.device)

        m = self.model
        with torch.no_grad():
            cond, patch_start_idx = m.aggregator.part1(images[:, :2])
            tgt, _ = m.aggregator.part1(images)
            x1 = torch.cat(tgt, dim=1)[:, 2:4]

            b, t, n, c = x1.shape
            fm_dtype = next(m.fm.parameters()).dtype
            shape_like = torch.zeros((b, t, n, c), device=self.device, dtype=fm_dtype)

            ps = m.aggregator.patch_size
            ph, pw = (ps, ps) if isinstance(ps, int) else ps
            h, w = window.shape[1:3]

            gen = m.fm.sample_euler(
                cond_layers=cond,
                shape_like=shape_like,
                steps=self.steps,
                patch_hw=(h // ph, w // pw),
            )
            combo = torch.cat(
                [torch.cat(cond, dim=1)[:, 0:2], torch.cat(gen, dim=1)], dim=1
            )
            agg_dtype = next(m.aggregator.parameters()).dtype
            stages, _ = m.aggregator.part2([combo.to(agg_dtype)])

            head_dtype = next(m.depth_head.parameters()).dtype
            depth, conf = m.depth_head(
                [s.to(head_dtype) for s in stages],
                images=images.to(head_dtype),
                patch_start_idx=patch_start_idx,
            )

        d = depth[0, 2, :, :, 0].float().cpu().numpy()
        c = conf[0, 2].float().cpu().numpy()
        return Forecast(depth=d, valid=np.isfinite(d) & (d > 0), conf=c)


def build_forecaster(name: str, device=None, ckpt: str | None = None, **kw):
    if name == "rigid":
        return RigidFlowForecaster(**kw)
    if name == "identity":
        return IdentityForecaster()
    if name == "vggtworld":
        if not ckpt:
            raise ValueError("--forecaster vggtworld needs --vggtworld-ckpt")
        return VGGTWorldForecaster(ckpt, device, **kw)
    raise ValueError(f"unknown forecaster {name!r}")
