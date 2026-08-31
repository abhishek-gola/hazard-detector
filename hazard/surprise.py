"""Turn a wrong forecast into a surprise map.

The raw residual between forecast and observation is not usable on its own. It
is large everywhere the car hit a bump, everywhere the sun moved, everywhere
VGGT's own depth is shaky. All of that is *global* error, and the job of this
module is to divide it out so that only *local* error survives.

The tool for that is a robust z-score. Take the median residual over the frame
and the median absolute deviation around it, and express every pixel as "how
many MADs above the frame's own typical error am I". A jolt raises the median
for every pixel at once and cancels. A pedestrian stepping off a kerb does not.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .backbone import WindowGeometry
from .forecast import Forecast


@dataclass
class SurpriseMap:
    score: np.ndarray  # (H, W) robust z-score, 0 where invalid
    valid: np.ndarray  # (H, W) bool
    residual: np.ndarray  # (H, W) raw relative depth error, for debugging
    frame_median: float  # the global error that was divided out
    frame_mad: float
    peak: float  # 99.5th percentile of score, a frame-level headline number
    area: float  # fraction of valid pixels above `threshold`


def _relative_residual(pred: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """|pred - obs| / obs, matching the abs_rel convention the paper reports.

    Dividing by the observed depth is what stops far-away scenery from
    dominating: a two-metre error at 80 m is a rounding error, the same two
    metres at 4 m is a car in your lane.
    """
    denom = np.maximum(obs, 1e-3)
    return np.abs(pred - obs) / denom


def _log_depth_gradient(depth: np.ndarray) -> np.ndarray:
    """Fractional depth change per pixel: |grad log D|.

    Central differences on log depth, so the result is scale-free -- which
    matters because VGGT's depth is only defined up to an unknown global factor.
    A value of 0.1 means depth changes by ~10% from one pixel to the next.
    """
    log_d = np.log(np.maximum(depth, 1e-3)).astype(np.float32)
    gx = cv2.Sobel(log_d, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(log_d, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    return np.hypot(gx, gy)


def expected_residual_scale(
    obs: np.ndarray,
    pred: np.ndarray,
    conf: np.ndarray,
    *,
    subpixel: float = 1.0,
    rel_floor: float = 0.03,
    conf_ref: float = 4.0,
    flow: np.ndarray | None = None,
    pose_rel_err: float = 0.02,
) -> np.ndarray:
    """How large a relative residual we should expect even if nothing moved.

    This replaces the hard depth-edge mask, and it is the difference between a
    heuristic and a statistic.

    Two error sources. First, registration: the forward warp lands on integer
    pixels and the constant-velocity pose is approximate, so the reprojected
    position is off by some sub-pixel amount `delta`. At a place where depth
    changes by a fraction g per pixel, that misregistration alone produces a
    relative residual of about `delta * g`. Second, VGGT's own depth noise,
    which is roughly a fixed fraction of depth and grows where confidence is
    low.

    Adding them in quadrature gives a per-pixel expected residual. Dividing the
    observed residual by it yields a quantity that is ~1 everywhere the static
    world holds -- flat road, textured wall, *and* the silhouette of a parked
    car. Steep depth gradients stop being untrustworthy and become correctly
    discounted, which is what lets small distant objects survive at all: the old
    binary mask deleted a 6 px band and took most of a distant car with it.
    """
    grad = np.maximum(_log_depth_gradient(obs), _log_depth_gradient(pred))
    # Registration error has a floor (`subpixel`) plus a part that grows with
    # how far the content actually moved: a fractional pose error `pose_rel_err`
    # over a displacement of d pixels misplaces content by pose_rel_err * d.
    delta = subpixel
    if flow is not None:
        delta = np.sqrt(subpixel ** 2 + (pose_rel_err * flow) ** 2)
    geometric = delta * grad
    photometric = rel_floor * np.sqrt(conf_ref / np.maximum(conf, 0.5))
    return np.sqrt(geometric ** 2 + photometric ** 2) + 1e-6


def depth_edge_mask(
    depth: np.ndarray,
    *,
    rel_step: float = 0.12,
    dilate: int = 2,
) -> np.ndarray:
    """Pixels sitting on a depth discontinuity, which must never be trusted.

    This is the single biggest source of false alarms, and it is a property of
    forward warping rather than of the scene. Where depth jumps from 8 m to
    40 m across one pixel -- a car's silhouette, the edge of a pole -- being
    off by a single pixel produces a 400% relative error that means nothing at
    all. Left alone it draws a bright outline around every object in the frame
    and buries the real signal.

    So: take the morphological gradient of *log* depth, which measures
    fractional change and is therefore immune to VGGT's unknown global scale,
    and drop anything above `rel_step`. Silhouettes get masked; the interiors
    they enclose do not, and the interior is where a genuinely mispredicted
    object lights up.
    """
    log_d = np.log(np.maximum(depth, 1e-3)).astype(np.float32)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    grad = cv2.morphologyEx(log_d, cv2.MORPH_GRADIENT, k)
    edge = (grad > rel_step).astype(np.uint8)
    if dilate > 0:
        edge = cv2.dilate(edge, k, iterations=dilate)
    return edge.astype(bool)


def compute_surprise(
    forecast: Forecast,
    geom: WindowGeometry,
    target: int,
    *,
    conf_percentile: float = 25.0,
    border: int = 8,
    smooth_sigma: float = 1.5,
    threshold: float = 4.0,
    max_depth_ratio: float = 40.0,
    edge_rel_step: float = 0.12,
    edge_dilate: int = 2,
    whiten: bool = True,
    subpixel: float = 1.0,
    rel_floor: float = 0.03,
    extreme_grad: float = 1.0,
    normalised_blur: bool = True,
    use_flow_term: bool = True,
) -> SurpriseMap:
    """Compare a forecast against what VGGT actually measured.

    conf_percentile  drop the least-confident pixels of the frame. Sky, glare
                     and motion blur live down there, and so do most false
                     alarms.
    border           forward warping is unreliable at the frame edge, where
                     content flows out of view. Ignore a margin.
    max_depth_ratio  drop pixels far beyond the frame's median depth -- VGGT
                     puts sky at enormous distances and the relative residual
                     there is meaningless.
    whiten           divide the residual by its expected size from
                     `expected_residual_scale` instead of hard-masking depth
                     edges. Keeps the pixels; discounts them correctly.
    extreme_grad     with whitening on, the only pixels still deleted outright
                     are true occlusion boundaries -- depth changing by this
                     fraction or more across a single pixel, where the
                     linearisation behind the noise model breaks down.
    normalised_blur  smooth score*valid and valid separately then divide, so
                     that masked neighbours do not drag down the pixels next to
                     them. Without this a small object ringed by masked pixels
                     is blurred toward zero before it is ever thresholded.
    """
    obs = geom.depth[target].astype(np.float32)
    obs_conf = geom.conf[target].astype(np.float32)
    pred = forecast.depth.astype(np.float32)

    h, w = obs.shape
    valid = forecast.valid & np.isfinite(obs) & (obs > 0) & np.isfinite(pred) & (pred > 0)

    # Confidence gate, computed on this frame's own distribution so it adapts
    # to lighting rather than needing an absolute number.
    both_conf = np.minimum(obs_conf, forecast.conf)
    if valid.any():
        conf_floor = np.percentile(both_conf[valid], conf_percentile)
        valid &= both_conf >= conf_floor

    # Sky / far-field gate.
    if valid.any():
        med_depth = float(np.median(obs[valid]))
        valid &= obs < med_depth * max_depth_ratio

    # Depth discontinuities. Two ways to handle them, and they are not equal.
    if whiten:
        # Keep the pixels and weight them by how much error we should expect
        # there. Only genuine occlusion boundaries are dropped, undilated.
        grad = np.maximum(_log_depth_gradient(obs), _log_depth_gradient(pred))
        valid &= grad < extreme_grad
    elif edge_rel_step > 0:
        # The old way: delete a dilated band around every depth step. Cheap,
        # and it takes most of a distant vehicle with it.
        edges = depth_edge_mask(obs, rel_step=edge_rel_step, dilate=edge_dilate)
        edges |= depth_edge_mask(pred, rel_step=edge_rel_step, dilate=edge_dilate)
        valid &= ~edges

    if border > 0:
        edge = np.zeros((h, w), dtype=bool)
        edge[border:-border, border:-border] = True
        valid &= edge

    residual = _relative_residual(pred, obs)
    if whiten:
        sigma = expected_residual_scale(
            obs, pred, np.minimum(obs_conf, forecast.conf),
            subpixel=subpixel, rel_floor=rel_floor,
            flow=forecast.flow if use_flow_term else None,
        )
        residual = residual / sigma
    residual[~valid] = 0.0

    return score_from_residual(
        residual, valid, smooth_sigma=smooth_sigma, threshold=threshold,
        normalised_blur=normalised_blur,
    )


def score_from_residual(
    residual: np.ndarray,
    valid: np.ndarray,
    *,
    smooth_sigma: float = 1.5,
    threshold: float = 4.0,
    normalised_blur: bool = True,
) -> SurpriseMap:
    """Robust per-frame normalisation, shared by every residual space.

    Split out so that the depth-residual method and the optical-flow baseline
    go through *identical* downstream code. If they did not, a comparison
    between them would be measuring my two implementations rather than the two
    residuals, and the baseline would be worth nothing.
    """
    h, w = residual.shape
    residual = residual.copy()
    residual[~valid] = 0.0

    if valid.sum() < 64:
        empty = np.zeros((h, w), dtype=np.float32)
        return SurpriseMap(empty, valid, residual, 0.0, 0.0, 0.0, 0.0)

    vals = residual[valid]
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)))
    scale = 1.4826 * mad + 1e-6

    score = (residual - med) / scale
    score[~valid] = 0.0
    score = np.clip(score, 0.0, None)

    if smooth_sigma > 0:
        if normalised_blur:
            wv = valid.astype(np.float32)
            num = cv2.GaussianBlur(score * wv, (0, 0), smooth_sigma)
            den = cv2.GaussianBlur(wv, (0, 0), smooth_sigma)
            score = np.where(den > 1e-3, num / np.maximum(den, 1e-3), 0.0)
        else:
            score = cv2.GaussianBlur(score, (0, 0), smooth_sigma)
        score = score.astype(np.float32)
        score[~valid] = 0.0

    peak = float(np.percentile(score[valid], 99.5)) if valid.any() else 0.0
    area = float((score[valid] > threshold).mean()) if valid.any() else 0.0
    return SurpriseMap(
        score=score.astype(np.float32), valid=valid,
        residual=residual.astype(np.float32),
        frame_median=med, frame_mad=mad, peak=peak, area=area,
    )
