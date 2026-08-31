"""The baseline that decides whether the 5 GB transformer is earning its place.

Classical independent-motion detection, and it predates all of this by decades:
compute dense optical flow, fit the ego-motion that a rigid scene would produce,
and flag the flow that does not fit. No depth, no foundation model, no weights
at all -- Farneback and a RANSAC fundamental matrix, both in OpenCV.

The ego model is the fundamental matrix rather than a homography. A homography
only describes a planar scene or a purely rotating camera, and a car driving
forward past buildings is neither. With F, a static point's correspondence must
lie on its epipolar line, so the perpendicular distance from that line -- the
Sampson error -- is exactly "how much of this pixel's motion cannot be explained
by the camera moving through a rigid world". That is the same question the depth
residual asks, in a different space.

Worth knowing before reading the numbers: this shares the depth method's blind
spot rather than avoiding it. For a forward-moving camera the epipolar lines
radiate from the focus of expansion, so motion along the viewing ray slides
along its own epipolar line and stays invisible here too. Neither method sees a
car pulling away in your own lane.

Everything downstream -- robust normalisation, blob extraction, tracking -- is
the same code the depth method uses, so the comparison isolates the residual.
"""

from __future__ import annotations

import cv2
import numpy as np

from .surprise import SurpriseMap, score_from_residual

# Farneback settings tuned for 224x448 automotive frames: a coarse pyramid
# because inter-frame motion at 5 Hz is large, and a wide window because the
# road surface is low-texture.
_FARNEBACK = dict(pyr_scale=0.5, levels=4, winsize=21, iterations=5,
                  poly_n=7, poly_sigma=1.5, flags=0)


def _gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if img.dtype != np.uint8:
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return img


def _texture_mask(gray: np.ndarray, percentile: float = 25.0) -> np.ndarray:
    """Where flow is worth believing.

    Farneback invents plausible-looking flow in textureless regions -- sky,
    smooth tarmac, blown-out highlights -- and that invented flow will not fit
    any ego model, so it becomes false surprise. Gate on local gradient energy.
    """
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    energy = cv2.GaussianBlur(np.hypot(gx, gy), (0, 0), 2.0)
    return energy >= np.percentile(energy, percentile)


def _sampson(F: np.ndarray, p0: np.ndarray, p1: np.ndarray) -> np.ndarray:
    """Sampson distance to the epipolar constraint, per correspondence."""
    n = len(p0)
    x0 = np.concatenate([p0, np.ones((n, 1))], axis=1)
    x1 = np.concatenate([p1, np.ones((n, 1))], axis=1)
    Fx0 = x0 @ F.T
    Ftx1 = x1 @ F
    num = np.einsum("ij,ij->i", x1, Fx0) ** 2
    den = Fx0[:, 0] ** 2 + Fx0[:, 1] ** 2 + Ftx1[:, 0] ** 2 + Ftx1[:, 1] ** 2
    return np.sqrt(num / np.maximum(den, 1e-9))


def flow_residual(
    img_prev: np.ndarray,
    img_cur: np.ndarray,
    *,
    texture_percentile: float = 25.0,
    ransac_thresh: float = 1.5,
    sample_stride: int = 4,
    border: int = 8,
    normalise: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """-> (residual, valid). Sampson error, optionally range-normalised.

    `normalise` exists for fairness. The depth method's residual is *relative*
    (|dD|/D), so it is automatically scale-aware: a 2 m error at 80 m counts for
    little, the same error at 4 m counts for a lot. Raw Sampson error is in
    absolute pixels, so a distant object's epipolar deviation is tiny however
    fast it moves, and comparing the two would be handicapping the baseline on a
    technicality rather than on substance.

    For a forward-moving camera, flow magnitude goes as 1/depth, so dividing by
    it recovers the same range normalisation without needing depth -- which
    would defeat the purpose of a depth-free baseline.
    """
    g0, g1 = _gray(img_prev), _gray(img_cur)
    h, w = g0.shape

    flow = cv2.calcOpticalFlowFarneback(g0, g1, None, **_FARNEBACK)

    vv, uu = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    p0 = np.stack([uu.ravel(), vv.ravel()], axis=1).astype(np.float32)
    p1 = p0 + flow.reshape(-1, 2)

    valid = _texture_mask(g0, texture_percentile)
    if border > 0:
        keep = np.zeros((h, w), bool)
        keep[border:-border, border:-border] = True
        valid &= keep
    valid &= np.isfinite(flow).all(axis=2)

    # Fit the ego model on a strided, textured subset. RANSAC needs the
    # majority of its input to be static, which strided sampling over the whole
    # frame gives us -- most of any street scene is buildings and road.
    sub = np.zeros((h, w), bool)
    sub[::sample_stride, ::sample_stride] = True
    sel = (sub & valid).ravel()
    if sel.sum() < 200:
        return np.zeros((h, w), np.float32), np.zeros((h, w), bool)

    F, inliers = cv2.findFundamentalMat(
        p0[sel], p1[sel], cv2.FM_RANSAC, ransac_thresh, 0.999, 5000
    )
    if F is None or F.shape != (3, 3):
        return np.zeros((h, w), np.float32), np.zeros((h, w), bool)

    resid = _sampson(F, p0, p1).reshape(h, w).astype(np.float32)
    if normalise:
        mag = np.linalg.norm(flow, axis=2)
        scale = np.maximum(cv2.GaussianBlur(mag, (0, 0), 4.0), 0.5)
        resid = resid / scale
    resid[~valid] = 0.0
    return resid, valid


def flow_surprise(
    img_prev: np.ndarray,
    img_cur: np.ndarray,
    *,
    threshold: float = 4.0,
    smooth_sigma: float = 1.5,
    **kw,
) -> SurpriseMap:
    """Flow-residual surprise map, normalised exactly like the depth one."""
    resid, valid = flow_residual(img_prev, img_cur, **kw)
    return score_from_residual(
        resid, valid, smooth_sigma=smooth_sigma, threshold=threshold,
        normalised_blur=True,
    )
