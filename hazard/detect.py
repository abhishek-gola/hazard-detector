"""From a surprise map to a list of things worth looking at.

Thresholding a heat map gives you blobs; most of them are one-frame flicker.
The filter that actually matters is persistence. A real object that broke the
static-world assumption keeps breaking it on the next frame too, and it does so
in roughly the same place. Noise does not. So we associate blobs across frames
by overlap and only report a track once it has survived a couple of frames.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .surprise import SurpriseMap


@dataclass
class Blob:
    frame: int
    box: tuple[int, int, int, int]  # x, y, w, h
    area_px: int
    peak: float
    mean: float
    centroid: tuple[float, float]
    track_id: int = -1
    age: int = 1


@dataclass
class Track:
    track_id: int
    blobs: list[Blob] = field(default_factory=list)
    last_frame: int = -1

    @property
    def length(self) -> int:
        return len(self.blobs)

    @property
    def peak(self) -> float:
        return max(b.peak for b in self.blobs)


def min_area_at_row(base_row: float, horizon: float, k: float,
                    floor: int, ceiling: int) -> float:
    """Smallest believable blob for an object whose base sits at `base_row`.

    A fixed pixel area is the wrong gate, and it is biased in the worst
    direction. On a flat road the range to a point whose image row is r goes as
    1 / (r - horizon), so an object's pixel height goes as (r - horizon) and its
    area as (r - horizon)^2. A constant `min_area = 60` therefore asks a car at
    40 m to produce as many above-threshold pixels as one at 10 m, when the
    whole object only covers a few hundred pixels and barely a third of those
    survive the validity masks.

    Scaling the requirement quadratically with distance below the horizon puts
    every range on the same footing.
    """
    d = max(float(base_row) - float(horizon), 1.0)
    return float(np.clip(k * d * d, floor, ceiling))


def extract_blobs(
    sm: SurpriseMap,
    frame_idx: int,
    *,
    threshold: float = 4.0,
    min_area: int = 60,
    open_ksize: int = 3,
    close_ksize: int = 7,
    row_scaled_area: bool = False,
    horizon_row: float | None = None,
    area_k: float = 0.01,
    area_floor: int = 8,
) -> list[Blob]:
    """Threshold, clean up, and label the surprise map."""
    mask = ((sm.score > threshold) & sm.valid).astype(np.uint8)
    if mask.sum() == 0:
        return []

    if open_ksize > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ksize, open_ksize))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    if close_ksize > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    horizon = sm.score.shape[0] / 2.0 if horizon_row is None else horizon_row

    blobs: list[Blob] = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        x, y = int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP])
        w, h = int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT])
        need = (min_area_at_row(y + h, horizon, area_k, area_floor, min_area)
                if row_scaled_area else min_area)
        if area < need:
            continue
        region = sm.score[labels == i]
        blobs.append(
            Blob(
                frame=frame_idx,
                box=(x, y, w, h),
                area_px=area,
                peak=float(region.max()),
                mean=float(region.mean()),
                centroid=(float(centroids[i][0]), float(centroids[i][1])),
            )
        )
    blobs.sort(key=lambda b: b.peak, reverse=True)
    return blobs


def _integral(score: np.ndarray, valid: np.ndarray):
    """Integral images of the masked score and of the valid mask itself.

    Two of them, because the mean has to be taken over *valid* pixels only.
    Roughly a third of any frame is masked out -- sky, low confidence, depth
    edges -- and dividing by the full box area instead would penalise a
    detection for sitting next to a masked region.
    """
    masked = (score * valid).astype(np.float64)
    return cv2.integral(masked), cv2.integral(valid.astype(np.float64))


def _window_means(S, V, bw: int, bh: int, stride: int):
    """Mean score for every box of size (bw, bh) on a grid, via integral image."""
    h, w = S.shape[0] - 1, S.shape[1] - 1
    ys = np.arange(0, max(h - bh + 1, 1), stride)
    xs = np.arange(0, max(w - bw + 1, 1), stride)
    if not len(ys) or not len(xs):
        return None
    Y, X = ys[:, None], xs[None, :]
    tot = S[Y + bh, X + bw] - S[Y, X + bw] - S[Y + bh, X] + S[Y, X]
    cnt = V[Y + bh, X + bw] - V[Y, X + bw] - V[Y + bh, X] + V[Y, X]
    return ys, xs, tot, cnt


def default_scales(fx: float = 305.0) -> list[tuple[int, int]]:
    """Box sizes covering a vehicle from ~8 m to ~45 m at 224x448.

    A 1.8 m wide car subtends 1.8 * fx / range pixels, so 8 m gives ~68 px and
    45 m gives ~12 px. Three aspect ratios because a car seen side-on is wide
    and one seen head-on is not.
    """
    heights = [12, 18, 26, 36, 50, 70]
    aspects = [0.8, 1.3, 2.0]
    out = []
    for h in heights:
        for a in aspects:
            w = int(round(h * a))
            if 6 <= w <= 200:
                out.append((w, h))
    return out


def detect_windows(
    sm: SurpriseMap,
    frame_idx: int,
    *,
    threshold: float = 3.0,
    scales: list[tuple[int, int]] | None = None,
    stride_frac: float = 0.34,
    min_valid_frac: float = 0.45,
    nms_iou: float = 0.20,
    max_per_frame: int = 5,
) -> list[Blob]:
    """Multi-scale region scoring. Kept for the record; it did NOT work.

    The reasoning was: the control experiment showed that *mean surprise inside
    a box* separates moving from parked vehicles at AUC 0.813, so score boxes
    that way and slide them over the frame. Clean, and directly optimises the
    quantity that was validated.

    Measured on a held-out segment it is worse than the blob chain it was meant
    to replace, at every matched false-alarm rate -- F1 0.206 against 0.344.

    The flaw is that the control used *ground-truth* boxes. Given the box, the
    mean inside it is a fine discriminator. But as a detector you must find the
    box, and box-mean is a bad objective for that: a mispredicted car produces a
    compact, intense patch, and averaging over a car-sized window divides that
    patch by an area that is mostly correctly predicted. Connected components
    adapt their region to the actual shape of the surprise, which turns out to
    matter more than scoring the region well.

    Left in the tree because "the validated statistic makes a bad detector
    objective" is worth being able to reproduce. Use `extract_blobs`.
    """
    if scales is None:
        scales = default_scales()

    S, V = _integral(sm.score, sm.valid)
    cands: list[Blob] = []

    for bw, bh in scales:
        stride = max(2, int(round(min(bw, bh) * stride_frac)))
        got = _window_means(S, V, bw, bh, stride)
        if got is None:
            continue
        ys, xs, tot, cnt = got
        need = min_valid_frac * bw * bh
        with np.errstate(divide="ignore", invalid="ignore"):
            means = np.where(cnt > 0, tot / np.maximum(cnt, 1.0), 0.0)
        hits = np.argwhere((cnt >= need) & (means >= threshold))
        for iy, ix in hits:
            y, x = int(ys[iy]), int(xs[ix])
            region = sm.score[y:y + bh, x:x + bw]
            region_valid = sm.valid[y:y + bh, x:x + bw]
            cands.append(
                Blob(
                    frame=frame_idx,
                    box=(x, y, bw, bh),
                    area_px=int(region_valid.sum()),
                    peak=float(region[region_valid].max()) if region_valid.any() else 0.0,
                    mean=float(means[iy, ix]),
                    centroid=(x + bw / 2.0, y + bh / 2.0),
                )
            )

    return _nms(cands, nms_iou)[:max_per_frame]


def _nms(blobs: list[Blob], iou_thresh: float) -> list[Blob]:
    """Keep the strongest box in each cluster of overlapping ones."""
    kept: list[Blob] = []
    for b in sorted(blobs, key=lambda b: b.mean, reverse=True):
        if all(_iou(b.box, k.box) <= iou_thresh for k in kept):
            kept.append(b)
    return kept


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x0, y0 = max(ax, bx), max(ay, by)
    x1, y1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    return inter / float(aw * ah + bw * bh - inter)


def _assoc_score(a: tuple[int, int, int, int], b: tuple[int, int, int, int],
                 iou_thresh: float, centre_frac: float) -> float:
    """How much two boxes look like the same object one frame apart.

    IoU alone is the wrong gate here, and it fails in exactly the worst place.
    At 5 Hz a car crossing a junction can move further than its own width
    between frames, giving IoU = 0 against its own previous box -- so the
    fastest, most hazardous objects were the ones the tracker refused to
    confirm, and the persistence filter then deleted them. A centroid gate
    scaled by box size catches those while still rejecting unrelated blobs at
    opposite ends of the frame.
    """
    iou = _iou(a, b)
    if iou > iou_thresh:
        return 1.0 + iou  # a real overlap always wins
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ca = np.array([ax + aw / 2, ay + ah / 2])
    cb = np.array([bx + bw / 2, by + bh / 2])
    span = max(np.hypot(aw, ah), np.hypot(bw, bh))
    d = float(np.linalg.norm(ca - cb))
    reach = centre_frac * span
    return max(0.0, 1.0 - d / reach) if reach > 0 else 0.0


class BlobTracker:
    """Greedy association on overlap, falling back to a size-scaled centroid gate."""

    def __init__(self, iou_thresh: float = 0.15, max_gap: int = 1,
                 centre_frac: float = 1.6):
        self.iou_thresh = iou_thresh
        self.max_gap = max_gap
        self.centre_frac = centre_frac
        self.tracks: dict[int, Track] = {}
        self._next_id = 0

    def update(self, blobs: list[Blob], frame_idx: int) -> list[Blob]:
        live = [
            t for t in self.tracks.values()
            if frame_idx - t.last_frame <= self.max_gap + 1
        ]
        taken: set[int] = set()

        for blob in blobs:
            best, best_iou = None, 0.05
            for t in live:
                if t.track_id in taken:
                    continue
                score = _assoc_score(blob.box, t.blobs[-1].box,
                                     self.iou_thresh, self.centre_frac)
                if score > best_iou:
                    best, best_iou = t, score
            if best is None:
                best = Track(track_id=self._next_id)
                self.tracks[self._next_id] = best
                self._next_id += 1
            taken.add(best.track_id)
            blob.track_id = best.track_id
            blob.age = best.length + 1
            best.blobs.append(blob)
            best.last_frame = frame_idx
        return blobs

    def confirmed(self, min_length: int = 2) -> list[Track]:
        out = [t for t in self.tracks.values() if t.length >= min_length]
        out.sort(key=lambda t: t.peak, reverse=True)
        return out
