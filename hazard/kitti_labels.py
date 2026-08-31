"""Read KITTI raw tracklets, and work out which objects are actually moving.

Tracklets are the right ground truth for this project because the drives we run
on *are* KITTI raw drives -- no sequence re-mapping, no separate benchmark. But
they answer the wrong question on their own: they label every car, including
the dozens of parked ones lining every residential street. A parked car is not
a hazard and the detector is right to ignore it, so scoring against raw
tracklets would punish correct behaviour.

What we need is the moving subset. Tracklet poses are given in the velodyne
(ego) frame, so a parked car still appears to drift backwards as you drive past
it. The fix does not need GPS: in any KITTI street scene the *majority* of
annotated objects are stationary, so the median apparent velocity across all
tracklets in a frame is a robust estimate of ego motion. Subtract it, and
whatever still has velocity is genuinely moving.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Things that can move under their own power. 'Misc' and 'Tram' are in the
# KITTI vocabulary too but are rare enough not to matter here.
DYNAMIC_TYPES = {"Car", "Van", "Truck", "Pedestrian", "Cyclist", "Person (sitting)", "Tram"}


@dataclass
class Tracklet:
    object_type: str
    h: float
    w: float
    l: float
    first_frame: int
    positions: np.ndarray  # (M, 3) tx, ty, tz in velodyne coords
    rotations: np.ndarray  # (M, 3) rx, ry, rz
    states: np.ndarray  # (M,) occlusion/truncation state codes
    moving: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))

    @property
    def frames(self) -> np.ndarray:
        return np.arange(self.first_frame, self.first_frame + len(self.positions))

    def at(self, frame: int) -> np.ndarray | None:
        i = frame - self.first_frame
        if 0 <= i < len(self.positions):
            return self.positions[i]
        return None

    def is_moving_at(self, frame: int) -> bool:
        i = frame - self.first_frame
        return bool(0 <= i < len(self.moving) and self.moving[i])


def _float(node, tag: str, default: float = 0.0) -> float:
    el = node.find(tag)
    return float(el.text) if el is not None and el.text else default


def parse_tracklets(xml_path: str | Path) -> list[Tracklet]:
    root = ET.parse(str(xml_path)).getroot()
    container = root.find("tracklets")
    if container is None:
        raise ValueError(f"{xml_path} has no <tracklets> element")

    out: list[Tracklet] = []
    for item in container.findall("item"):
        poses = item.find("poses")
        if poses is None:
            continue
        pos, rot, st = [], [], []
        for p in poses.findall("item"):
            pos.append([_float(p, "tx"), _float(p, "ty"), _float(p, "tz")])
            rot.append([_float(p, "rx"), _float(p, "ry"), _float(p, "rz")])
            st.append(_float(p, "occlusion", 0.0))
        if not pos:
            continue
        type_el = item.find("objectType")
        out.append(
            Tracklet(
                object_type=type_el.text if type_el is not None else "Unknown",
                h=_float(item, "h"),
                w=_float(item, "w"),
                l=_float(item, "l"),
                first_frame=int(_float(item, "first_frame")),
                positions=np.asarray(pos, dtype=np.float64),
                rotations=np.asarray(rot, dtype=np.float64),
                states=np.asarray(st, dtype=np.float64),
            )
        )
    return out


def _fit_rigid_2d(pos: np.ndarray, vel: np.ndarray) -> tuple[np.ndarray, float] | None:
    """Least-squares 2D rigid motion field: v = t + omega x p.

    A single translation is not enough. The moment the car turns, a *parked*
    object's apparent velocity in the ego frame picks up a rotational term that
    grows with range -- which is why a median-translation model calls every
    distant parked car "moving" through every junction. In 2D:

        vx = tx - omega * py
        vy = ty + omega * px

    Three unknowns, two equations per object, so two objects would do and three
    give some slack.
    """
    n = len(pos)
    if n < 3:
        return None
    A = np.zeros((2 * n, 3))
    b = np.zeros(2 * n)
    A[0::2, 0] = 1.0
    A[0::2, 2] = -pos[:, 1]
    A[1::2, 1] = 1.0
    A[1::2, 2] = pos[:, 0]
    b[0::2] = vel[:, 0]
    b[1::2] = vel[:, 1]
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    return sol, float(np.linalg.norm(A @ sol - b))


def _rigid_residuals(pos: np.ndarray, vel: np.ndarray, sol: np.ndarray) -> np.ndarray:
    pred = np.stack(
        [sol[0] - sol[2] * pos[:, 1], sol[1] + sol[2] * pos[:, 0]], axis=1
    )
    return np.linalg.norm(vel[:, :2] - pred, axis=1)


def mark_moving(
    tracklets: list[Tracklet],
    *,
    speed_thresh: float = 0.45,
    smooth: int = 3,
    trim: float = 0.35,
) -> list[Tracklet]:
    """Flag, per frame, which tracklets move independently of the ego vehicle.

    `speed_thresh` is metres per frame of *residual* motion once ego motion is
    removed. KITTI raw runs at 10 Hz, so 0.45 m/frame is about 16 km/h -- high
    enough to shrug off annotation jitter on parked cars, low enough that
    anything flagged is unambiguously in motion. Deliberately conservative:
    for a ground truth, precision matters more than coverage.

    Ego motion is fitted as a full 2D rigid field rather than a translation,
    and fitted twice -- the second time after discarding the worst `trim`
    fraction, so that a few genuinely moving cars cannot drag the estimate of
    what "stationary" looks like.
    """
    vel: dict[int, np.ndarray] = {}
    for i, t in enumerate(tracklets):
        v = np.zeros_like(t.positions)
        if len(t.positions) > 1:
            v[1:] = np.diff(t.positions, axis=0)
            v[0] = v[1]
        vel[i] = v

    all_frames = [f for t in tracklets for f in t.frames]
    if not all_frames:
        return tracklets
    lo, hi = min(all_frames), max(all_frames)

    ego: dict[int, np.ndarray | None] = {}
    for f in range(lo, hi + 1):
        idx = [i for i, t in enumerate(tracklets)
               if t.first_frame <= f < t.first_frame + len(t.positions)]
        if len(idx) < 3:
            ego[f] = None
            continue
        pos = np.stack([tracklets[i].positions[f - tracklets[i].first_frame] for i in idx])
        vv = np.stack([vel[i][f - tracklets[i].first_frame] for i in idx])

        fit = _fit_rigid_2d(pos[:, :2], vv[:, :2])
        if fit is None:
            ego[f] = None
            continue
        sol, _ = fit
        # Refit on the quietest objects only.
        res = _rigid_residuals(pos[:, :2], vv[:, :2], sol)
        keep = res <= np.quantile(res, 1.0 - trim)
        if keep.sum() >= 3:
            refit = _fit_rigid_2d(pos[keep, :2], vv[keep, :2])
            if refit is not None:
                sol = refit[0]
        ego[f] = sol

    for i, t in enumerate(tracklets):
        residual = np.zeros(len(t.positions))
        for j, f in enumerate(t.frames):
            sol = ego.get(int(f))
            if sol is None:
                continue
            p = t.positions[j][None, :2]
            v = vel[i][j][None, :2]
            residual[j] = float(_rigid_residuals(p, v, sol)[0])
        if smooth > 1 and len(residual) >= smooth:
            k = np.ones(smooth) / smooth
            residual = np.convolve(residual, k, mode="same")
        t.moving = (residual > speed_thresh) & np.isin(
            [t.object_type] * len(residual), list(DYNAMIC_TYPES)
        )
    return tracklets


def moving_frames(tracklets: list[Tracklet]) -> dict[int, list[Tracklet]]:
    """frame index -> the tracklets that are moving in it."""
    out: dict[int, list[Tracklet]] = {}
    for t in tracklets:
        for f in t.frames:
            if t.is_moving_at(int(f)):
                out.setdefault(int(f), []).append(t)
    return out


def summarise(tracklets: list[Tracklet]) -> str:
    from collections import Counter

    types = Counter(t.object_type for t in tracklets)
    mov = moving_frames(tracklets)
    n_moving_tracklets = sum(1 for t in tracklets if t.moving.any())
    lines = [
        f"{len(tracklets)} tracklets: " + ", ".join(f"{k}x{v}" for k, v in types.most_common()),
        f"{n_moving_tracklets} of them move at some point",
        f"{len(mov)} frames contain at least one moving object",
    ]
    if mov:
        keys = sorted(mov)
        runs, start, prev = [], keys[0], keys[0]
        for f in keys[1:]:
            if f != prev + 1:
                runs.append((start, prev))
                start = f
            prev = f
        runs.append((start, prev))
        span = ", ".join(f"{a}-{b}" for a, b in runs if b - a >= 2)
        lines.append(f"moving-object frame ranges: {span or '(none longer than 2 frames)'}")
    return "\n".join(lines)
