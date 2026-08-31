#!/usr/bin/env python3
"""Is the surprise signal real, or is it just noise from VGGT's depth head?

The objection is fair and it deserves a control, not an argument. Here is the
one that settles it.

**Parked cars and moving cars are both cars.** Same paint, same glass, same
silhouette, same depth discontinuities, same everything that makes a monocular
depth model wobble. If the score were driven by depth-head noise, a parked car
and a moving car at the same distance would score the same, because noise has
no idea which is which.

So we skip the detector entirely -- no threshold, no blobs, no tracking -- and
read the raw surprise map inside every annotated vehicle box in the drive.
Then:

  1. moving vs parked, matched by range
  2. AUC: how well does the raw score rank moving above parked
  3. a random-box control at matched sizes, for a floor

Nothing here can be tuned. The boxes come from KITTI, the maps come from a run
that never saw them.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hazard.kitti_calib import KittiCalib, project_box  # noqa: E402
from hazard.kitti_labels import DYNAMIC_TYPES, mark_moving, parse_tracklets  # noqa: E402

RNG = np.random.default_rng(0)


def box_score(score: np.ndarray, valid: np.ndarray,
              box: tuple[int, int, int, int], erode: int = 2) -> float | None:
    """Mean surprise over a box's interior.

    Eroded, because the silhouette of any object is a depth discontinuity and
    those are masked out anyway -- what we want is whether the *body* of the
    thing was mispredicted.
    """
    x, y, w, h = box
    m = np.zeros(score.shape, np.uint8)
    m[y:y + h, x:x + w] = 1
    if erode > 0 and min(w, h) > 2 * erode + 2:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * erode + 1, 2 * erode + 1))
        m = cv2.erode(m, k)
    sel = (m > 0) & valid
    if sel.sum() < 20:
        return None
    return float(score[sel].mean())


def auc(pos: list[float], neg: list[float]) -> float:
    """Probability a random positive outranks a random negative (Mann-Whitney)."""
    if not pos or not neg:
        return float("nan")
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort().astype(float) + 1
    # average ties
    order = np.argsort(allv)
    sorted_v = allv[order]
    i = 0
    while i < len(sorted_v):
        j = i
        while j + 1 < len(sorted_v) and sorted_v[j + 1] == sorted_v[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2
        i = j + 1
    rp = ranks[:len(pos)].sum()
    return (rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--drive", required=True)
    ap.add_argument("--calib", required=True)
    args = ap.parse_args()

    maps_dir = Path(args.run) / "maps"
    if not maps_dir.is_dir():
        print(f"error: {maps_dir} missing -- re-run with --save-maps", file=sys.stderr)
        return 1

    tracklets = mark_moving(parse_tracklets(Path(args.drive) / "tracklet_labels.xml"))
    calib = KittiCalib(args.calib)

    moving, parked, random_ctl = [], [], []
    rows = []

    for npz_path in sorted(maps_dir.glob("*.npz")):
        raw = int(npz_path.stem)
        d = np.load(npz_path)
        score, valid = d["score"].astype(np.float32), d["valid"]
        h, w = score.shape

        for t in tracklets:
            p = t.at(raw)
            if p is None or t.object_type not in DYNAMIC_TYPES:
                continue
            i = raw - t.first_frame
            box = project_box(calib, p, t.h, t.w, t.l, float(t.rotations[i][2]))
            if box is None:
                continue
            s = box_score(score, valid, box)
            if s is None:
                continue
            rng_m = float(np.linalg.norm(p[:2]))
            is_moving = t.is_moving_at(raw)
            (moving if is_moving else parked).append(s)
            rows.append((rng_m, s, is_moving, t.object_type))

            # size-matched random control box somewhere else in the frame
            bw, bh = box[2], box[3]
            for _ in range(3):
                rx = int(RNG.integers(0, max(1, w - bw)))
                ry = int(RNG.integers(0, max(1, h - bh)))
                rs = box_score(score, valid, (rx, ry, bw, bh))
                if rs is not None:
                    random_ctl.append(rs)
                    break

    def stat(name, v):
        v = np.asarray(v)
        if not len(v):
            print(f"  {name:<26} (none)")
            return
        print(f"  {name:<26} n={len(v):<5} mean={v.mean():7.3f}  "
              f"median={np.median(v):7.3f}  p90={np.percentile(v, 90):8.3f}")

    print("=" * 74)
    print("raw surprise score inside annotated boxes -- no threshold, no blobs")
    print("=" * 74)
    stat("MOVING vehicles", moving)
    stat("PARKED vehicles", parked)
    stat("random size-matched boxes", random_ctl)

    print()
    print(f"  moving / parked mean ratio     {np.mean(moving) / max(np.mean(parked), 1e-9):.2f}x")
    print(f"  AUC  moving vs parked          {auc(moving, parked):.3f}   (0.5 = pure noise)")
    print(f"  AUC  moving vs random boxes    {auc(moving, random_ctl):.3f}")
    print(f"  AUC  parked vs random boxes    {auc(parked, random_ctl):.3f}   "
          f"(should be near 0.5 if cars are not special)")

    print("\nmatched by range -- same distance, same object class, only motion differs")
    print(f"  {'range':<12} {'n_mov':>6} {'mean_mov':>10} {'n_park':>7} {'mean_park':>10} {'ratio':>7}")
    for lo, hi in [(0, 15), (15, 25), (25, 40), (40, 200)]:
        mv = [s for r, s, m, _ in rows if m and lo <= r < hi]
        pk = [s for r, s, m, _ in rows if not m and lo <= r < hi]
        if not mv or not pk:
            continue
        ratio = np.mean(mv) / max(np.mean(pk), 1e-9)
        print(f"  {lo:>3}-{hi:<8}m {len(mv):>6} {np.mean(mv):>10.3f} "
              f"{len(pk):>7} {np.mean(pk):>10.3f} {ratio:>6.2f}x")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
