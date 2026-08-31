#!/usr/bin/env python3
"""Compare detectors across their whole operating range, on saved surprise maps.

Two reasons this script exists rather than a single validate.py run.

First, comparing two detectors at one arbitrary threshold each is meaningless --
you can make either one look better by picking its point. What matters is the
precision/recall curve, and specifically precision *at matched false-alarm
rate*.

Second, honesty about tuning. The ground truth here is nine moving tracklets.
Choosing an operating point by maximising a score against nine objects is
overfitting, not engineering. So `--calibrate` picks the threshold on one
segment of the drive and `--eval` reports on a disjoint one, and the threshold
is chosen by a criterion that never looks at recall: the false-alarm rate on
parked cars.

Runs on the .npz maps, so it never touches VGGT and a full sweep takes seconds.

    python scripts/sweep.py --calib outputs/calib --eval outputs/kitti_busy \\
        --drive data/2011_09_26/2011_09_26_drive_0009_sync --calib-dir data/2011_09_26
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hazard.detect import BlobTracker, detect_windows, extract_blobs  # noqa: E402
from hazard.kitti_calib import KittiCalib, project_box  # noqa: E402
from hazard.kitti_labels import DYNAMIC_TYPES, mark_moving, parse_tracklets  # noqa: E402
from hazard.surprise import SurpriseMap  # noqa: E402


def load_maps(run: Path) -> list[tuple[int, SurpriseMap]]:
    out = []
    for f in sorted((run / "maps").glob("*.npz")):
        d = np.load(f)
        score = d["score"].astype(np.float32)
        valid = d["valid"]
        out.append((int(f.stem), SurpriseMap(
            score=score, valid=valid, residual=score,
            frame_median=0.0, frame_mad=0.0, peak=0.0, area=0.0)))
    return out


def gt_boxes(tracklets, calib: KittiCalib, raw: int):
    """(box, is_moving) for every dynamic-class tracklet visible in this frame."""
    out = []
    for t in tracklets:
        p = t.at(raw)
        if p is None or t.object_type not in DYNAMIC_TYPES:
            continue
        i = raw - t.first_frame
        box = project_box(calib, p, t.h, t.w, t.l, float(t.rotations[i][2]))
        if box is None:
            continue
        out.append((box, t.is_moving_at(raw)))
    return out


def overlap_frac(det, obj) -> float:
    dx, dy, dw, dh = det
    ox, oy, ow, oh = obj
    x0, y0 = max(dx, ox), max(dy, oy)
    x1, y1 = min(dx + dw, ox + ow), min(dy + dh, oy + oh)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return (x1 - x0) * (y1 - y0) / float(max(dw * dh, 1))


def score_run(maps, tracklets, calib, detector: str, threshold: float,
              min_age: int, hit_frac: float) -> dict:
    tracker = BlobTracker()
    per_frame = []
    for raw, sm in maps:
        if detector == "window":
            dets = detect_windows(sm, raw, threshold=threshold)
        else:
            dets = extract_blobs(sm, raw, threshold=threshold, min_area=60)
        dets = tracker.update(dets, raw)
        per_frame.append((raw, [d for d in dets if d.age >= min_age]))

    n_mov = n_mov_hit = n_park = n_park_hit = 0
    n_det = d_mov = d_park = d_none = 0
    for raw, dets in per_frame:
        objs = gt_boxes(tracklets, calib, raw)
        for box, moving in objs:
            hit = any(overlap_frac(d.box, box) >= hit_frac for d in dets)
            if moving:
                n_mov += 1
                n_mov_hit += hit
            else:
                n_park += 1
                n_park_hit += hit
        n_det += len(dets)
        for d in dets:
            on_mov = any(overlap_frac(d.box, b) >= hit_frac for b, m in objs if m)
            on_park = any(overlap_frac(d.box, b) >= hit_frac for b, m in objs if not m)
            if on_mov:
                d_mov += 1
            elif on_park:
                d_park += 1
            else:
                d_none += 1

    return {
        "threshold": threshold,
        "recall": n_mov_hit / max(n_mov, 1),
        "parked_fp_rate": n_park_hit / max(n_park, 1),
        "precision": d_mov / max(n_det, 1),
        "n_det": n_det,
        "dets_per_frame": n_det / max(len(maps), 1),
        "on_moving": d_mov, "on_parked": d_park, "on_nothing": d_none,
        "n_moving": n_mov, "n_parked": n_park,
    }


def table(rows: list[dict], title: str) -> None:
    print(f"\n{title}")
    print(f"  {'thresh':>7} {'recall':>8} {'precision':>10} {'parkedFP':>9} "
          f"{'dets/frame':>11} {'F1':>6}")
    for r in rows:
        f1 = (2 * r["recall"] * r["precision"] / (r["recall"] + r["precision"])
              if r["recall"] + r["precision"] else 0.0)
        print(f"  {r['threshold']:>7.1f} {100 * r['recall']:>7.1f}% "
              f"{100 * r['precision']:>9.1f}% {100 * r['parked_fp_rate']:>8.1f}% "
              f"{r['dets_per_frame']:>11.2f} {f1:>6.3f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calib", required=True, help="run dir for the CALIBRATION segment")
    ap.add_argument("--eval", required=True, help="run dir for the EVAL segment")
    ap.add_argument("--drive", required=True)
    ap.add_argument("--calib-dir", required=True, help="dir with calib_cam_to_cam.txt")
    ap.add_argument("--min-age", type=int, default=2)
    ap.add_argument("--hit-frac", type=float, default=0.3)
    ap.add_argument("--target-fp", type=float, default=0.05,
                    help="parked-car false-alarm rate to calibrate the threshold at")
    args = ap.parse_args()

    tracklets = mark_moving(parse_tracklets(Path(args.drive) / "tracklet_labels.xml"))
    calib = KittiCalib(args.calib_dir)
    m_cal = load_maps(Path(args.calib))
    m_evl = load_maps(Path(args.eval))
    print(f"calibration segment: {len(m_cal)} frames   eval segment: {len(m_evl)} frames")

    win_grid = [3, 4, 5, 6, 8, 10, 12, 15, 20, 25]
    blob_grid = [2, 3, 4, 5, 6, 8, 10, 12]

    cal_win = [score_run(m_cal, tracklets, calib, "window", t, args.min_age, args.hit_frac)
               for t in win_grid]
    table(cal_win, "CALIBRATION segment (frames 0-196) -- window detector")

    # Choose the smallest threshold whose parked false-alarm rate is at target.
    # Recall is never consulted here.
    ok = [r for r in cal_win if r["parked_fp_rate"] <= args.target_fp]
    chosen = min(ok, key=lambda r: r["threshold"])["threshold"] if ok else win_grid[-1]
    print(f"\n  -> chosen threshold {chosen:.1f} "
          f"(smallest with parked FP <= {100 * args.target_fp:.0f}% on the "
          f"calibration segment; recall not consulted)")

    evl_win = [score_run(m_evl, tracklets, calib, "window", t, args.min_age, args.hit_frac)
               for t in win_grid]
    evl_blob = [score_run(m_evl, tracklets, calib, "blob", t, args.min_age, args.hit_frac)
                for t in blob_grid]
    table(evl_blob, "EVAL segment (frames 200-408) -- OLD blob chain")
    table(evl_win, "EVAL segment (frames 200-408) -- NEW window scoring")

    print("\n" + "=" * 72)
    print("HEADLINE -- eval segment, compared at matched parked false-alarm rate")
    print("=" * 72)
    old_best = max(evl_blob, key=lambda r: r["precision"] if r["parked_fp_rate"] <= 0.08 else -1)
    new_at = min([r for r in evl_win if r["threshold"] >= chosen],
                 key=lambda r: r["threshold"])
    new_matched = max([r for r in evl_win if r["parked_fp_rate"] <= old_best["parked_fp_rate"]],
                      key=lambda r: r["recall"], default=new_at)
    for name, r in (("old blob chain", old_best),
                    ("new, matched FP rate", new_matched),
                    (f"new, calibrated thresh={chosen:.0f}", new_at)):
        print(f"  {name:<30} recall {100 * r['recall']:5.1f}%   "
              f"precision {100 * r['precision']:5.1f}%   "
              f"parkedFP {100 * r['parked_fp_rate']:4.1f}%   "
              f"dets/frame {r['dets_per_frame']:.2f}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
