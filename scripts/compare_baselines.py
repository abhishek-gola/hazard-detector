#!/usr/bin/env python3
"""Depth-residual forecasting vs classical flow-residual. The decisive experiment.

`identity` was a straw man and any reviewer would say so in one sentence: it
predicts that nothing changes, so of course it loses, and beating it only proves
the warp does something. The question that actually matters is whether a 1.3 B
parameter transformer and a 5 GB checkpoint buy anything over Farneback plus a
RANSAC fundamental matrix, which has been available since the 1990s and runs on
a CPU.

Both methods here share the same robust normalisation, the same blob extractor,
the same tracker and the same threshold grid. The only thing that differs is the
residual space: "how wrong was the forecast 3D geometry" versus "how much of this
flow cannot be explained by a rigid scene". Compared as PR curves and AP, not at
one hand-picked operating point each.

    python scripts/compare_baselines.py --cache cache/junction \\
        --drive data/2011_09_26/2011_09_26_drive_0009_sync --calib-dir data/2011_09_26
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hazard.backbone import WindowGeometry  # noqa: E402
from hazard.baseline_flow import flow_surprise  # noqa: E402
from hazard.detect import BlobTracker, extract_blobs  # noqa: E402
from hazard.forecast import ForecastContext, RigidFlowForecaster  # noqa: E402
from hazard.kitti_calib import KittiCalib, project_box  # noqa: E402
from hazard.kitti_labels import DYNAMIC_TYPES, mark_moving, parse_tracklets  # noqa: E402
from hazard.surprise import compute_surprise  # noqa: E402

SIZE_BUCKETS = [(0, 600, "<600"), (600, 2000, "600-2k"), (2000, 10 ** 9, ">2k")]


def _load_chunks(cache: Path):
    man = json.loads((cache / "manifest.json").read_text())
    for ch in man["chunks"]:
        d = np.load(cache / ch["file"])
        geom = WindowGeometry(
            depth=d["depth"].astype(np.float32), conf=d["conf"].astype(np.float32),
            extrinsic=d["extrinsic"], intrinsic=d["intrinsic"], seconds=0.0,
        )
        yield ch, geom, d["images"]
    return


def maps_for(cache: Path, method: str, threshold: float):
    """Surprise map per frame stem, for one method. Returns (maps, seconds)."""
    man = json.loads((cache / "manifest.json").read_text())
    stems, ctx = man["frames"], man["ctx"]
    fc = RigidFlowForecaster()
    out, t0 = {}, time.time()
    for ch, geom, images_u8 in _load_chunks(cache):
        imgs = images_u8.astype(np.float32) / 255.0
        s, e = ch["start"], ch["end"]
        for local in range(ctx, e - s):
            g = s + local
            if stems[g] in out:
                continue
            if method == "depth":
                f = fc.forecast(ForecastContext(geom=geom, frames=imgs, target=local))
                sm = compute_surprise(f, geom, local, threshold=threshold)
            else:
                sm = flow_surprise(images_u8[local - 1], images_u8[local],
                                   threshold=threshold)
            out[stems[g]] = sm
    return out, time.time() - t0


def detect_and_score(maps, tracklets, calib, threshold, min_age=2, hit_frac=0.3):
    tracker = BlobTracker(centre_frac=1.6)
    per_frame = []
    for stem in sorted(maps):
        blobs = extract_blobs(maps[stem], int(stem), threshold=threshold, min_area=60)
        per_frame.append((stem, tracker.update(blobs, int(stem))))

    buckets = {n: [0, 0] for _, _, n in SIZE_BUCKETS}
    n_mov = n_hit = n_park = n_pfp = n_det = d_mov = 0
    for stem, dets in per_frame:
        raw = int(stem)
        dets = [d for d in dets if d.age >= min_age]
        objs = []
        for t in tracklets:
            p = t.at(raw)
            if p is None or t.object_type not in DYNAMIC_TYPES:
                continue
            i = raw - t.first_frame
            b = project_box(calib, p, t.h, t.w, t.l, float(t.rotations[i][2]))
            if b:
                objs.append((b, t.is_moving_at(raw)))
        for b, mv in objs:
            hit = any(_ov(d.box, b) >= hit_frac for d in dets)
            if mv:
                n_mov += 1
                n_hit += hit
                a = b[2] * b[3]
                for lo, hi, nm in SIZE_BUCKETS:
                    if lo <= a < hi:
                        buckets[nm][0] += 1
                        buckets[nm][1] += hit
            else:
                n_park += 1
                n_pfp += hit
        n_det += len(dets)
        d_mov += sum(1 for d in dets
                     if any(_ov(d.box, b) >= hit_frac for b, mv in objs if mv))
    rec, prec = n_hit / max(n_mov, 1), d_mov / max(n_det, 1)
    return {
        "recall": rec, "precision": prec, "parked_fp": n_pfp / max(n_park, 1),
        "f1": 2 * rec * prec / (rec + prec) if rec + prec else 0.0,
        "n_det": n_det, "n_mov": n_mov, "n_frames": len(per_frame),
        "buckets": {k: (v[1], v[0]) for k, v in buckets.items()},
    }


def _ov(det, obj) -> float:
    dx, dy, dw, dh = det
    ox, oy, ow, oh = obj
    x0, y0 = max(dx, ox), max(dy, oy)
    x1, y1 = min(dx + dw, ox + ow), min(dy + dh, oy + oh)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return (x1 - x0) * (y1 - y0) / float(max(dw * dh, 1))


def ap(curve) -> float:
    pts = sorted(((c["recall"], c["precision"]) for c in curve))
    rp, a = 0.0, 0.0
    for r, p in pts:
        a += (r - rp) * p
        rp = r
    return a


def main() -> int:
    ap_ = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap_.add_argument("--cache", required=True)
    ap_.add_argument("--drive", required=True)
    ap_.add_argument("--calib-dir", required=True)
    ap_.add_argument("--grid", default="4,6,8,10,13,16,20,25,32")
    args = ap_.parse_args()

    tracklets = mark_moving(parse_tracklets(Path(args.drive) / "tracklet_labels.xml"))
    calib = KittiCalib(args.calib_dir)
    grid = [float(x) for x in args.grid.split(",")]
    cache = Path(args.cache)

    results = {}
    for method in ("depth", "flow"):
        # The map only depends on threshold through peak/area bookkeeping, so
        # build it once and re-detect at each threshold.
        maps, secs = maps_for(cache, method, grid[0])
        curve = [detect_and_score(maps, tracklets, calib, t) for t in grid]
        results[method] = (curve, secs)

        print(f"\n{method.upper()} residual   "
              f"({secs / max(len(maps), 1):.3f} s/frame after VGGT)")
        print(f"  {'thresh':>7} {'recall':>8} {'prec':>7} {'parkFP':>7} {'F1':>6} {'d/f':>6}"
              f"   " + " ".join(f"{n:>8}" for _, _, n in SIZE_BUCKETS))
        for t, c in zip(grid, curve):
            b = " ".join(f"{c['buckets'][n][0]:>3}/{c['buckets'][n][1]:<4}"
                         for _, _, n in SIZE_BUCKETS)
            print(f"  {t:>7.0f} {100 * c['recall']:>7.1f}% {100 * c['precision']:>6.1f}% "
                  f"{100 * c['parked_fp']:>6.1f}% {c['f1']:>6.3f} "
                  f"{c['n_det'] / max(c['n_frames'], 1):>6.2f}   {b}")
        print(f"  -> AP {ap(curve):.3f}   best F1 {max(c['f1'] for c in curve):.3f}")

    print(f"\n{'=' * 74}")
    print("HEAD TO HEAD")
    print("=" * 74)
    print(f"  {'method':<12} {'AP':>7} {'bestF1':>8} {'maxRecall':>11} "
          f"{'R@FP<=5%':>10} {'prec@R>=30%':>12}")
    for m, (curve, _) in results.items():
        lo_fp = [c for c in curve if c["parked_fp"] <= 0.05]
        hi_r = [c for c in curve if c["recall"] >= 0.30]
        print(f"  {m:<12} {ap(curve):>7.3f} {max(c['f1'] for c in curve):>8.3f} "
              f"{100 * max(c['recall'] for c in curve):>10.1f}% "
              f"{100 * max((c['recall'] for c in lo_fp), default=0):>9.1f}% "
              f"{100 * max((c['precision'] for c in hi_r), default=0):>11.1f}%")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
