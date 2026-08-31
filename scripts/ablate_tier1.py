#!/usr/bin/env python3
"""Attribute each small-object fix separately, replaying cached VGGT geometry.

The premise being tested: recall on small objects is near zero not because the
signal is absent but because the validity masks and the blur destroy it before
the detector runs. A distant car covers ~420 px, keeps ~30% of that through the
masks, and is then asked for 60 connected pixels above threshold.

Four fixes, added cumulatively, each measured on both segments:

  base      what shipped
  +blur     normalised blur -- smooth score*valid and valid, then divide
  +noise    residual whitened by its expected size instead of a hard edge mask
  +area     min_area scaled quadratically with distance below the horizon

Reported as recall against apparent size, because a single recall number hides
the entire effect.

    python scripts/ablate_tier1.py --cache cache --drive data/..._0009_sync \\
        --calib-dir data/2011_09_26
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hazard.backbone import WindowGeometry  # noqa: E402
from hazard.detect import BlobTracker, extract_blobs  # noqa: E402
from hazard.forecast import ForecastContext, RigidFlowForecaster  # noqa: E402
from hazard.kitti_calib import KittiCalib, project_box  # noqa: E402
from hazard.kitti_labels import DYNAMIC_TYPES, mark_moving, parse_tracklets  # noqa: E402
from hazard.surprise import compute_surprise  # noqa: E402

SIZE_BUCKETS = [(0, 600, "< 600"), (600, 2000, "600-2000"), (2000, 10 ** 9, "> 2000")]


def replay(cache: Path, surprise_kw: dict, detect_kw: dict, threshold: float):
    """Recompute surprise + detections for a whole segment from cached geometry."""
    man = json.loads((cache / "manifest.json").read_text())
    stems = man["frames"]
    ctx = man["ctx"]
    fc = RigidFlowForecaster()
    tracker = BlobTracker(centre_frac=1.6)
    out = {}

    for ch in man["chunks"]:
        d = np.load(cache / ch["file"])
        geom = WindowGeometry(
            depth=d["depth"].astype(np.float32), conf=d["conf"].astype(np.float32),
            extrinsic=d["extrinsic"], intrinsic=d["intrinsic"], seconds=0.0,
        )
        images = d["images"].astype(np.float32) / 255.0
        s, e = ch["start"], ch["end"]
        for local in range(ctx, e - s):
            g = s + local
            if stems[g] in out:
                continue
            f = fc.forecast(ForecastContext(geom=geom, frames=images, target=local))
            sm = compute_surprise(f, geom, local, threshold=threshold, **surprise_kw)
            blobs = extract_blobs(sm, g, threshold=threshold, **detect_kw)
            out[stems[g]] = tracker.update(blobs, g)
    return out


def overlap_frac(det, obj) -> float:
    dx, dy, dw, dh = det
    ox, oy, ow, oh = obj
    x0, y0 = max(dx, ox), max(dy, oy)
    x1, y1 = min(dx + dw, ox + ow), min(dy + dh, oy + oh)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return (x1 - x0) * (y1 - y0) / float(max(dw * dh, 1))


def score(dets_by_stem, tracklets, calib, min_age=2, hit_frac=0.3):
    per_bucket = {name: [0, 0] for _, _, name in SIZE_BUCKETS}
    n_mov = n_hit = n_park = n_park_hit = n_det = d_mov = 0
    for stem, dets in dets_by_stem.items():
        raw = int(stem)
        dets = [d for d in dets if d.age >= min_age]
        objs = []
        for t in tracklets:
            p = t.at(raw)
            if p is None or t.object_type not in DYNAMIC_TYPES:
                continue
            i = raw - t.first_frame
            b = project_box(calib, p, t.h, t.w, t.l, float(t.rotations[i][2]))
            if b is None:
                continue
            objs.append((b, t.is_moving_at(raw)))
        for b, mv in objs:
            hit = any(overlap_frac(x.box, b) >= hit_frac for x in dets)
            if mv:
                n_mov += 1
                n_hit += hit
                area = b[2] * b[3]
                for lo, hi, name in SIZE_BUCKETS:
                    if lo <= area < hi:
                        per_bucket[name][0] += 1
                        per_bucket[name][1] += hit
            else:
                n_park += 1
                n_park_hit += hit
        n_det += len(dets)
        d_mov += sum(1 for x in dets
                     if any(overlap_frac(x.box, b) >= hit_frac for b, mv in objs if mv))
    rec = n_hit / max(n_mov, 1)
    prec = d_mov / max(n_det, 1)
    return {
        "recall": rec, "precision": prec,
        "parked_fp": n_park_hit / max(n_park, 1),
        "f1": 2 * rec * prec / (rec + prec) if rec + prec else 0.0,
        "n_det": n_det, "n_mov": n_mov,
        "buckets": {k: (v[1], v[0]) for k, v in per_bucket.items()},
    }


CONFIGS = [
    ("base  (as shipped)",
     dict(whiten=False, normalised_blur=False), dict(row_scaled_area=False, min_area=60)),
    ("+ normalised blur",
     dict(whiten=False, normalised_blur=True), dict(row_scaled_area=False, min_area=60)),
    ("+ noise model (drop edge mask)",
     dict(whiten=True, normalised_blur=True), dict(row_scaled_area=False, min_area=60)),
    ("+ row-scaled min_area",
     dict(whiten=True, normalised_blur=True), dict(row_scaled_area=True, min_area=60)),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--drive", required=True)
    ap.add_argument("--calib-dir", required=True)
    ap.add_argument("--threshold", type=float, default=8.0)
    args = ap.parse_args()

    tracklets = mark_moving(parse_tracklets(Path(args.drive) / "tracklet_labels.xml"))
    calib = KittiCalib(args.calib_dir)
    cache = Path(args.cache)

    for seg in ("calib", "junction"):
        seg_dir = cache / seg
        if not (seg_dir / "manifest.json").exists():
            print(f"skip {seg}: no cache", file=sys.stderr)
            continue
        print(f"\n{'=' * 86}")
        print(f"SEGMENT: {seg}   (threshold {args.threshold})")
        print("=" * 86)
        hdr = (f"  {'config':<32} {'recall':>7} {'prec':>6} {'parkFP':>7} {'F1':>6} "
               + " ".join(f"{n:>10}" for _, _, n in SIZE_BUCKETS))
        print(hdr)
        for name, skw, dkw in CONFIGS:
            dets = replay(seg_dir, skw, dkw, args.threshold)
            r = score(dets, tracklets, calib)
            buckets = " ".join(
                f"{r['buckets'][n][0]:>3}/{r['buckets'][n][1]:<3}({100 * r['buckets'][n][0] / max(r['buckets'][n][1], 1):>2.0f}%)"
                if r['buckets'][n][1] else f"{'-':>10}"
                for _, _, n in SIZE_BUCKETS)
            print(f"  {name:<32} {100 * r['recall']:>6.1f}% {100 * r['precision']:>5.1f}% "
                  f"{100 * r['parked_fp']:>6.1f}% {r['f1']:>6.3f} {buckets}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
