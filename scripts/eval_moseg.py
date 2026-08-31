#!/usr/bin/env python3
"""Evaluate against KittiMoSeg, and check the self-derived labels against it.

Three questions, in order of how much they matter.

**1. Was the derived ground truth any good?** `hazard/kitti_labels.py` decides
which tracklets move by fitting a 2D rigid velocity field to the tracklets
themselves. KittiMoSeg decides the same thing from GPS/IMU odometry. Matching
their object sets by IoU and comparing the motion flags says directly whether
that substitution was sound -- and it is the first thing a reviewer should ask,
because every number produced before this rests on it.

**2. What are the detector's numbers against real labels?** Box-level recall and
precision, same detector, same threshold, only the labels swapped.

**3. mIoU.** The metric the moving-object-segmentation literature actually
reports, over the pixel motion mask. Without it these results compare to nothing
published.

    python scripts/eval_moseg.py --drives 0009,0013,0014 --calib-dir data/2011_09_26
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hazard import kittimoseg as ms  # noqa: E402
from hazard.backbone import WindowGeometry  # noqa: E402
from hazard.detect import BlobTracker, extract_blobs  # noqa: E402
from hazard.forecast import ForecastContext, RigidFlowForecaster  # noqa: E402
from hazard.kitti_calib import KittiCalib, project_box  # noqa: E402
from hazard.kitti_labels import DYNAMIC_TYPES, mark_moving, parse_tracklets  # noqa: E402
from hazard.surprise import compute_surprise  # noqa: E402

THRESHOLD = 8.0
SURPRISE_KW = dict(whiten=True, normalised_blur=True)


def iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x0, y0 = max(ax, bx), max(ay, by)
    x1, y1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    i = (x1 - x0) * (y1 - y0)
    return i / float(aw * ah + bw * bh - i)


def ov_of_det(det, obj) -> float:
    """Intersection as a fraction of the detection (as used throughout)."""
    dx, dy, dw, dh = det
    ox, oy, ow, oh = obj
    x0, y0 = max(dx, ox), max(dy, oy)
    x1, y1 = min(dx + dw, ox + ow), min(dy + dh, oy + oh)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return (x1 - x0) * (y1 - y0) / float(max(dw * dh, 1))


def bootstrap(vals, groups, n_boot=3000, seed=0):
    """Bootstrap a mean over group ids (tracklets / frames)."""
    if not vals:
        return 0.0, 0.0, 0.0
    by: dict = {}
    for v, g in zip(vals, groups):
        by.setdefault(g, []).append(v)
    keys = list(by)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        pick = rng.choice(len(keys), len(keys), replace=True)
        flat = [x for i in pick for x in by[keys[i]]]
        out.append(np.mean(flat) if flat else 0.0)
    return float(np.mean(vals)), float(np.percentile(out, 2.5)), \
        float(np.percentile(out, 97.5))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--drives", default="0009,0013,0014")
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--moseg", default="data/moseg")
    ap.add_argument("--data", default="data/2011_09_26")
    ap.add_argument("--calib-dir", default="data/2011_09_26")
    ap.add_argument("--cache-name", default="",
                    help="override cache subdir for a single drive (e.g. junction)")
    args = ap.parse_args()

    calib = KittiCalib(args.calib_dir)
    fc = RigidFlowForecaster()

    # ---- accumulators ---------------------------------------------------
    gt_tp = gt_fp = gt_fn = 0            # derived GT vs KittiMoSeg moving flags
    flag_agree = flag_total = 0
    det_rec, det_rec_g = [], []          # per-instance hit, grouped by track
    n_det = d_mov = n_static = n_sfp = 0
    ious, iou_g = [], []
    per_drive = {}

    for dv in args.drives.split(","):
        cdir = Path(args.cache) / (args.cache_name or f"d{dv}")
        if not (cdir / "manifest.json").exists() and dv == "0009":
            cdir = Path(args.cache) / "junction"
        ldir = Path(args.moseg) / dv
        if not (cdir / "manifest.json").exists() or not (ldir / "Text_data_Output").is_dir():
            print(f"  drive {dv}: missing cache or labels, skipped")
            continue

        tracklets = mark_moving(parse_tracklets(
            Path(args.data) / f"2011_09_26_drive_{dv}_sync" / "tracklet_labels.xml"))
        man = json.loads((cdir / "manifest.json").read_text())
        stems, ctx = man["frames"], man["ctx"]
        tracker = BlobTracker(centre_frac=1.6)
        seen = set()
        d_rec, d_det, d_dmov, d_iou = [], 0, 0, []

        for ch in man["chunks"]:
            d = np.load(cdir / ch["file"])
            geom = WindowGeometry(
                depth=d["depth"].astype(np.float32), conf=d["conf"].astype(np.float32),
                extrinsic=d["extrinsic"], intrinsic=d["intrinsic"], seconds=0.0)
            imgs = d["images"].astype(np.float32) / 255.0
            s, e = ch["start"], ch["end"]
            for local in range(ctx, e - s):
                g = s + local
                if stems[g] in seen:
                    continue
                seen.add(stems[g])
                raw = int(stems[g])

                f = fc.forecast(ForecastContext(geom=geom, frames=imgs, target=local))
                sm = compute_surprise(f, geom, local, threshold=THRESHOLD, **SURPRISE_KW)
                dets = [x for x in tracker.update(
                    extract_blobs(sm, raw, threshold=THRESHOLD, min_area=60), raw)
                    if x.age >= 2]

                # ---- KittiMoSeg labels for this frame
                mo = ms.moving_boxes_cropped(ldir, raw)
                mo_moving = [(b, c, tid) for b, mv, c, tid in mo if mv]
                mo_static = [(b, c) for b, mv, c, _ in mo if not mv]

                # ---- (1) derived GT vs KittiMoSeg
                derived = []
                for t in tracklets:
                    p = t.at(raw)
                    if p is None or t.object_type not in DYNAMIC_TYPES:
                        continue
                    i = raw - t.first_frame
                    b = project_box(calib, p, t.h, t.w, t.l, float(t.rotations[i][2]))
                    if b:
                        derived.append((b, t.is_moving_at(raw)))
                used = set()
                for b, mv in derived:
                    best, bj = 0.0, -1
                    for j, (mb, _mv, _c, _tid) in enumerate(mo):
                        if j in used:
                            continue
                        v = iou(b, mb)
                        if v > best:
                            best, bj = v, j
                    if best >= 0.3 and bj >= 0:
                        used.add(bj)
                        flag_total += 1
                        flag_agree += int(mv == mo[bj][1])
                        if mv and mo[bj][1]:
                            gt_tp += 1
                        elif mv and not mo[bj][1]:
                            gt_fp += 1
                        elif (not mv) and mo[bj][1]:
                            gt_fn += 1
                    elif mv:
                        gt_fp += 1
                for j, (mb, mv, _, _tid) in enumerate(mo):
                    if mv and j not in used:
                        gt_fn += 1

                # ---- (2) detector vs KittiMoSeg boxes
                for b, cname, tid in mo_moving:
                    hit = any(ov_of_det(x.box, b) >= 0.3 for x in dets)
                    det_rec.append(int(hit))
                    det_rec_g.append(f"{dv}:{tid}")   # real KittiMoSeg track id
                    d_rec.append(int(hit))
                for b, _ in mo_static:
                    n_static += 1
                    n_sfp += any(ov_of_det(x.box, b) >= 0.3 for x in dets)
                n_det += len(dets)
                d_det += len(dets)
                k = sum(1 for x in dets
                        if any(ov_of_det(x.box, b) >= 0.3 for b, _, _ in mo_moving))
                d_mov += k
                d_dmov += k

                # ---- (3) mIoU on the pixel motion mask
                mn = ms.load_mask_native(ldir, raw)
                if mn is not None:
                    gt_mask = ms.mask_to_cropped(mn)
                    pred = (sm.score > THRESHOLD) & sm.valid
                    if gt_mask.any() or pred.any():
                        inter = (pred & gt_mask).sum()
                        union = (pred | gt_mask).sum()
                        v = float(inter / max(union, 1))
                        ious.append(v)
                        iou_g.append(f"{dv}:{raw // 20}")
                        d_iou.append(v)

        per_drive[dv] = dict(frames=len(seen), inst=len(d_rec),
                             recall=np.mean(d_rec) if d_rec else 0.0,
                             prec=d_dmov / max(d_det, 1),
                             miou=np.mean(d_iou) if d_iou else 0.0)

    # ---- report ---------------------------------------------------------
    print("=" * 80)
    print("1. IS THE SELF-DERIVED GROUND TRUTH ANY GOOD?")
    print("   derived rigid-field motion flags vs KittiMoSeg's GPS/IMU-derived ones")
    print("=" * 80)
    prec = gt_tp / max(gt_tp + gt_fp, 1)
    rec = gt_tp / max(gt_tp + gt_fn, 1)
    print(f"  matched objects with both flags        {flag_total}")
    print(f"  motion-flag agreement on those         "
          f"{100 * flag_agree / max(flag_total, 1):.1f}%")
    print(f"  derived 'moving' that MoSeg calls moving (precision)  "
          f"{100 * prec:.1f}%   ({gt_tp}/{gt_tp + gt_fp})")
    print(f"  MoSeg 'moving' the derived GT also finds (recall)     "
          f"{100 * rec:.1f}%   ({gt_tp}/{gt_tp + gt_fn})")

    print("\n" + "=" * 80)
    print("2. DETECTOR vs KittiMoSeg BOXES")
    print("=" * 80)
    pt, lo, hi = bootstrap(det_rec, det_rec_g)
    print(f"  moving instances                {len(det_rec)}")
    print(f"  distinct tracks (CI grouped on these)  {len(set(det_rec_g))}")
    print(f"  recall                          {100 * pt:.1f}% [{100 * lo:.1f}, {100 * hi:.1f}]")
    print(f"  precision                       {100 * d_mov / max(n_det, 1):.1f}%")
    print(f"  static-object false-flag rate   {100 * n_sfp / max(n_static, 1):.1f}%"
          f"   ({n_sfp}/{n_static})")

    print("\n" + "=" * 80)
    print("3. mIoU ON THE PIXEL MOTION MASK  (the metric the literature reports)")
    print("=" * 80)
    pt, lo, hi = bootstrap(ious, iou_g)
    print(f"  frames scored                   {len(ious)}")
    print(f"  IoU of the moving class         {100 * pt:.2f}% "
          f"[{100 * lo:.2f}, {100 * hi:.2f}]")
    print(f"  (MODNet-era published mIoU on this task is far higher; see README)")

    print("\nper drive")
    print(f"  {'drive':<7} {'frames':>7} {'inst':>6} {'recall':>8} {'prec':>7} {'IoU':>7}")
    for dv, r in per_drive.items():
        print(f"  {dv:<7} {r['frames']:>7} {r['inst']:>6} {100 * r['recall']:>7.1f}% "
              f"{100 * r['prec']:>6.1f}% {100 * r['miou']:>6.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
