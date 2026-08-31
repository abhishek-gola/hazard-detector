#!/usr/bin/env python3
"""Look at the false positives instead of hypothesising about them.

The README has been claiming that detections landing on nothing annotated are
"some thin structures, some real motion KITTI does not label". That is a guess,
and it matters which: if a meaningful fraction is genuinely moving things outside
the annotation set, the reported precision is wrong in our own favour, and saying
so is worth more than another point of recall.

This crops every unmatched detection with context, sorts them into automatic
buckets, and builds a contact sheet to be read by eye. The automatic buckets are
geometric priors only -- they propose, they do not conclude:

  thin-vertical   aspect ratio < 0.6 and tall: poles, signposts, bollards
  near-road       bottom corners, where forward-warp parallax is most extreme
  sky-band        above the horizon, where depth is least reliable
  other           everything the priors do not explain

Then `--label` writes a CSV you can correct by hand while looking at the sheet.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hazard.backbone import WindowGeometry  # noqa: E402
from hazard.detect import BlobTracker, extract_blobs  # noqa: E402
from hazard.forecast import ForecastContext, RigidFlowForecaster  # noqa: E402
from hazard.kitti_calib import KittiCalib, project_box  # noqa: E402
from hazard.kitti_labels import mark_moving, parse_tracklets  # noqa: E402
from hazard.surprise import compute_surprise  # noqa: E402

THRESHOLD = 8.0
SURPRISE_KW = dict(whiten=True, normalised_blur=True)


def _ov(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x0, y0 = max(ax, bx), max(ay, by)
    x1, y1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return (x1 - x0) * (y1 - y0) / float(max(aw * ah, 1))


def classify(box, shape) -> str:
    """Geometric prior for what this false positive probably is."""
    x, y, w, h = box
    H, W = shape
    horizon = H / 2.0
    aspect = w / max(h, 1)
    if aspect < 0.6 and h >= 14:
        return "thin-vertical"
    if y + h > 0.82 * H and (x < 0.18 * W or x + w > 0.82 * W):
        return "near-road-corner"
    if y + h < horizon:
        return "sky-band"
    return "other"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--drive", required=True)
    ap.add_argument("--calib-dir", required=True)
    ap.add_argument("--out", default="outputs/fp_analysis")
    ap.add_argument("--pad", type=int, default=16)
    ap.add_argument("--cols", type=int, default=6)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tracklets = mark_moving(parse_tracklets(Path(args.drive) / "tracklet_labels.xml"))
    calib = KittiCalib(args.calib_dir)

    man = json.loads((Path(args.cache) / "manifest.json").read_text())
    stems, ctx = man["frames"], man["ctx"]
    fc = RigidFlowForecaster()
    tracker = BlobTracker(centre_frac=1.6)

    rows, crops, seen = [], [], set()
    n_det = n_matched = 0

    for ch in man["chunks"]:
        d = np.load(Path(args.cache) / ch["file"])
        geom = WindowGeometry(
            depth=d["depth"].astype(np.float32), conf=d["conf"].astype(np.float32),
            extrinsic=d["extrinsic"], intrinsic=d["intrinsic"], seconds=0.0)
        imgs_u8 = d["images"]
        imgs = imgs_u8.astype(np.float32) / 255.0
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

            # every annotated object of ANY class, moving or not
            objs = []
            for t in tracklets:
                p = t.at(raw)
                if p is None:
                    continue
                i = raw - t.first_frame
                b = project_box(calib, p, t.h, t.w, t.l, float(t.rotations[i][2]))
                if b:
                    objs.append((b, t.object_type, t.is_moving_at(raw)))

            H, W = sm.score.shape
            for det in dets:
                n_det += 1
                best = max((_ov(det.box, b) for b, _, _ in objs), default=0.0)
                if best >= 0.3:
                    n_matched += 1
                    continue
                cat = classify(det.box, (H, W))
                x, y, w, h = det.box
                x0, y0 = max(0, x - args.pad), max(0, y - args.pad)
                x1, y1 = min(W, x + w + args.pad), min(H, y + h + args.pad)
                crop = cv2.cvtColor(imgs_u8[local][y0:y1, x0:x1], cv2.COLOR_RGB2BGR)
                crop = cv2.resize(crop, (128, 128), interpolation=cv2.INTER_NEAREST)
                cv2.rectangle(crop, (2, 2), (125, 125), (0, 255, 255), 1)
                strip = np.full((16, 128, 3), 20, np.uint8)
                cv2.putText(strip, f"{raw} {cat[:11]}", (2, 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32, (230, 230, 230), 1)
                crops.append(np.vstack([strip, crop]))
                rows.append(dict(frame=raw, box=str(det.box), auto_category=cat,
                                 peak=round(det.peak, 1), area_px=det.area_px,
                                 best_overlap=round(best, 3), manual_category=""))

    from collections import Counter
    counts = Counter(r["auto_category"] for r in rows)
    print(f"detections (age>=2): {n_det}")
    print(f"  matched an annotated object (any class): {n_matched}")
    print(f"  unmatched -> false positives:            {len(rows)}"
          f"   ({100 * len(rows) / max(n_det, 1):.0f}%)")
    print("\nautomatic geometric buckets (priors, not conclusions):")
    for k, v in counts.most_common():
        print(f"  {k:<18} {v:>4}  ({100 * v / max(len(rows), 1):>4.0f}%)")

    if crops:
        cols = args.cols
        rows_n = (len(crops) + cols - 1) // cols
        ch_, cw = crops[0].shape[:2]
        sheet = np.full((rows_n * ch_, cols * cw, 3), 15, np.uint8)
        for i, c in enumerate(crops):
            r, cc = divmod(i, cols)
            sheet[r * ch_:(r + 1) * ch_, cc * cw:(cc + 1) * cw] = c
        cv2.imwrite(str(out / "false_positives.png"), sheet)
        print(f"\ncontact sheet: {out / 'false_positives.png'}  ({len(crops)} crops)")

    with open(out / "false_positives.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else ["frame"])
        if rows:
            w.writeheader()
            w.writerows(rows)
    print(f"csv (manual_category left blank to fill in): {out / 'false_positives.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
