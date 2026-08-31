#!/usr/bin/env python3
"""Score the detector against KITTI tracklets it never saw.

The honest question is not "did it find objects" -- an object detector does that
better. It is: **did the things it flagged turn out to be the things that were
moving?** So we score only against tracklets whose motion cannot be explained by
ego motion (see `hazard/kitti_labels.py`), and we report what happened on the
static ones separately, because flagging a parked car would be a failure, not a
success.

    python scripts/validate.py --run outputs/kitti_busy \\
        --drive data/2011_09_26/2011_09_26_drive_0009_sync \\
        --calib data/2011_09_26
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hazard.kitti_calib import KittiCalib, project_box  # noqa: E402
from hazard.kitti_labels import DYNAMIC_TYPES, mark_moving, parse_tracklets  # noqa: E402


def overlap_fraction(blob: tuple[int, int, int, int],
                     obj: tuple[int, int, int, int]) -> float:
    """Intersection as a fraction of the *blob*.

    Deliberately not IoU. A surprise blob usually covers the part of a car whose
    depth was mispredicted -- a bumper, a flank -- not the whole silhouette, so
    IoU would punish correct detections for being compact. What we want to know
    is whether the blob is sitting on the object.
    """
    bx, by, bw, bh = blob
    ox, oy, ow, oh = obj
    x0, y0 = max(bx, ox), max(by, oy)
    x1, y1 = min(bx + bw, ox + ow), min(by + bh, oy + oh)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return (x1 - x0) * (y1 - y0) / float(max(bw * bh, 1))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="an output dir containing blobs.json")
    ap.add_argument("--drive", required=True, help="KITTI drive dir with tracklet_labels.xml")
    ap.add_argument("--calib", required=True, help="dir with calib_cam_to_cam.txt")
    ap.add_argument("--hit-frac", type=float, default=0.3,
                    help="blob counts as on-object above this overlap fraction")
    ap.add_argument("--min-age", type=int, default=2,
                    help="only score blobs that persisted this many frames")
    args = ap.parse_args()

    run = Path(args.run)
    blobs_path = run / "blobs.json"
    if not blobs_path.exists():
        print(f"error: {blobs_path} not found -- re-run the detector to emit it",
              file=sys.stderr)
        return 1
    per_frame = json.loads(blobs_path.read_text())

    tracklets = mark_moving(parse_tracklets(Path(args.drive) / "tracklet_labels.xml"))
    calib = KittiCalib(args.calib)

    n_moving_seen = n_moving_hit = 0
    n_static_seen = n_static_hit = 0
    n_blobs = n_blobs_on_moving = n_blobs_on_static = n_blobs_on_nothing = 0
    per_type: dict[str, list[int]] = {}
    frame_rows = []

    for stem, blobs in sorted(per_frame.items()):
        try:
            raw = int(stem)
        except ValueError:
            continue
        blobs = [b for b in blobs if b["age"] >= args.min_age]
        boxes = [tuple(b["box"]) for b in blobs]

        moving_boxes, static_boxes = [], []
        for t in tracklets:
            p = t.at(raw)
            if p is None or t.object_type not in DYNAMIC_TYPES:
                continue
            i = raw - t.first_frame
            ry = float(t.rotations[i][2])
            box = project_box(calib, p, t.h, t.w, t.l, ry)
            if box is None:
                continue
            (moving_boxes if t.is_moving_at(raw) else static_boxes).append(
                (box, t.object_type)
            )

        hit_flags = []
        for box, otype in moving_boxes:
            n_moving_seen += 1
            hit = any(overlap_fraction(b, box) >= args.hit_frac for b in boxes)
            n_moving_hit += hit
            hit_flags.append(hit)
            per_type.setdefault(otype, [0, 0])
            per_type[otype][0] += 1
            per_type[otype][1] += hit

        for box, _ in static_boxes:
            n_static_seen += 1
            n_static_hit += any(overlap_fraction(b, box) >= args.hit_frac for b in boxes)

        for b in boxes:
            n_blobs += 1
            on_m = any(overlap_fraction(b, box) >= args.hit_frac for box, _ in moving_boxes)
            on_s = any(overlap_fraction(b, box) >= args.hit_frac for box, _ in static_boxes)
            n_blobs_on_moving += on_m
            n_blobs_on_static += on_s and not on_m
            n_blobs_on_nothing += not (on_m or on_s)

        if moving_boxes or blobs:
            frame_rows.append((raw, len(moving_boxes), sum(hit_flags), len(blobs)))

    print("=" * 68)
    print(f"validation: {run.name}")
    print("=" * 68)
    print(f"frames scored                    {len(per_frame)}")
    print(f"moving-object instances visible  {n_moving_seen}")
    print(f"  ... flagged by the detector    {n_moving_hit}"
          f"  ({100 * n_moving_hit / max(n_moving_seen, 1):.1f}% recall)")
    print()
    print(f"parked-object instances visible  {n_static_seen}")
    print(f"  ... wrongly flagged            {n_static_hit}"
          f"  ({100 * n_static_hit / max(n_static_seen, 1):.1f}%)  <- lower is better")
    print()
    print(f"blobs reported (age >= {args.min_age})         {n_blobs}")
    print(f"  on a moving object             {n_blobs_on_moving}"
          f"  ({100 * n_blobs_on_moving / max(n_blobs, 1):.1f}% precision)")
    print(f"  on a parked annotated object   {n_blobs_on_static}")
    print(f"  on neither                     {n_blobs_on_nothing}")

    if per_type:
        print("\nrecall by object type")
        for k, (seen, hit) in sorted(per_type.items(), key=lambda kv: -kv[1][0]):
            print(f"  {k:<14} {hit:>4}/{seen:<4}  {100 * hit / max(seen, 1):5.1f}%")

    busiest = sorted(frame_rows, key=lambda r: -r[1])[:10]
    if busiest:
        print("\nbusiest frames  (raw, moving visible, flagged, blobs)")
        for raw, nm, nh, nb in busiest:
            print(f"  {raw:>5}   {nm:>2} moving   {nh:>2} flagged   {nb:>2} blobs")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
