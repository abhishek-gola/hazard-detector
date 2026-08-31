#!/usr/bin/env python3
"""How much does the non-causal protocol flatter the results? Measure it.

The shipped pipeline pushes an 8-frame chunk through VGGT in one pass, and
VGGT's global attention means the depth it assigns to frame t-1 was computed with
frame t inside the window. The forecast itself only reads past frames, and this
mirrors VGGT-World's own evaluation protocol -- but "VGGT saw the target frame"
is a real asterisk and an asterisk is not a number.

So here is the number. A strictly causal variant, per target frame t:

  pass A   VGGT on [t-2, t-1]  ->  the forecast's inputs. Nothing after t-1.
  pass B   VGGT on [t-1, t]    ->  the observation at t.

The two passes have independent arbitrary depth scales, so they are aligned on
their shared frame t-1 by the ratio of median depths. That alignment is itself a
source of error and is part of what is being measured: a causal deployment would
have to do something like it.

Twelve times the VGGT compute of the chunked path, which is why it runs on a
subset rather than everywhere.

    python scripts/causal_check.py --frames data/2011_09_26/..._0051_sync \\
        --drive data/2011_09_26/..._0051_sync --calib-dir data/2011_09_26 \\
        --start 140 --count 60      # the invocation the README table reports
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hazard.device import apply_memory_guards  # noqa: E402

apply_memory_guards()

import numpy as np  # noqa: E402

from hazard.backbone import WindowGeometry, load_backbone, run_window  # noqa: E402
from hazard.data import load_sequence  # noqa: E402
from hazard.detect import BlobTracker, extract_blobs  # noqa: E402
from hazard.device import describe, free_cache, pick_device  # noqa: E402
from hazard.forecast import ForecastContext, RigidFlowForecaster  # noqa: E402
from hazard.kitti_calib import KittiCalib, project_box  # noqa: E402
from hazard.kitti_labels import DYNAMIC_TYPES, mark_moving, parse_tracklets  # noqa: E402
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


def score(maps, stems_used, tracklets, calib, threshold: float = THRESHOLD):
    tracker = BlobTracker(centre_frac=1.6)
    n_mov = n_hit = n_park = n_pfp = n_det = d_mov = 0
    for stem in stems_used:
        raw = int(stem)
        dets = [d for d in tracker.update(
            extract_blobs(maps[stem], raw, threshold=threshold, min_area=60), raw)
            if d.age >= 2]
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
            hit = any(_ov(d.box, b) >= 0.3 for d in dets)
            if mv:
                n_mov += 1
                n_hit += hit
            else:
                n_park += 1
                n_pfp += hit
        n_det += len(dets)
        d_mov += sum(1 for d in dets if any(_ov(d.box, b) >= 0.3 for b, mv in objs if mv))
    rec, prec = n_hit / max(n_mov, 1), d_mov / max(n_det, 1)
    return dict(recall=rec, precision=prec, parked_fp=n_pfp / max(n_park, 1),
                f1=2 * rec * prec / (rec + prec) if rec + prec else 0.0,
                n_det=n_det, n_mov=n_mov)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--drive", required=True)
    ap.add_argument("--calib-dir", required=True)
    ap.add_argument("--start", type=int, default=140)
    ap.add_argument("--count", type=int, default=60)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--weights",
                    default=str(Path(__file__).resolve().parent.parent
                                / "weights" / "vggt_1b.safetensors"))
    args = ap.parse_args()

    device = pick_device()
    print(f"[env] {describe(device)}")
    model = load_backbone(args.weights, device)
    tracklets = mark_moving(parse_tracklets(Path(args.drive) / "tracklet_labels.xml"))
    calib = KittiCalib(args.calib_dir)

    frames, paths = load_sequence(args.frames, start=args.start, count=args.count,
                                  stride=args.stride)
    n = len(frames)
    stems = [p.stem for p in paths]
    fc = RigidFlowForecaster()

    # ---------- non-causal: the shipped chunked protocol --------------------
    nc_maps, step = {}, args.chunk - 2
    for s in range(0, max(1, n - 2), step):
        e = min(s + args.chunk, n)
        if e - s <= 2:
            break
        geom = run_window(model, frames[s:e], device)
        for local in range(2, e - s):
            g = s + local
            if stems[g] in nc_maps:
                continue
            f = fc.forecast(ForecastContext(geom=geom, frames=frames[s:e], target=local))
            nc_maps[stems[g]] = compute_surprise(f, geom, local, threshold=THRESHOLD,
                                                **SURPRISE_KW)
        free_cache(device)

    # ---------- causal: two passes, aligned on the shared frame ------------
    c_maps, ratios = {}, []
    for t in range(2, n):
        geom_a = run_window(model, frames[t - 2:t], device)          # [t-2, t-1]
        geom_b = run_window(model, frames[t - 1:t + 1], device)      # [t-1, t]

        # Align B into A's scale using their shared frame t-1.
        ma = float(np.median(geom_a.depth[1]))
        mb = float(np.median(geom_b.depth[0]))
        ratio = ma / max(mb, 1e-6)
        ratios.append(ratio)

        # Forecast for t needs a 3-slot window in A's scale; the target slot is
        # only used for its shape and for the observation we substitute.
        stitched = WindowGeometry(
            depth=np.stack([geom_a.depth[0], geom_a.depth[1], geom_b.depth[1] * ratio]),
            conf=np.stack([geom_a.conf[0], geom_a.conf[1], geom_b.conf[1]]),
            extrinsic=np.stack([geom_a.extrinsic[0], geom_a.extrinsic[1],
                                geom_a.extrinsic[1]]),
            intrinsic=np.stack([geom_a.intrinsic[0], geom_a.intrinsic[1],
                                geom_a.intrinsic[1]]),
            seconds=0.0,
        )
        f = fc.forecast(ForecastContext(geom=stitched, frames=frames[t - 2:t + 1],
                                        target=2))
        c_maps[stems[t]] = compute_surprise(f, stitched, 2, threshold=THRESHOLD,
                                           **SURPRISE_KW)
        free_cache(device)
        if t % 10 == 0:
            print(f"  causal {t}/{n - 1}", flush=True)

    shared = [s for s in stems if s in nc_maps and s in c_maps]
    print(f"\nframes scored by both protocols: {len(shared)}")
    print(f"scale-alignment ratio across frames: median {np.median(ratios):.4f}, "
          f"IQR {np.percentile(ratios, 25):.4f}-{np.percentile(ratios, 75):.4f}")

    # Compared as curves, not at one threshold. The two protocols emit
    # different numbers of detections at the same threshold (151 vs 113 at
    # z=8), so a single-row comparison is reading two different operating
    # points and calling it a difference between protocols.
    grid = [4, 6, 8, 10, 13, 16, 20, 25]

    def ap(curve):
        pts = sorted(((c["recall"], c["precision"]) for c in curve))
        rp, a = 0.0, 0.0
        for r, pr in pts:
            a += (r - rp) * pr
            rp = r
        return a

    curves = {}
    for name, maps in (("non-causal (shipped chunk)", nc_maps),
                       ("strictly causal (2 passes)", c_maps)):
        curve = [score(maps, shared, tracklets, calib, threshold=t) for t in grid]
        curves[name] = curve
        print(f"\n{name}")
        print(f"  {'thresh':>7} {'recall':>8} {'prec':>7} {'parkFP':>7} {'F1':>6} {'dets':>6}")
        for t, c in zip(grid, curve):
            f1 = (2 * c["recall"] * c["precision"] / (c["recall"] + c["precision"])
                  if c["recall"] + c["precision"] else 0.0)
            print(f"  {t:>7} {100 * c['recall']:>7.1f}% {100 * c['precision']:>6.1f}% "
                  f"{100 * c['parked_fp']:>6.1f}% {f1:>6.3f} {c['n_det']:>6}")
        print(f"  -> AP {ap(curve):.3f}   best F1 {max(c['f1'] for c in curve):.3f}")

    print(f"\n{'=' * 74}")
    print("  comparison at matched detection count and matched false-alarm rate")
    print("=" * 74)
    print(f"  {'protocol':<28} {'AP':>6} {'bestF1':>8} {'R@FP<=20%':>11} {'R@FP<=10%':>11}")
    for name, curve in curves.items():
        cells = []
        for cap in (0.20, 0.10):
            ok = [c for c in curve if c["parked_fp"] <= cap]
            cells.append(f"{100 * max((c['recall'] for c in ok), default=0):>10.1f}%")
        print(f"  {name:<28} {ap(curve):>6.3f} "
              f"{max(c['f1'] for c in curve):>8.3f} " + " ".join(cells))
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
