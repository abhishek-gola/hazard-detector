#!/usr/bin/env python3
"""Run the frozen detector over held-out drives, once, and report with CIs.

Protocol, stated so it can be checked. Everything -- the noise model, the
normalised blur, the threshold of 8, the tracker gate, the blob rules -- was
chosen on drive 0009. Drives 0013 and 0014 were downloaded afterwards and this
script is the first and only thing run on them. No knob is touched after seeing
these numbers; if the result is worse, the result is worse.

Confidence intervals are bootstrapped over **tracklets**, not instances. One
moving car contributes twenty highly-correlated frames, so resampling instances
would treat twenty views of the same event as twenty independent trials and give
intervals several times too narrow. Resampling whole tracklets keeps the unit of
independence honest.

    python scripts/test_once.py --drives 0013,0014 --calib-dir data/2011_09_26
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hazard.backbone import WindowGeometry  # noqa: E402
from hazard.baseline_flow import flow_surprise  # noqa: E402
from hazard.detect import BlobTracker, extract_blobs  # noqa: E402
from hazard.forecast import ForecastContext, RigidFlowForecaster  # noqa: E402
from hazard.kitti_calib import KittiCalib, project_box  # noqa: E402
from hazard.kitti_labels import (DYNAMIC_TYPES, mark_moving,  # noqa: E402
                                 parse_tracklets, residual_velocity)
from hazard.surprise import compute_surprise  # noqa: E402

# FROZEN. Chosen on drive 0009 before any held-out drive was downloaded.
THRESHOLD = 8.0
SURPRISE_KW = dict(whiten=True, normalised_blur=True)
DETECT_KW = dict(row_scaled_area=False, min_area=60)
MIN_AGE = 2
HIT_FRAC = 0.3


def _ov(det, obj) -> float:
    dx, dy, dw, dh = det
    ox, oy, ow, oh = obj
    x0, y0 = max(dx, ox), max(dy, oy)
    x1, y1 = min(dx + dw, ox + ow), min(dy + dh, oy + oh)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return (x1 - x0) * (y1 - y0) / float(max(dw * dh, 1))


def run_drive(cache: Path, drive: Path, calib: KittiCalib, method: str):
    """-> list of per-instance records, plus detection bookkeeping."""
    tracklets = mark_moving(parse_tracklets(drive / "tracklet_labels.xml"))
    resid_vel = residual_velocity(tracklets)
    man = json.loads((cache / "manifest.json").read_text())
    stems, ctx = man["frames"], man["ctx"]
    fc = RigidFlowForecaster()
    tracker = BlobTracker(centre_frac=1.6)

    maps = {}
    for ch in man["chunks"]:
        d = np.load(cache / ch["file"])
        geom = WindowGeometry(
            depth=d["depth"].astype(np.float32), conf=d["conf"].astype(np.float32),
            extrinsic=d["extrinsic"], intrinsic=d["intrinsic"], seconds=0.0,
        )
        imgs_u8 = d["images"]
        imgs = imgs_u8.astype(np.float32) / 255.0
        s, e = ch["start"], ch["end"]
        for local in range(ctx, e - s):
            g = s + local
            if stems[g] in maps:
                continue
            if method == "depth":
                f = fc.forecast(ForecastContext(geom=geom, frames=imgs, target=local))
                maps[stems[g]] = compute_surprise(f, geom, local, threshold=THRESHOLD,
                                                  **SURPRISE_KW)
            else:
                backend = "raft" if method == "raft" else "farneback"
                maps[stems[g]] = flow_surprise(imgs_u8[local - 1], imgs_u8[local],
                                               threshold=THRESHOLD, normalise=False,
                                               backend=backend)

    records, n_det, d_mov, n_park, n_pfp = [], 0, 0, 0, 0
    for stem in sorted(maps):
        raw = int(stem)
        dets = tracker.update(
            extract_blobs(maps[stem], raw, threshold=THRESHOLD, **DETECT_KW), raw)
        dets = [d for d in dets if d.age >= MIN_AGE]
        objs = []
        for t in tracklets:
            p = t.at(raw)
            if p is None or t.object_type not in DYNAMIC_TYPES:
                continue
            i = raw - t.first_frame
            b = project_box(calib, p, t.h, t.w, t.l, float(t.rotations[i][2]))
            if b:
                objs.append((b, t.is_moving_at(raw), id(t), t.object_type))
        for b, mv, tid, otype in objs:
            hit = any(_ov(d.box, b) >= HIT_FRAC for d in dets)
            if mv:
                rv = resid_vel.get((tid, raw), (float("nan"), float("nan")))
                records.append({"tracklet": tid, "hit": bool(hit),
                                "area": b[2] * b[3], "type": otype,
                                "radial": rv[0], "tangential": rv[1]})
            else:
                n_park += 1
                n_pfp += hit
        n_det += len(dets)
        d_mov += sum(1 for d in dets
                     if any(_ov(d.box, b) >= HIT_FRAC for b, mv, _, _ in objs if mv))
    return records, dict(n_det=n_det, d_mov=d_mov, n_park=n_park, n_pfp=n_pfp,
                         n_frames=len(maps))


def bootstrap_recall(records, n_boot=4000, seed=0):
    """Resample tracklets with replacement; recall is over their instances."""
    if not records:
        return 0.0, 0.0, 0.0
    by_t: dict[int, list[bool]] = {}
    for r in records:
        by_t.setdefault(r["tracklet"], []).append(r["hit"])
    keys = list(by_t)
    rng = np.random.default_rng(seed)
    point = np.mean([h for r in records for h in [r["hit"]]])
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(len(keys), len(keys), replace=True)
        hits = [h for i in pick for h in by_t[keys[i]]]
        vals.append(np.mean(hits) if hits else 0.0)
    return point, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--drives", default="0013,0014")
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--data", default="data/2011_09_26")
    ap.add_argument("--calib-dir", default="data/2011_09_26")
    ap.add_argument("--methods", default="depth,flow,raft")
    args = ap.parse_args()

    calib = KittiCalib(args.calib_dir)
    drives = args.drives.split(",")
    methods = args.methods.split(",")

    print("=" * 84)
    print("HELD-OUT TEST -- frozen config, run once")
    print(f"  threshold {THRESHOLD}  surprise {SURPRISE_KW}  detect {DETECT_KW}")
    print("=" * 84)

    pooled = {m: ([], dict(n_det=0, d_mov=0, n_park=0, n_pfp=0, n_frames=0))
              for m in methods}

    for dv in drives:
        cache = Path(args.cache) / f"d{dv}"
        drive = Path(args.data) / f"2011_09_26_drive_{dv}_sync"
        if not (cache / "manifest.json").exists():
            print(f"\n  drive {dv}: no cache, skipped")
            continue
        print(f"\ndrive {dv}")
        print(f"  {'method':<8} {'frames':>7} {'inst':>6} {'trk':>5} {'recall':>8} "
              f"{'95% CI':>16} {'prec':>7} {'parkFP':>7}")
        for m in methods:
            rec, bk = run_drive(cache, drive, calib, m)
            pooled[m][0].extend(rec)
            for k in bk:
                pooled[m][1][k] += bk[k]
            pt, lo, hi = bootstrap_recall(rec)
            ntrk = len({r["tracklet"] for r in rec})
            prec = bk["d_mov"] / max(bk["n_det"], 1)
            pfp = bk["n_pfp"] / max(bk["n_park"], 1)
            print(f"  {m:<8} {bk['n_frames']:>7} {len(rec):>6} {ntrk:>5} "
                  f"{100 * pt:>7.1f}% [{100 * lo:>5.1f}, {100 * hi:>5.1f}]% "
                  f"{100 * prec:>6.1f}% {100 * pfp:>6.1f}%")

    print(f"\n{'=' * 84}")
    print("POOLED over held-out drives")
    print("=" * 84)
    print(f"  {'method':<8} {'inst':>6} {'trk':>5} {'recall':>8} {'95% CI':>16} "
          f"{'prec':>7} {'parkFP':>7}")
    for m in methods:
        rec, bk = pooled[m]
        if not rec:
            continue
        pt, lo, hi = bootstrap_recall(rec)
        ntrk = len({r["tracklet"] for r in rec})
        print(f"  {m:<8} {len(rec):>6} {ntrk:>5} {100 * pt:>7.1f}% "
              f"[{100 * lo:>5.1f}, {100 * hi:>5.1f}]% "
              f"{100 * bk['d_mov'] / max(bk['n_det'], 1):>6.1f}% "
              f"{100 * bk['n_pfp'] / max(bk['n_park'], 1):>6.1f}%")

    if "depth" in pooled and pooled["depth"][0]:
        rec = pooled["depth"][0]
        print("\nrecall by TANGENTIAL residual speed (depth method, pooled)")
        print("  the mechanism: tangential motion moves an object onto pixels the")
        print("  forecast had assigned to background; radial motion barely shifts depth")
        for lo_t, hi_t, nm in [(0, 0.15, "0.00-0.15"), (0.15, 0.40, "0.15-0.40"),
                               (0.40, 0.80, "0.40-0.80"), (0.80, 1e9, "0.80+")]:
            sel = [r for r in rec
                   if not np.isnan(r["tangential"]) and lo_t <= r["tangential"] < hi_t]
            if not sel:
                continue
            pt, lo, hi = bootstrap_recall(sel)
            med_a = np.median([r["area"] for r in sel])
            print(f"  {nm:<11} m/frame  n={len(sel):>4}  recall {100 * pt:>5.1f}% "
                  f"[{100 * lo:>5.1f}, {100 * hi:>5.1f}]%   median area {med_a:>6.0f} px2")

    # recall by apparent size, pooled, depth method only
    if "depth" in pooled and pooled["depth"][0]:
        print("\nrecall by apparent size (depth method, pooled held-out)")
        rec = pooled["depth"][0]
        for lo_a, hi_a, nm in [(0, 600, "< 600 px2"), (600, 2000, "600-2000"),
                               (2000, 10 ** 9, "> 2000")]:
            sel = [r for r in rec if lo_a <= r["area"] < hi_a]
            if not sel:
                continue
            pt, lo, hi = bootstrap_recall(sel)
            print(f"  {nm:<12} n={len(sel):>4}  recall {100 * pt:>5.1f}% "
                  f"[{100 * lo:>5.1f}, {100 * hi:>5.1f}]%")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
