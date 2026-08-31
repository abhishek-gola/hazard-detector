#!/usr/bin/env python3
"""Cache VGGT geometry for every drive that has frames and KittiMoSeg labels.

The scale-up is compute, not ideas: labels exist for all 38 drives in the
release, and each drive needs one pass of geometry caching before it can be
scored. Seven drives gave 2,024 moving instances over 108 tracks; all 38 should
give roughly five times that, which turns "per-drive precision ranges 18-83 %"
from seven anecdotes into a distribution.

Windows are chosen from the **KittiMoSeg** labels rather than the self-derived
ones, since those labels turned out to be only 58.9 % complete and would pick
windows biased toward fast movers.

Resumable: a drive with a manifest is skipped, so this can be re-run as more
KITTI raw downloads land.

    python scripts/scale_up.py --count 110          # all available drives
    python scripts/scale_up.py --drives 0001,0002   # named drives only
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hazard import kittimoseg as ms  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "2011_09_26"
LABELS = ROOT / "data" / "moseg"
CACHE = ROOT / "cache"


def densest_window(label_dir: Path, n_frames: int, span: int, stride: int) -> int:
    """Start frame of the window holding the most KittiMoSeg moving instances."""
    counts = {}
    for f in ms.frames_available(label_dir):
        if f >= n_frames:
            continue
        counts[f] = sum(1 for o in ms.load_objects(label_dir, f) if o.moving)
    if not counts:
        return 0
    best_start, best = 0, -1
    raw_span = span * stride
    for s in range(0, max(1, n_frames - raw_span), 10):
        c = sum(counts.get(f, 0) for f in range(s, min(s + raw_span, n_frames)))
        if c > best:
            best, best_start = c, s
    return best_start


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--drives", default="", help="comma list; default = all available")
    ap.add_argument("--count", type=int, default=110, help="strided frames per drive")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--chunk", type=int, default=8)
    args = ap.parse_args()

    wanted = ([d.strip() for d in args.drives.split(",")] if args.drives
              else sorted(p.name for p in LABELS.iterdir() if p.is_dir()))

    todo, skipped = [], []
    for dv in wanted:
        frames_dir = RAW / f"2011_09_26_drive_{dv}_sync" / "image_02" / "data"
        ldir = LABELS / dv
        if not frames_dir.is_dir():
            skipped.append((dv, "no KITTI raw frames"))
            continue
        if not (ldir / "Text_data_Output").is_dir():
            skipped.append((dv, "no KittiMoSeg labels"))
            continue
        if (CACHE / f"d{dv}" / "manifest.json").exists():
            skipped.append((dv, "already cached"))
            continue
        n = len(list(frames_dir.glob("*.png")))
        if n < 3 * args.stride + 6:
            skipped.append((dv, f"only {n} frames"))
            continue
        start = densest_window(ldir, n, args.count, args.stride)
        count = min(args.count, (n - start) // args.stride)
        todo.append((dv, start, count))

    print(f"to cache: {len(todo)}   skipped: {len(skipped)}")
    for dv, why in skipped:
        print(f"  skip {dv}: {why}")

    for i, (dv, start, count) in enumerate(todo, 1):
        print(f"\n[{i}/{len(todo)}] drive {dv}  start={start} count={count}", flush=True)
        r = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "cache_geometry.py"),
             "--frames", str(RAW / f"2011_09_26_drive_{dv}_sync"),
             "--start", str(start), "--count", str(count),
             "--stride", str(args.stride), "--chunk", str(args.chunk),
             "--out", str(CACHE / f"d{dv}")])
        if r.returncode != 0:
            print(f"  drive {dv} FAILED (rc={r.returncode}), continuing", file=sys.stderr)

    cached = sorted(p.name[1:] for p in CACHE.glob("d*")
                    if (p / "manifest.json").exists())
    print(f"\ncached drives ({len(cached)}): {','.join(cached)}")
    print(f"score them with:\n  python scripts/eval_moseg.py --drives {','.join(cached)} "
          f"--calib-dir data/2011_09_26")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
