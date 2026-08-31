#!/usr/bin/env python3
"""Run VGGT once over a segment and cache the geometry it produces.

Everything downstream of the backbone -- the forecast, the noise model, the
blur, the thresholds, the blob rules -- is cheap numpy. Only VGGT is expensive.
So dump depth, confidence and camera parameters per chunk, and every subsequent
ablation replays from disk in a second instead of a minute.

Cached per chunk rather than per frame because that is the unit that matters:
VGGT's global attention runs across the whole window, so depths and poses are
only mutually consistent inside one forward pass, and the forecast for a frame
uses its two predecessors from that same pass.

    python scripts/cache_geometry.py --frames data/2011_09_26/..._0009_sync \\
        --start 200 --count 105 --out cache/junction
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hazard.device import apply_memory_guards  # noqa: E402

apply_memory_guards()

import numpy as np  # noqa: E402
import torch  # noqa: E402

from hazard.backbone import load_backbone, run_window  # noqa: E402
from hazard.data import load_sequence  # noqa: E402
from hazard.device import describe, free_cache, pick_device  # noqa: E402

DEFAULT_WEIGHTS = Path(__file__).resolve().parent.parent / "weights" / "vggt_1b.safetensors"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=None)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--ctx", type=int, default=2, help="forecaster context length")
    ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--target-h", type=int, default=224)
    ap.add_argument("--target-w", type=int, default=448)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    device = pick_device()
    print(f"[env] {describe(device)}")
    model = load_backbone(args.weights, device)

    frames, paths = load_sequence(
        args.frames, start=args.start, count=args.count, stride=args.stride,
        target_h=args.target_h, target_w=args.target_w,
    )
    n = len(frames)
    step = args.chunk - args.ctx
    starts = list(range(0, max(1, n - args.ctx), step))
    print(f"[cache] {n} frames, {len(starts)} chunks of {args.chunk} "
          f"({args.target_h}x{args.target_w})")

    manifest = {
        "frames": [p.stem for p in paths],
        "chunk": args.chunk, "ctx": args.ctx, "stride": args.stride,
        "start": args.start, "target_hw": [args.target_h, args.target_w],
        "chunks": [],
    }

    for ci, s in enumerate(starts):
        e = min(s + args.chunk, n)
        if e - s <= args.ctx:
            break
        geom = run_window(model, frames[s:e], device)
        np.savez_compressed(
            out / f"chunk{ci:04d}.npz",
            # float32, not float16. The noise model takes central differences
            # of log depth, and float16's ~0.1% relative quantisation shows up
            # directly in that gradient -- which is the denominator of the
            # whitened residual. Halving the file size is not worth perturbing
            # the statistic every downstream number depends on.
            depth=geom.depth.astype(np.float32),
            conf=geom.conf.astype(np.float32),
            extrinsic=geom.extrinsic.astype(np.float64),
            intrinsic=geom.intrinsic.astype(np.float64),
            images=(frames[s:e] * 255).astype(np.uint8),
        )
        manifest["chunks"].append({"file": f"chunk{ci:04d}.npz", "start": s, "end": e})
        free_cache(device)
        print(f"  chunk {ci + 1}/{len(starts)}  frames {s}-{e - 1}  "
              f"vggt {geom.seconds:.1f}s", flush=True)

    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    total = sum(f.stat().st_size for f in out.glob("*.npz")) / 1e6
    print(f"[cache] wrote {out}  ({total:.0f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
