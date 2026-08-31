#!/usr/bin/env python3
"""Unsupervised hazard detection from a geometry forecast.

    python run.py --frames data/kitti/2011_09_26_drive_0009_sync --count 60

Ask the model what the 3D shape of the world will look like a moment from now,
then look at the moment when it arrives. Wherever the guess was wrong, in a
tight blob, something moved that the static world could not account for.

No object detector, no class list, no labels, no training.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Memory guards must be installed before torch initialises its MPS allocator,
# so this import block runs before anything touches torch.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from hazard.device import apply_memory_guards  # noqa: E402

apply_memory_guards()

import torch  # noqa: E402

from hazard import pipeline  # noqa: E402
from hazard.backbone import load_backbone  # noqa: E402
from hazard.device import describe, pick_device  # noqa: E402
from hazard.forecast import build_forecaster  # noqa: E402

DEFAULT_WEIGHTS = Path(__file__).resolve().parent / "weights" / "vggt_1b.safetensors"


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    src = p.add_argument_group("input")
    src.add_argument("--frames", required=True,
                     help="KITTI drive root, an image_02/data dir, or any folder of frames")
    src.add_argument("--start", type=int, default=0, help="skip this many frames first")
    src.add_argument("--count", type=int, default=None, help="how many frames to score")
    src.add_argument("--stride", type=int, default=2,
                     help="sample every Nth frame (2 = the paper's KITTI protocol)")

    mdl = p.add_argument_group("model")
    mdl.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    mdl.add_argument("--forecaster", default="rigid",
                     choices=["rigid", "identity", "vggtworld"])
    mdl.add_argument("--vggtworld-ckpt", default=None,
                     help="path to the paper's kitti_checkpoint.pt")
    mdl.add_argument("--device", default="auto")
    mdl.add_argument("--fp32", action="store_true",
                     help="run the trunk in fp32 (2x the memory, marginally better depth)")

    det = p.add_argument_group("detector")
    det.add_argument("--chunk", type=int, default=8,
                     help="frames per VGGT pass; lower it if you hit an MPS OOM")
    det.add_argument("--detector", default="blob", choices=["blob", "window"],
                     help="blob = connected components (default, and the better one); "
                          "window = multi-scale region scoring, kept but measurably worse")
    det.add_argument("--window-threshold", type=float, default=3.0,
                     help="mean surprise over a region, for --detector window")
    det.add_argument("--threshold", type=float, default=8.0,
                     help="robust z-score above which a pixel counts as surprising; "
                          "8 emits ~0.65 detections/frame, see README")
    det.add_argument("--min-area", type=int, default=60, help="smallest blob, in pixels")
    det.add_argument("--min-track-len", type=int, default=2,
                     help="frames a blob must persist before it is reported")
    det.add_argument("--top-k", type=int, default=8)
    det.add_argument("--no-whiten", action="store_true",
                     help="disable the residual noise model, hard-mask depth edges instead")
    det.add_argument("--no-normalised-blur", action="store_true",
                     help="disable normalised smoothing (masked pixels drag neighbours down)")
    det.add_argument("--row-scaled-area", action="store_true",
                     help="scale min_area with distance below the horizon: reaches 70%% recall "
                          "but costs precision, so it is off by default")
    det.add_argument("--full-fov", action="store_true",
                     help="224x742 instead of the 224x448 centre crop, recovering the 40%% of "
                          "horizontal field of view the crop discards")

    out = p.add_argument_group("output")
    out.add_argument("--out", default="outputs/run")
    out.add_argument("--save-maps", action="store_true",
                     help="dump the raw surprise map per frame as .npz (for analysis)")
    out.add_argument("--no-panels", action="store_true",
                     help="skip the 4-up diagnostic images (faster, less disk)")

    args = p.parse_args()

    device = pick_device(args.device)
    print(f"[env] {describe(device)}")
    if device.type == "cpu":
        print("[env] WARNING: no GPU backend found; this will be extremely slow.")

    trunk_dtype = torch.float32 if args.fp32 else torch.bfloat16
    model = load_backbone(args.weights, device, trunk_dtype=trunk_dtype)

    forecaster = build_forecaster(
        args.forecaster, device=device, ckpt=args.vggtworld_ckpt
    )

    summary = pipeline.run(
        model,
        forecaster,
        args.frames,
        args.out,
        device,
        start=args.start,
        count=args.count,
        stride=args.stride,
        chunk=args.chunk,
        threshold=args.threshold,
        min_area=args.min_area,
        detector=args.detector,
        surprise_kw=dict(whiten=not args.no_whiten,
                         normalised_blur=not args.no_normalised_blur),
        detect_kw=dict(row_scaled_area=args.row_scaled_area),
        full_fov=args.full_fov,
        window_threshold=args.window_threshold,
        min_track_len=args.min_track_len,
        top_k=args.top_k,
        save_panels=not args.no_panels,
        save_maps=args.save_maps,
        trunk_dtype=trunk_dtype,
    )

    print("\n" + "=" * 64)
    print(f"scored {summary['frames_scored']} frames in {summary['seconds_total']}s "
          f"({summary['seconds_per_frame']}s/frame)")
    print(f"confirmed tracks (persisted >= {args.min_track_len} frames): "
          f"{summary['confirmed_tracks']}")
    print("\nmost surprising moments:")
    for row in summary["top_frames"]:
        print(f"  frame {row['frame']:>5}  {row['source']:<24} "
              f"peak z={row['peak_z']:>6.1f}  area={row['area_pct']:.3f}%")
    print(f"\nwrote {args.out}/  (top_moments.png, surprise.mp4, panels/, frame_scores.csv)")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
