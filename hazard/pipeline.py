"""The driver: frames in, ranked hazards out.

The one structural decision worth explaining is chunking. VGGT's global
attention runs across every frame in a window, so depth and pose are only
comparable inside a single forward pass -- run it twice and you get two
different arbitrary scales. That rules out a naive sliding window, which would
need a fresh pass per frame anyway and would be brutally slow.

So instead we process a chunk of K frames in one pass and harvest K - 2
forecasts from it, each predicting one frame from its own two predecessors.
Consecutive chunks overlap by two frames so the sequence stays continuous.
That is a ~6x saving over per-frame windows at K=8, and it makes scale
consistency free rather than something to correct for afterwards.

The forecast still only ever looks backwards. VGGT is the measuring instrument
here, not the predictor -- the same split the paper's own evaluation uses.
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from . import viz
from .backbone import VGGTCore, run_window
from .data import load_sequence
from .detect import BlobTracker, detect_windows, extract_blobs
from .device import free_cache, memory_used_gb
from .forecast import ForecastContext, Forecaster
from .surprise import compute_surprise


@dataclass
class FrameResult:
    frame_idx: int
    source: str
    peak: float
    area: float
    frame_median_residual: float
    n_blobs: int
    top_blob_peak: float
    seconds: float


def run(
    model: VGGTCore,
    forecaster: Forecaster,
    frames_root: str | Path,
    out_dir: str | Path,
    device,
    *,
    start: int = 0,
    count: int | None = None,
    stride: int = 2,
    chunk: int = 8,
    threshold: float = 4.0,
    min_area: int = 60,
    min_track_len: int = 2,
    detector: str = "blob",
    surprise_kw: dict | None = None,
    detect_kw: dict | None = None,
    full_fov: bool = False,
    window_threshold: float = 3.0,
    top_k: int = 8,
    save_panels: bool = True,
    save_maps: bool = False,
    trunk_dtype=None,
) -> dict:
    import torch

    trunk_dtype = trunk_dtype or torch.bfloat16
    surprise_kw = surprise_kw or {}
    detect_kw = detect_kw or {}
    out_dir = Path(out_dir)
    (out_dir / "panels").mkdir(parents=True, exist_ok=True)
    (out_dir / "overlays").mkdir(parents=True, exist_ok=True)
    if save_maps:
        (out_dir / "maps").mkdir(parents=True, exist_ok=True)

    from .data import CROP_FOV, FULL_FOV
    t_h, t_w = FULL_FOV if full_fov else CROP_FOV
    frames, paths = load_sequence(frames_root, start=start, count=count, stride=stride,
                                  target_h=t_h, target_w=t_w)
    n = len(frames)
    ctx_len = forecaster.n_context
    if chunk <= ctx_len:
        raise ValueError(f"chunk must exceed the forecaster's context ({ctx_len})")

    print(f"[pipeline] {n} frames @ stride {stride} from {frames_root}")
    print(f"[pipeline] forecaster={forecaster.name} chunk={chunk} "
          f"targets/chunk={chunk - ctx_len} detector={detector} "
          f"threshold={window_threshold if detector == 'window' else threshold}")

    tracker = BlobTracker()
    results: list[FrameResult] = []
    overlays: list[np.ndarray] = []
    per_frame_cache: dict[int, tuple] = {}

    step = chunk - ctx_len
    t_start = time.time()
    chunk_starts = list(range(0, max(1, n - ctx_len), step))

    for ci, s in enumerate(chunk_starts):
        e = min(s + chunk, n)
        if e - s <= ctx_len:
            break
        window = frames[s:e]

        geom = run_window(model, window, device, trunk_dtype=trunk_dtype)

        for local in range(ctx_len, e - s):
            g_idx = s + local
            if g_idx in per_frame_cache:  # overlap between chunks
                continue
            t0 = time.time()

            fc = forecaster.forecast(
                ForecastContext(geom=geom, frames=window, target=local)
            )
            sm = compute_surprise(fc, geom, local, threshold=threshold, **surprise_kw)
            if detector == "window":
                blobs = detect_windows(sm, g_idx, threshold=window_threshold)
            else:
                blobs = extract_blobs(sm, g_idx, threshold=threshold,
                                      min_area=min_area, **detect_kw)
            blobs = tracker.update(blobs, g_idx)

            over = viz.overlay_surprise(window[local], sm, blobs, threshold=threshold,
                                        min_age=min_track_len)
            overlays.append(over)
            cv2.imwrite(str(out_dir / "overlays" / f"{g_idx:05d}.png"), over)

            if save_panels:
                pan = viz.panel(window[local], fc.depth, geom.depth[local], sm, blobs,
                                g_idx, threshold=threshold)
                cv2.imwrite(str(out_dir / "panels" / f"{g_idx:05d}.png"), pan)

            if save_maps:
                np.savez_compressed(
                    out_dir / "maps" / f"{paths[g_idx].stem}.npz",
                    score=sm.score.astype(np.float16),
                    valid=sm.valid,
                )

            res = FrameResult(
                frame_idx=g_idx,
                source=paths[g_idx].name,
                peak=sm.peak,
                area=sm.area,
                frame_median_residual=sm.frame_median,
                n_blobs=len(blobs),
                top_blob_peak=max((b.peak for b in blobs), default=0.0),
                seconds=time.time() - t0,
            )
            results.append(res)
            per_frame_cache[g_idx] = (sm, blobs)

        free_cache(device)
        done = len(results)
        rate = (time.time() - t_start) / max(done, 1)
        print(
            f"  chunk {ci + 1}/{len(chunk_starts)}  frames {s}-{e - 1}  "
            f"vggt {geom.seconds:5.1f}s  mem {memory_used_gb(device):.1f}GB  "
            f"{rate:.1f}s/frame  eta {(n - done) * rate / 60:.1f}min",
            flush=True,
        )

    elapsed = time.time() - t_start

    # ---- rank and report -------------------------------------------------
    confirmed = tracker.confirmed(min_length=min_track_len)
    results.sort(key=lambda r: r.frame_idx)

    with open(out_dir / "frame_scores.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(asdict(results[0]).keys()) if results else [])
        if results:
            w.writeheader()
            for r in results:
                w.writerow(asdict(r))

    # Every blob, keyed by the source filename, so validation can line them up
    # with KITTI frame numbers without re-deriving the stride arithmetic.
    blob_dump = {
        paths[idx].stem: [
            {"box": list(b.box), "peak": round(b.peak, 3), "mean": round(b.mean, 3),
             "area_px": b.area_px, "track_id": b.track_id, "age": b.age}
            for b in blobs
        ]
        for idx, (_, blobs) in per_frame_cache.items()
    }
    with open(out_dir / "blobs.json", "w") as fh:
        json.dump(blob_dump, fh, indent=1)

    ranked = sorted(results, key=lambda r: r.top_blob_peak, reverse=True)[:top_k]
    sheet_imgs = []
    for r in ranked:
        sm, blobs = per_frame_cache[r.frame_idx]
        f_local = r.frame_idx
        img = viz.overlay_surprise(frames[f_local], sm, blobs, threshold=threshold,
                                   min_age=min_track_len)
        sheet_imgs.append(viz.banner(
            img, f"frame {r.frame_idx} ({r.source})  peak z={r.top_blob_peak:.1f}"))
    if sheet_imgs:
        cv2.imwrite(str(out_dir / "top_moments.png"), viz.contact_sheet(sheet_imgs, cols=2))

    made_video = viz.write_video(overlays, out_dir / "surprise.mp4", fps=5.0)

    summary = {
        "frames_scored": len(results),
        "forecaster": forecaster.name,
        "stride": stride,
        "chunk": chunk,
        "detector": detector,
        "target_hw": [t_h, t_w],
        "surprise_kw": surprise_kw,
        "detect_kw": detect_kw,
        "threshold": window_threshold if detector == "window" else threshold,
        "seconds_total": round(elapsed, 1),
        "seconds_per_frame": round(elapsed / max(len(results), 1), 2),
        "confirmed_tracks": len(confirmed),
        "top_tracks": [
            {
                "track_id": t.track_id,
                "frames": [b.frame for b in t.blobs],
                "length": t.length,
                "peak_z": round(t.peak, 2),
                "box_first": t.blobs[0].box,
            }
            for t in confirmed[:top_k]
        ],
        "top_frames": [
            {"frame": r.frame_idx, "source": r.source,
             "peak_z": round(r.top_blob_peak, 2), "area_pct": round(r.area * 100, 3)}
            for r in ranked
        ],
        "video": bool(made_video),
    }
    with open(out_dir / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    return summary
