# Hazard detection by geometry forecasting

Ask a model what the 3D shape of the world will look like a moment from now.
Then wait for the moment to arrive and look at it. Wherever the guess was wrong
*in a tight blob*, something moved that a static world cannot account for.

No object detector. No class list. No labels. No training. The detector has
never been told what a car is, and it never learns.

Everything here runs on an Apple Silicon Mac. Measured on a **MacBook Air M5,
16 GB**: **0.59 s per frame, 2.8 GB peak memory.**

---

## What it does

VGGT-World (arXiv:2603.12655) forecasts the *geometry* of the next frame
instead of its pixels. Their own demo prints two depth maps side by side: the
forecast, and what VGGT actually measured once the frame arrived. They ship it
as a quality comparison.

But the disagreement between those two maps is a **surprise signal**, and that
is the whole project.

Most of a driving scene evolves predictably. The road recedes, buildings sweep
past, parked cars grow in the frame — all of it follows from ego motion, and a
forecast nails it. What no forecast can anticipate is something moving under its
own agency: a car cutting into your lane, a pedestrian stepping out from between
parked vans. There, the guess fails, and it fails *exactly where the surprising
thing is*.

The property that makes this more than a re-derived object detector: **a car
travelling steadily beside you is not surprising.** Constant-velocity scene flow
is predictable. What fires is a *change* in the pattern — which is much closer
to what you actually mean by "hazard" than "there is a car here".

---

## Results

KITTI raw drive `2011_09_26_drive_0009`, frames 200–408 at stride 2, 103 frames
scored. Ground truth is KITTI tracklets restricted to objects whose motion
cannot be explained by ego motion — 9 of the drive's 98 tracklets, 58 visible
instances against 562 parked ones. The detector never sees any of it.

### Is the signal real?

The first thing to rule out is that this is just noise from VGGT's depth head.
The control: **a parked car and a moving car are the same object** — same paint,
same glass, same silhouette, same depth discontinuities. Depth noise cannot tell
them apart. So read the raw surprise map inside every annotated box, with no
threshold, no blobs and no tracking:

| | n | mean score |
|---|---|---|
| **Moving** vehicles | 57 | **7.100** |
| **Parked** vehicles | 533 | 1.253 |
| Random size-matched boxes | 590 | 0.764 |

- **AUC moving vs parked: 0.813** (0.500 would be noise)
- AUC parked vs random: **0.593** — being a car barely lifts the score; *moving* does
- Range-matched ratios: **8.9× / 9.2× / 3.2× / 6.3×** at 0–15 / 15–25 / 25–40 / 40+ m

An object detector separates moving from parked at AUC 0.500 by construction:
same pixels, same answer. This is the one thing here that a detector cannot do.

### The forecast is what produces it

Both forecasters at the same threshold, so recall is matched exactly:

| | **rigid** (geometry forecast) | **identity** (ablation) |
|---|---|---|
| recall, moving objects | 34.5 % | 34.5 % |
| **precision** | **34.3 %** | 5.4 % |
| **false-flag rate on parked vehicles** | **2.8 %** | 28.4 % |
| detections emitted | 67 | 351 |

`identity` predicts nothing changes, so every parked car sliding past the camera
looks like a hazard. **At identical recall the geometry forecast is 6.4× more
precise and flags parked cars 10× less often.**

### Fixing the decision layer

The first version of the detector wasted most of that signal. Two changes, and
neither works without the other:

| configuration | recall | precision | parked FP | F1 |
|---|---|---|---|---|
| original — IoU tracker, threshold 4 | 27.6 % | 26.5 % | 4.6 % | 0.270 |
| + centroid-gated tracker only | 46.6 % | 19.0 % | 14.6 % | 0.270 |
| + threshold 8 only, old tracker | 13.8 % | 42.1 % | 0.7 % | 0.208 |
| **both** | **34.5 %** | **34.3 %** | **2.8 %** | **0.344** |

Pure IoU association fails on objects that move further than their own width
between frames — the fastest, most hazardous ones — so the persistence filter
deleted exactly what mattered. Fixing that alone just slides along the curve
(same F1). Raising the threshold alone starves recall. Together they are a strict
Pareto improvement on all three axes, **+27 % relative F1**.

### An idea that did not work

Since box-mean surprise separates moving from parked at AUC 0.813, scoring
sliding windows that way should make a better detector. It does not: multi-scale
window scoring tops out at **F1 0.206 against the blob chain's 0.344**, worse at
every matched false-alarm rate.

The flaw is that the AUC was measured on *ground-truth* boxes. Given the box,
the mean inside it discriminates well; but a detector has to find the box, and
box-mean is a bad objective for that — a mispredicted car is a compact intense
patch, and averaging over a car-sized window divides it by an area that is
mostly correct. Connected components adapt to the shape of the surprise, which
matters more than scoring the region well. Kept in the tree as
`--detector window` so the result is reproducible.

![top moments](outputs/final/top_moments.png)

At the junction (raw frames 370–388) the cars crossing the intersection get
boxed and the parked car at the kerb does not. The remaining false positives are
mostly road surface in the bottom corners, where forward warping has the most
extreme parallax.

---

## Quick start

```bash
python scripts/selftest_geometry.py      # ~1 s, no weights needed
python scripts/fetch_weights.py          # 5 GB, VGGT-1B from Hugging Face
python run.py --frames data/demo_seq --stride 1 --chunk 4 --out outputs/smoke
```

`selftest_geometry.py` builds a synthetic street with ray-traced exact depth and
checks two things: a static scene stays quiet (0.00 % of pixels flagged), and a
box sliding into the lane lights up 91× brighter than its background. Run it
first — if it fails, nothing downstream is worth debugging.

On a real drive:

```bash
python scripts/fetch_kitti.py --drive 0009               # 1.8 GB + tracklets
python run.py --frames data/2011_09_26/2011_09_26_drive_0009_sync \
              --start 200 --count 105 --stride 2 --chunk 8 \
              --out outputs/kitti_busy
python scripts/validate.py --run outputs/kitti_busy \
              --drive data/2011_09_26/2011_09_26_drive_0009_sync \
              --calib data/2011_09_26
```

Outputs land in `outputs/<name>/`: `top_moments.png` (contact sheet of the most
surprising moments), `surprise.mp4`, `panels/` (4-up diagnostics), `overlays/`,
`frame_scores.csv`, `blobs.json`, `summary.json`.

---

## How it works

```
frames ──► VGGT ──► depth + confidence + camera pose   (the measuring instrument)
                        │
                        ├─► forecaster: warp frame t-1's depth through the
                        │   extrapolated camera pose        (uses only the past)
                        │
                        └─► observed depth at frame t
                                │
              surprise = robust-z( |forecast − observed| / observed )
                                │
                     gate ─► threshold ─► blobs ─► persistence ─► ranked hazards
```

Three decisions carry most of the weight.

**Chunking.** VGGT's global attention runs across every frame in a window, so
depth and pose are only comparable *inside one forward pass* — run it twice and
you get two different arbitrary scales. So we push a chunk of 8 frames through
once and harvest 6 forecasts from it, each predicting one frame from its own two
predecessors. That is roughly a 6× saving over per-frame windows, and it makes
scale consistency free instead of something to correct afterwards. The forecast
still only ever looks backwards; VGGT is the instrument, not the predictor.

**Robust normalisation.** The raw residual is large everywhere the car hits a
bump or the sun moves. All of that is *global* error. Expressing each pixel as
"how many MADs above this frame's own typical error" divides it out: a jolt
raises the median for every pixel at once and cancels, a pedestrian does not.

**Depth-edge suppression.** The single biggest source of false alarms, and it is
an artifact of forward warping rather than anything in the scene. Where depth
jumps from 8 m to 40 m across one pixel, being off by a single pixel is a 400 %
relative error that means nothing. Left alone it draws a bright outline around
every object in the frame. Masking the morphological gradient of *log* depth
took the synthetic static scene from 2.81 % of pixels flagged to 0.00 %, while
leaving object interiors — where a genuinely mispredicted car lights up —
untouched.

---

## Running the paper's learned forecaster

The `rigid` forecaster is the baseline any learned forecaster has to beat, and
it needs no weights beyond VGGT itself. To swap in VGGT-World's flow-matching
model:

1. Download `kitti_checkpoint.pt` from the OneDrive folder linked in
   [the upstream README](https://github.com/SimonSun0810/VGGT-World). It is a
   *folder share* that needs an interactive login, so no script can fetch it.
2. `pip install -r requirements-vggtworld.txt` (hydra, diffusers, einops).
3. `python run.py --forecaster vggtworld --vggtworld-ckpt weights/kitti_checkpoint.pt ...`

That path is wired against the released code but is **unverified** — I could not
obtain the checkpoint. Everything else in this repo was run end to end.

---

## Apple Silicon notes

Upstream VGGT-World is Linux + CUDA only. `scripts/patch_for_mps.py` fixes the
model code and is idempotent, so you can point it at a fresh clone. What it
found:

| where | problem |
|---|---|
| `fm.py:228,230` | `.to("cuda")` on schedule buffers **inside `__init__`** — the model will not even construct |
| `fm.py:451` | `torch.autocast("cuda", ...)` around the sampling loop |
| `aggregator.py` ×3 | `torch.cuda.synchronize()` **inside the per-block loop** of `forward`, `part1`, `part2` |
| `aggregator.py` ×6 | `device="cuda"` and `.to("cuda")` for rotary embeddings — plain **string literals**, so they survive any grep for `torch.cuda` or `.cuda()` |
| `vggt.py:223` | `torch.cuda.amp.autocast(enabled=False)` — warns rather than raises, but on every frame |

Two more things that are not in the patcher because they are not model bugs:

- **All four `eval/*.py` scripts are broken as shipped.** They call
  `compose(config_name="default")` when no `default.yaml` exists, and pass an
  *absolute* path to Hydra's `initialize()`, which requires a relative one. The
  demo gets both right. Build on the demo.
- **The VGGT camera head must run in fp32.** It refines pose over four
  iterations, and in bf16 that loop diverges to NaN — silently, because the
  tokens going in are perfectly finite. The symptom is a NaN focal length and an
  empty surprise map three stages downstream. `backbone.py` casts explicitly.

Memory guards (`hazard/device.py`, applied before torch imports) cap the MPS
allocator at 0.8 of the recommended working set so PyTorch raises a clean OOM
instead of letting macOS swap and beachball. Note both watermarks must move
together — lowering only the high one throws `invalid low watermark ratio`.

Measured on the M5 Air: 3.3 s per 8-frame VGGT pass, **0.59 s per scored frame**,
2.8 GB peak against a 12.7 GB budget. Nothing needed to be shrunk.

---

## Limits

- **Blind to motion along the viewing ray.** A car directly ahead at a similar
  speed produces almost no depth residual. The method sees crossing traffic far
  better than following traffic — visible in the range table and in which
  junction frames score highest.
- **Range-limited.** Below ~600 px of object area, recall is 0 %.
- **Precision is 21.5 %.** Of 93 reported blobs, 45 sat on nothing KITTI
  annotates. Some are thin vertical structures (poles, fence posts) where the
  warp is genuinely unreliable; some are real motion outside KITTI's annotation
  range. Separating those two is the obvious next piece of work.
- **The ground truth is derived, not given.** KITTI does not label
  moving-vs-parked, so `hazard/kitti_labels.py` fits a 2D rigid ego-motion field
  per frame and calls the residual motion. Fitting a *translation* instead — the
  obvious first attempt — mislabels every distant parked car as moving through
  every turn, because ego rotation induces apparent velocity that grows with
  range. That bug cost 10 points of apparent recall before it was found.

## Layout

```
run.py                     CLI
hazard/
  device.py                MPS selection + memory guards
  backbone.py              VGGT trunk + camera/depth heads (no point/track head)
  forecast.py              rigid / identity / vggtworld forecasters
  surprise.py              residual -> robust z-score, gating, edge suppression
  detect.py                blobs + IoU tracking + persistence
  viz.py                   overlays, panels, contact sheet, video
  pipeline.py              chunked driver
  kitti_labels.py          tracklets + rigid ego-motion moving/parked split
  kitti_calib.py           velodyne -> cropped image projection
scripts/
  selftest_geometry.py     synthetic ray-traced validation, no weights
  fetch_weights.py         VGGT-1B
  fetch_kitti.py           drive + tracklets + calibration
  patch_for_mps.py         the Apple Silicon fixes
  validate.py              score against tracklets
vggt/                      vendored from VGGT-World, MPS-patched
```

Built on [VGGT](https://github.com/facebookresearch/vggt) and
[VGGT-World](https://github.com/SimonSun0810/VGGT-World); vendored code keeps
its upstream licence.
