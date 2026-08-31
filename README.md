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

Two things to read first, because they change how the rest should be taken.

**Everything was tuned on one drive, and the held-out numbers are much worse.**
Drive 0009's junction gives 48.3 % recall. The same frozen configuration, run
once on two drives downloaded afterwards, gives **10.5 %**. That gap is the
headline finding of this project.

**Recall is governed by tangential motion and apparent size, not by a single
number.** Any scalar recall figure here is a property of the traffic in the clip
as much as of the method.

### Held-out test — frozen config, run once

Everything (noise model, blur, threshold 8, tracker gate, blob rules) was chosen
on drive 0009. Drives 0013 and 0014 were fetched afterwards and
`scripts/test_once.py` is the only thing ever run on them. No knob was touched
after seeing this.

Confidence intervals bootstrap over **tracklets**, not instances — one moving car
contributes ~20 correlated frames, so resampling instances would give intervals
several times too narrow.

| method | instances | tracklets | recall | 95 % CI | precision | parked FP |
|---|---|---|---|---|---|---|
| **depth residual** | 257 | 31 | **10.5 %** | [3.4, 22.0] | 19.9 % | 13.6 % |
| flow residual | 257 | 31 | 1.6 % | [0.3, 3.8] | 1.6 % | 3.0 % |

Per drive: 0013 gives 22.8 % recall, 0014 gives 7.0 %. The CIs are wide because
31 tracklets is not many — that is the honest width, not a presentation choice.

**Recall by apparent size, pooled held-out:**

| box area | n | recall | 95 % CI |
|---|---|---|---|
| < 600 px² | 135 | **0.0 %** | [0.0, 0.0] |
| 600–2000 px² | 77 | 5.2 % | [0.0, 11.3] |
| > 2000 px² | 45 | **51.1 %** | [29.2, 69.6] |

Large objects generalise. Everything below ~2000 px² — which is more than half
of all annotated moving vehicles — is essentially invisible. Drive 0009 looked
good largely because its junction traffic is close, large and crossing.

### The baseline that decides whether VGGT earns its place

`identity` was a straw man. The real question is whether a 1.3 B-parameter
transformer beats Farnebäck plus a RANSAC fundamental matrix — dense flow, fit
the ego-motion a rigid scene would produce, flag the Sampson error that does not
fit. No depth, no weights. Both methods share the same normalisation, blob
extractor, tracker and threshold grid, so only the residual space differs.

Drive 0009 junction, as PR curves rather than operating points:

| method | AP | best F1 | max recall | R @ FP≤5 % | prec @ R≥30 % |
|---|---|---|---|---|---|
| **depth residual** | **0.186** | **0.374** | 51.7 % | 37.9 % | 39.6 % |
| flow residual | 0.021 | 0.114 | 31.0 % | 8.6 % | 5.4 % |

And at the raw-signal level, no detector involved — mean surprise inside
annotated boxes, moving vs parked:

| residual space | AUC |
|---|---|
| **depth** | **0.838** |
| flow, range-normalised | 0.675 |
| flow, raw Sampson | 0.627 |

Depth wins in the signal, not just in my tuning of the detector. The range
normalisation is a fairness fix: the depth residual is relative (|ΔD|/D) and so
scale-aware, while raw Sampson error is in absolute pixels; flow magnitude goes
as 1/depth for a forward-moving camera, so dividing by it recovers the same
normalisation without using depth. It helps the baseline's box-level AUC and
*hurts* it as a detector (AP 0.005), so the stronger raw variant is reported
above.

**Caveat that matters:** Farnebäck is a weak flow estimator. RAFT would likely
narrow this gap and has not been tried. The claim is "depth residual beats
classical flow residual", not "beats the best possible flow method".

### Is the signal real, or is it depth-head noise?

The control: **a parked car and a moving car are the same object** — same paint,
same glass, same silhouette, same depth discontinuities. Depth noise cannot tell
them apart. Raw surprise inside every annotated box, no threshold, no blobs:

| | n | mean score |
|---|---|---|
| **Moving** vehicles | 57 | **7.100** |
| **Parked** vehicles | 533 | 1.253 |
| Random size-matched boxes | 590 | 0.764 |

- **AUC moving vs parked: 0.813** (0.500 would be noise)
- AUC parked vs random: **0.593** — being a car barely lifts the score; *moving* does
- Range-matched ratios: **8.9× / 9.2× / 3.2× / 6.3×** at 0–15 / 15–25 / 25–40 / 40+ m

An object detector separates moving from parked at AUC 0.500 *by construction* —
same pixels, same answer. That is the one thing here a detector cannot do.

### The forecast is what produces it

Both forecasters at the same threshold, so recall is matched exactly:

| | **rigid** (geometry forecast) | **identity** (ablation) |
|---|---|---|
| recall | 34.5 % | 34.5 % |
| **precision** | **34.3 %** | 5.4 % |
| **parked false-flag rate** | **2.8 %** | 28.4 % |
| detections | 67 | 351 |

At identical recall the geometry forecast is 6.4× more precise and flags parked
cars 10× less often.

### What the small-object fixes bought

Drive 0009 junction, cumulative, threshold 8:

| config | recall | precision | parked FP | F1 | <600 | 600–2k | >2k |
|---|---|---|---|---|---|---|---|
| base (hard edge mask) | 36.2 % | 32.0 % | 3.2 % | 0.340 | 0/5 | 6/30 | 15/23 |
| + normalised blur | 41.4 % | 31.8 % | 3.7 % | 0.359 | 0/5 | 8/30 | 16/23 |
| **+ noise model** (shipped) | **48.3 %** | **31.5 %** | **6.8 %** | **0.381** | 0/5 | 7/30 | **21/23** |
| + row-scaled min_area | 69.0 % | 23.8 % | 12.5 % | 0.354 | 1/5 | 16/30 | 23/23 |

Compared as curves rather than points, because these configs emit different
numbers of detections and some of that recall is bought with false alarms:

| config | AP | best F1 | R @ FP≤3 % | R @ FP≤5 % | R @ FP≤8 % |
|---|---|---|---|---|---|
| base | 0.174 | 0.340 | **27.6 %** | 36.2 % | 39.7 % |
| shipped | **0.193** | **0.381** | 17.2 % | **39.7 %** | **48.3 %** |

**Not a uniform win.** The noise model dominates on AP, on best F1 and
everywhere above FP ≥ 5 %, but at very tight false-alarm rates the old hard mask
is better. Large-object recall goes 65 % → 91 %.

`--row-scaled-area` reaches the highest recall anything reached (69 %) but costs
precision throughout the useful region, so it is off by default.

---

## Negative results

Three, all measured, all reproducible. They cost real compute and they are the
most informative part of the project.

### 1. The small-object floor is in the signal, not the decision layer

The premise was that masking and blurring destroyed small objects before the
detector saw them. Partly true — normalised blur and the noise model bought 12
points of recall on drive 0009. But the calibration segment (frames 0–196,
16 moving instances, median 422 px², median range 42.6 m) yields **0 % recall at
every configuration and every threshold from 4 to 25**, and the pooled held-out
< 600 px² bucket is **0/135**.

So the floor is not fixable downstream. Box-level AUC on that segment is
**0.476** — below chance. At ~40 m a vehicle's inter-frame depth change is inside
VGGT's own depth noise, and no amount of thresholding recovers a signal that is
not there.

### 2. Recall tracks tangential motion, not size alone

Why the calibration segment produces nothing, when junction objects at the same
range are detected fine (AUC 0.828):

| tangential residual speed | n | mean score |
|---|---|---|
| 0.00–0.15 m/frame | 16 | 1.57 |
| 0.15–0.40 | 14 | 2.37 |
| 0.40–0.80 | 27 | **7.08** |
| 0.80+ | 17 | 6.89 |

Calibration segment: tangential fraction **0.14**, mean score 0.69. Junction:
tangential fraction **0.60**, mean score 6.13. Its movers are vehicles ahead on
the same road, moving almost purely radially, and depth-residual forecasting is
structurally blind to that. The flow baseline shares the blind spot — epipolar
lines radiate from the focus of expansion, so radial motion slides along its own
epipolar line.

These are not labelling errors: residual speed is 0.57 m/frame against 0.02 for
parked cars, and the ego-fit residual does not grow with range (0.018 at 0–20 m,
0.024 at 50–80 m).

### 3. Full field of view does not work

The 224×448 centre crop discards 40 % of KITTI's horizontal FOV. Recovering it
(224×742, both exact multiples of the patch size) brings 11 more moving instances
into view, and **objects newly visible in the periphery hit 73.7 % recall** — far
above the 51.7 % average, because vehicles entering from the side are strongly
tangential.

But central precision collapses:

| | moving visible | recall | precision | parked FP |
|---|---|---|---|---|
| 224×448 crop | 58 | 51.7 % | **29.1 %** | 6.2 % |
| 224×742 full FOV | 69 | 44.9 % | **6.9 %** | 25.0 % |

I diagnosed this as unmodelled peripheral parallax and added a displacement term
to the noise model — expected error scaling with how far content actually moved.
**It changed nothing** (51.7/29.1 identical with and without). So that diagnosis
was wrong. The likelier cause is that the robust median/MAD is computed
per-frame, so adding 66 % more high-parallax area inflates the MAD and depresses
z-scores everywhere; that needs per-region normalisation, which is not built.
`--full-fov` exists and is documented as not working.

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

**A noise model, not a mask.** The biggest source of false alarms is depth
discontinuities, and it is an artifact of forward warping rather than anything in
the scene: where depth jumps from 8 m to 40 m across one pixel, being off by a
single pixel is a 400 % relative error that means nothing. The first version
deleted a dilated band around every depth step. That works, and it also deletes
most of a distant vehicle — a car at 40 m keeps only ~124 usable pixels, and a
6 px masked ring takes the rest.

So instead of asking "is this pixel trustworthy", `expected_residual_scale`
predicts *how large a residual to expect here even if nothing moved*: a sub-pixel
registration error δ at a place where depth changes by a fraction g per pixel
produces about δ·g of relative residual, plus a confidence-scaled floor for
VGGT's own depth noise, added in quadrature. Dividing by that gives a quantity
which is ~1 wherever the static world holds — flat road, textured wall, *and* the
silhouette of a parked car. Steep gradients stop being untrustworthy and become
correctly discounted. Only true occlusion boundaries (>100 % depth change per
pixel, undilated) are still dropped.

That is the difference between a heuristic and a statistic, and it is worth 12
points of recall on drive 0009 with large-object recall going 65 % → 91 %. It is
not free: see the curve comparison above, where the old hard mask is still better
at very tight false-alarm rates.

**Normalised smoothing.** Ten lines, and the cheapest win in the repo. The score
is smoothed before thresholding, but masked pixels hold zero, so a naive blur
drags down exactly the pixels adjacent to a mask — which for a small object ringed
by masked pixels means most of it. Blurring `score × valid` and `valid`
separately and dividing fixes it. Worth 5 points of recall at flat precision.

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


**VGGT saw the target frame.** Worth being explicit about, because it sounds like
a leak and is not quite one. Depth and pose come from a single VGGT pass over the
whole 8-frame chunk, and VGGT's global attention means the depth assigned to
frame *t−1* was computed with frame *t* in the window. The *forecast* uses only
past frames — pose extrapolation from t−2 and t−1, warping t−1's depth — but the
depth values it warps were produced by a network that had seen the future frame.
This mirrors the protocol in VGGT-World's own evaluation, which likewise runs
`part1` over the full window to obtain target tokens. VGGT is being used as a
measuring instrument rather than a predictor, so the alternative — a separate pass
per window — would cost 6× the compute and introduce a different arbitrary depth
scale per frame, which is worse. But a strictly causal deployment would need
either a causal backbone or per-window passes with scale alignment, and the
numbers here would change.

**One tuning drive, two held-out drives.** 31 held-out tracklets is not many, and
the bootstrap CIs say so: [3.4, 22.0] % on pooled recall. Drives 0018, 0051, 0056
and 0059 are downloading and would tighten this considerably.

**Farneback, not RAFT.** The flow baseline uses the weakest respectable flow
estimator. A RAFT-based residual would be a stronger opponent and has not been
tried.

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
