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

## Ground truth: KittiMoSeg, and what the self-derived labels got wrong

A labelled dataset for exactly this task already existed and I should have used
it first. KittiMoSeg ([Siam et al., MODNet,
arXiv:1709.04821](https://arxiv.org/abs/1709.04821)) provides moving/static
ground truth on KITTI raw — 1,300 frames originally, extended ~10× by Hazem
Rashed to **12,919 frames**;
[KITTI InstanceMotSeg](https://arxiv.org/abs/2008.07008) adds instance-level
motion over 12.9 K samples. Both at
[sites.google.com/view/instancemotseg](https://sites.google.com/view/instancemotseg/).

It is now downloaded and wired in (`hazard/kittimoseg.py`,
`scripts/eval_moseg.py`), and the primary results below are scored against it.
The release is 38 per-drive archives carrying pixel motion masks
(`binary_img_Output`), per-instance masks, and one text line per object
(`track_id moving y0 x0 y1 x1 class`) at 2× KITTI native resolution.

Worth recording: **every archive in the release is `2011_09_26`** — the same
afternoon KITTI publishes tracklets for. The single-recording-session limitation
is a property of the field's standard benchmark for this task, not only of the
shortcut taken here. What KittiMoSeg does buy is 38 drives instead of 7, real
motion labels, and mIoU.

### Was the self-derived ground truth any good?

*Produced by `scripts/eval_moseg.py --drives 0009,0013,0014,0018,0051,0056,0059`, section 1.*

`hazard/kitti_labels.py` decides which tracklets move by fitting a 2D rigid
velocity field to the tracklets themselves and refitting on the quietest 65 %.
KittiMoSeg decides the same thing from GPS/IMU odometry. Matching their object
sets by IoU across seven drives, **3,702 matched objects**:

| | |
|---|---|
| motion-flag agreement | 75.3 % |
| derived "moving" that KittiMoSeg also calls moving (**precision**) | **93.6 %** (1192/1274) |
| KittiMoSeg "moving" that the derived GT finds (**recall**) | **58.9 %** (1192/2024) |

**The rigid-field fit was sound; the threshold was too conservative.** When it
says an object moves it is right 93.6 % of the time, so substituting a rigid-field
fit for GPS/IMU odometry works. But it finds only 58.9 % of moving objects,
because `speed_thresh = 0.45 m/frame` (~16 km/h) was chosen to make labels
trustworthy — the docstring says "precision matters more than coverage" — which
silently discarded slower movers.

**That biased earlier results in two directions at once**, and both are corrected
below:

- Recall was measured against a label set holding only the faster ~59 % of movers,
  which are the easier ones.
- Every detection on one of the missing 41 % counted as a false alarm, and
  specifically as *flagging a parked car*. The 16.9 % "parked false-flag rate"
  was largely my labels calling moving cars parked. Against KittiMoSeg it is
  **4.9 %**.

This is also what the false-positive contact sheet was showing: 56 % of unmatched
detections had a visible vehicle in them. They were real movers the labels missed.

---

## Results

Two things to read first, because they change how the rest should be taken.

**Everything was tuned on one drive, and the held-out numbers are much worse.**
Drive 0009's junction gives 48.3 % recall. The same frozen configuration, scored
against real KittiMoSeg labels across seven drives, gives **28.7 %**. That gap is
the headline finding of this project.

**Recall is governed by apparent size, and per-drive precision ranges 18 % to
83 %.** Any scalar figure here is a property of the traffic in the clip as much as
of the method.

Every table below names the script that produces it, and
`scripts/reproduce_all.sh` regenerates all of them in order.

### Primary results: scored against KittiMoSeg

*Produced by `scripts/eval_moseg.py`, section 2. CIs bootstrap over real
KittiMoSeg track ids.*

Frozen config (noise model, normalised blur, threshold 8, tracker gate, blob
rules), all chosen on drive 0009 before any other drive was fetched. **15 of the
release's 38 drives**, 1,262 frames, **3,138 moving instances over 157 tracks**.

| | 15 drives (current) | 7 drives (earlier) | self-derived labels |
|---|---|---|---|
| moving instances | **3138** | 2024 | 1216 |
| recall | **27.2 %** [22.8, 31.8] | 28.7 % [23.8, 33.6] | 24.7 % |
| precision | **38.7 %** | 43.3 % | 24.5 % |
| static false-flag rate | **4.9 %** | 4.9 % | 16.9 % |
| mask IoU | **10.21 %** [8.09, 12.40] | 12.45 % | — |

Every number drifts slightly down as drives are added, which is what should
happen: the first seven were chosen for traffic density and the later ones were
not. The interval tightens (±5.0 → ±4.5 points) but not dramatically, because
157 tracks is still not many.

### Per-drive spread, now a distribution rather than a range

This is the dominant source of variance in the whole project — larger than any
effect measured anywhere else in this README.

| metric | median | IQR | min | max |
|---|---|---|---|---|
| recall | 26.2 % | [15.3, 35.7] | 0.0 % | 74.1 % |
| precision | 55.3 % | [25.2, 68.5] | 0.0 % | 100.0 % |
| mask IoU | 7.0 % | [3.0, 12.8] | 0.1 % | 24.8 % |

Restricted to the 10 drives with ≥100 instances, which removes the misleading
extremes (precision 100 % is n=15; 9.4 % is n=27):

| metric | median | IQR | min | max |
|---|---|---|---|---|
| recall | 26.4 % | [21.0, 35.5] | 7.0 % | 41.3 % |
| precision | 52.8 % | [30.8, 61.3] | 18.3 % | 82.5 % |
| mask IoU | 12.3 % | [5.2, 14.3] | 1.9 % | 24.8 % |

Two things worth reading off this. **Instance-weighted precision (38.7 %) is far
below the unweighted median (55.3 %)**, because the drives with the most moving
objects are the hardest — so the pooled figure is dominated by difficult clips
and is the more honest headline. And **drive 0001 scores 0.0 % on everything with
88 instances**, which is a total failure rather than a bad clip, and is not yet
diagnosed.

### What actually governs recall: apparent size

*Produced by `scripts/test_once.py --drives 0013,0014,0018,0051,0056,0059` (self-derived labels).*

| box area | n | recall | 95 % CI |
|---|---|---|---|
| < 600 px² | 492 | 9.6 % | [5.2, 15.0] |
| 600–2000 px² | 531 | 23.5 % | [15.8, 32.0] |
| > 2000 px² | 193 | **66.3 %** | [55.7, 74.9] |

Cross-tabulated against tangential motion, to separate two correlated variables:

| size | tangential < 0.25 m/frame | tangential ≥ 0.25 |
|---|---|---|
| < 600 px² | 10.5 % [5.4, 16.6] n=333 | 7.6 % [2.8, 14.2] n=158 |
| 600–2k | 18.3 % [8.9, 33.3] n=278 | 29.2 % [20.7, 37.5] n=250 |
| > 2k | 54.7 % [37.3, 71.4] n=53 | **70.7 %** [60.7, 78.5] n=140 |

**Read down and the intervals separate cleanly; read across and they overlap.**
Apparent size is the dominant driver. Tangential motion has a consistent
positive effect in the two larger size bands but is not separable from noise at
this sample size, and in the smallest band it slightly reverses.

This corrects an earlier claim in this README. On drive 0009 alone the tangential
effect looked dominant — mean box score rose 1.57 → 7.08 across tangential
quartiles, a 4.5× spread — and that was written up as "recall tracks tangential
motion, not size". With 1216 instances across six drives instead of 58 on one,
that does not hold. Size dominates; tangential motion is real but secondary.
The mechanism argument still stands (an object arriving on pixels the forecast
assigned to background produces a much larger residual than one whose depth
merely drifts), it simply is not the largest term.

### Baselines: classical flow, and a modern one

*Produced by `scripts/test_once.py --methods depth,flow,raft` and `scripts/compare_baselines.py`.*

`identity` was a straw man. The real question is whether a 1.16 B-parameter
transformer beats a flow-residual method: compute dense optical flow, fit the
ego-motion a rigid scene would produce with a RANSAC fundamental matrix, and flag
the Sampson error that does not fit. Both share this method's normalisation, blob
extractor, tracker and threshold grid, so only the residual space differs.

Two flow backends, because reporting only the weak one would be choosing the
opponent:

- **Farnebäck** — classical, no weights, pure OpenCV.
- **RAFT-large**, torchvision `C_T_SKHT_K_V2` — Chairs → Things → Sintel +
  **KITTI** + HD1K. A modern learned estimator fine-tuned on this exact domain,
  5.3 M parameters, i.e. **220× smaller than VGGT**.

Pooled over the six held-out drives, 1216 instances:

| method | recall | 95 % CI | precision | parked FP |
|---|---|---|---|---|
| **depth residual** (VGGT, 1.16 B) | **24.7 %** | [18.8, 30.4] | 24.5 % | 16.9 % |
| RAFT-large residual (5.3 M) | 12.1 % | [8.2, 16.7] | **25.8 %** | **9.1 %** |
| Farnebäck residual (0) | 6.7 % | [4.1, 9.9] | 8.6 % | 7.5 % |

And as PR curves on drive 0009:

| method | AP | best F1 | max recall |
|---|---|---|---|
| depth residual | **0.186** | **0.374** | 51.7 % |
| RAFT-large residual | 0.065 | 0.171 | 25.9 % |
| Farnebäck residual | 0.021 | 0.114 | 31.0 % |

**RAFT changes the conclusion, and it should be read carefully.** Depth wins
decisively on recall — 2× RAFT's, with non-overlapping confidence intervals — and
on AP, 2.9×. But **RAFT matches depth on precision (25.8 % vs 24.5 %) and has
almost half the parked false-alarm rate (9.1 % vs 16.9 %)**. So the defensible
claim is narrow: *the depth residual finds roughly twice as many moving objects
as a KITTI-finetuned RAFT residual, at comparable precision, for 220× the
parameters and about 40× the latency.* Whether that trade is worth it depends
entirely on the application.

An earlier version of this README reported only Farnebäck (AP 0.186 vs 0.021,
8.9×). That was choosing a weak opponent, and the honest multiple is 2.9×.

### Is the signal real, or is it depth-head noise?

*Produced by `scripts/control_test.py --run outputs/kitti_busy`.*

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

*Produced by `scripts/ablate_tier1.py --cache cache`.*

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

### False positives, counted rather than guessed

*Produced by `scripts/analyse_fp.py --cache cache/d0051`; labels added by hand from the contact sheet.*

Earlier this README asserted that unmatched detections were "some thin
structures, some real motion KITTI does not label". That was a guess. Here is the
count, from drive 0051 (the richest, 315 detections), with every unmatched crop
rendered to a contact sheet and read by eye —
`outputs/fp_0051/false_positives.png`, labels in the adjacent CSV.

Of 315 detections, 261 overlapped some annotated object; **54 matched nothing**:

| category | n | share |
|---|---|---|
| **a vehicle is visible in the crop** | **30** | **56 %** |
| foliage / tree canopy | 14 | 26 % |
| pole, road sign, billboard | 10 | 19 % |

So only **44 % are genuine false alarms** — thin vertical structures and moving
foliage, both plausible failure modes for a forward warp. The majority contain a
vehicle that the tracklets do not cover, mostly cars in flowing traffic on a busy
road.

**That means precision was understated, and KittiMoSeg later confirmed it.** The
guess at the time was that precision sat between the reported 56.5 % and ~66 % if
those 30 vehicles were moving traffic. Scoring against real labels settled it
from the other direction: the static-object false-flag rate is **3.1 %, not
16.9 %**, because most of what my labels called "flagging a parked car" was
flagging a car they had wrongly labelled parked. The contact sheet was reading
the ground truth's errors, not the detector's.

### The causal asterisk, measured — and it points the other way

*Produced by `scripts/causal_check.py --start 140 --count 60` on drive 0051.*

VGGT's global attention means the depth being warped was computed in a window
containing the target frame. The forecast reads only past frames, and this matches
VGGT-World's own evaluation protocol, but the asterisk needed a number:

- **pass A** — VGGT on [t−2, t−1] only: the forecast's inputs.
- **pass B** — VGGT on [t−1, t]: the observation.
- Independent depth scales, aligned on their shared frame t−1.

58 frames, 12× the VGGT compute. **Compared as curves, because at a fixed
threshold the two protocols emit different numbers of detections** (151 vs 113 at
z = 8) — a single-row comparison reads two different operating points and calls
the difference a protocol effect. An earlier draft of this README made exactly
that mistake and concluded "the penalty is 5.5 points of recall, and precision
improves".

| protocol | AP | best F1 | R @ FP≤20 % | R @ FP≤10 % |
|---|---|---|---|---|
| non-causal (shipped 8-frame chunk) | 0.258 | 0.447 | 23.8 % | **0.0 %** |
| **strictly causal** (2 passes, aligned) | **0.280** | **0.453** | **29.0 %** | **17.9 %** |

**The strictly causal protocol is slightly better, not worse.** AP 0.280 vs
0.258, and at a tight false-alarm budget the non-causal protocol cannot operate
at all while the causal one reaches 17.9 % recall.

The likely mechanism, untested: when the target frame is inside VGGT's window,
global attention partly explains away the moving object — the depth assigned to
t−1 is influenced by where that object is at t, which smears the motion across
frames and shrinks the residual the detector depends on. Removing the target
frame gives a cleaner forecast input.

So the asterisk on every other number in this README is not a penalty that
flatters the method; if anything the shipped protocol is the weaker of the two,
and a causal deployment costs 12× compute rather than accuracy. The scale
alignment turned out to be a non-issue: the ratio between two fully independent
VGGT passes has median 0.9918, IQR 0.972–1.006.

---

## Negative results

Three, all measured, all reproducible. They cost real compute and they are the
most informative part of the project.

### 1. The small-object floor is real, but it is not a floor at zero

The premise was that masking and blurring destroyed small objects before the
detector saw them. Partly true — normalised blur and the noise model bought 12
points of recall on drive 0009, and large-object recall went 65 % → 91 %.

But drive 0009's calibration segment (frames 0–196, 16 moving instances, median
422 px², median range 42.6 m) yields **0 % recall at every configuration and
every threshold from 4 to 25**, with box-level AUC **0.476** — below chance. No
amount of thresholding recovers a signal that is not there.

An earlier version of this section generalised that to "0 out of 135 instances
under 600 px²", from two held-out drives. **That was a small-sample artifact.**
With 492 such instances across six drives the figure is **9.6 % [5.2, 15.0]** —
poor, and far below the 66.3 % for objects over 2000 px², but not zero. The
honest statement is that recall degrades steeply with apparent size and that
particular clips at ~40 m produce nothing.

### 2. Tangential motion is the mechanism but not the dominant term

Depth-residual forecasting should respond to tangential motion, where an object
arrives on pixels the forecast had assigned to background, and be nearly blind to
radial motion, where the object holds its pixels and only its depth drifts. On
drive 0009 that showed up strongly — mean box score by tangential quartile:

| tangential residual speed | n | mean score |
|---|---|---|
| 0.00–0.15 m/frame | 16 | 1.57 |
| 0.15–0.40 | 14 | 2.37 |
| 0.40–0.80 | 27 | **7.08** |
| 0.80+ | 17 | 6.89 |

Its calibration segment has tangential fraction 0.14 and mean score 0.69; its
junction has 0.60 and 6.13. Those movers are vehicles ahead on the same road,
moving almost purely radially, and they produce nothing. Not labelling errors:
residual speed is 0.57 m/frame against 0.02 for parked cars, and the ego-fit
residual does not grow with range (0.018 at 0–20 m, 0.024 at 50–80 m).

**But it does not survive as the primary effect.** Cross-tabulated against size
over 1216 held-out instances, the tangential effect is not separable from noise
within size bands (see the table above), while the size effect is unambiguous.
The mechanism is right; its explanatory power was overstated from a single clip.

The flow baseline shares the blind spot, incidentally — epipolar lines radiate
from the focus of expansion, so radial motion slides along its own epipolar line.
Neither method sees a car pulling away in your lane.

### 3. Full field of view does not work, and I do not know why

*Produced by the local-norm comparison in this section's commit; run via
`compute_surprise(..., local_norm=True)` on `cache/junction_fov`.*

The 224×448 centre crop discards 40 % of KITTI's horizontal field of view.
Recovering it (224×742, both exact multiples of the patch size) brings more
moving objects into view and **objects newly visible in the periphery are found at
73.7 %** — far above average, because vehicles entering from the side are strongly
tangential. But precision collapses.

Scored against KittiMoSeg on drive 0009, swept, with both hypotheses tested:

| config | AP | best F1 | max recall | precision @ best F1 |
|---|---|---|---|---|
| 224×448 crop, global norm | **0.212** | **0.413** | 40.4 % | **45.6 %** |
| 224×448 crop, per-region norm | 0.178 | 0.354 | 52.3 % | 33.3 % |
| 224×742 full FOV, global norm | 0.092 | 0.242 | 45.8 % | 18.9 % |
| 224×742 full FOV, per-region norm | 0.089 | 0.210 | 52.4 % | 15.3 % |

**Two hypotheses, both mine, both wrong.**

*First:* unmodelled peripheral parallax — a fractional pose error over a large
displacement misplaces content proportionally, so the noise model should scale
with how far each pixel actually moved. Implemented (`Forecast.flow`,
`pose_rel_err` in `expected_residual_scale`). **Changed nothing** — identical
numbers with and without at 224×448.

*Second:* the robust median/MAD is computed per frame, so adding 66 % more
high-parallax area inflates it and depresses z-scores everywhere. Fixed by tiling
the frame into a 4×8 grid, computing median/MAD per tile over valid pixels, and
bilinearly upsampling to a smooth field (`local_norm=True`). **Also wrong.** It
does not rescue full FOV (AP 0.092 → 0.089) and it makes the *crop* worse
(0.212 → 0.178). It raises max recall in both cases while wrecking precision,
which is what you would expect if it is mostly amplifying the noise floor of
quiet regions rather than correcting a real spatial bias.

So full FOV fails for a reason I have not identified. Both options ship, both
default off: `--full-fov` and `local_norm=True`. Naming a hypothesis and then
leaving it untested was the weakest thing in this repo; it is now tested and I
was wrong twice.

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

Stated against current results. Every figure here is measured, and several
contradict earlier drafts of this file — those corrections are in the sections
above rather than hidden.

**Not usable as a detector.** 28.7 % recall [23.8, 33.6] at 43.3 % precision,
0.59 s/frame. That is a dataset-mining tool, not a perception component.

**Per-drive variance exceeds every effect measured.** Over 15 drives, precision
has median 52.8 % and IQR [30.8, 61.3] among drives with >=100 instances, and
instance-weighted precision (38.7 %) sits well below the unweighted median
(55.3 %) because the busiest clips are the hardest. Drive 0001 scores 0.0 % on
recall, precision and IoU across 88 instances -- an outright failure, not yet
diagnosed. Clip selection matters more than any parameter in this repo.

**Recall collapses with apparent size.** Objects over 2000 px² are found at
66.3 %; under 600 px², 9.6 % [5.2, 15.0]. More than half of all annotated moving
vehicles are in that lower band. On drive 0009's calibration segment — median
422 px², median range 42.6 m — box-level AUC is 0.476, below chance: at that
range a vehicle's inter-frame depth change is inside VGGT's own depth noise, and
no threshold recovers a signal that is not there.

**Blind to motion along the viewing ray.** A car ahead at a similar speed
produces almost no depth residual. Tangential motion is the mechanism, though
cross-tabulation over 1216 instances shows it is secondary to size and not
separable from noise within size bands. The flow baseline shares this blind spot:
epipolar lines radiate from the focus of expansion.

**mIoU is 3–3.6× below supervised work.** 12.45 % against MODNet's 37.22–45.41 %,
at a quarter of their input resolution and with no training — but also not on
their test split, so indicative only. The gap is structural: the surprise mask
marks the mispredicted part of an object, not its silhouette. It detects better
than it segments.

**RAFT is closer than Farnebäck suggested.** A KITTI-finetuned RAFT-large
residual (5.3 M params, 220× smaller) matches this method's precision and beats
its false-alarm rate at half the recall. The remaining advantage is recall, not
accuracy.

**One recording session, and that is forced.** All seven drives are
`2011_09_26`. KITTI raw publishes tracklets for no other date (verified — every
other date 404s), and **all 38 archives in the KittiMoSeg release are also
`2011_09_26`**. So this is the ceiling of the field's standard benchmark for the
task, not only of the shortcut taken here. Nothing here speaks to weather, night,
rain, or another sensor.

**Seven of 38 available drives.** Labels exist for all 38; only seven have cached
VGGT geometry. 2,024 instances over 108 tracks could be roughly five times that,
which would turn "precision ranges 18–83 %" from seven anecdotes into a
distribution. This is compute, not ideas.

**The ground truth used for six turns of development was 58.9 % complete.** The
derived rigid-field labels are 93.6 % precise but found only 58.9 % of KittiMoSeg's
movers, because `speed_thresh = 0.45 m/frame` was set for label trustworthiness.
That flattered recall and inflated the parked-car false-alarm rate 3.4×. Any
result in this repo predating `scripts/eval_moseg.py` should be read with that in
mind.

**Untested hypotheses, named so they are not mistaken for conclusions.** Full-FOV
input fails and per-frame MAD inflation is the leading explanation — not tested.
Growing the surprise blob into an object mask should close the mIoU gap if the
structural diagnosis is right — not built. Both are the loosest threads here.

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
