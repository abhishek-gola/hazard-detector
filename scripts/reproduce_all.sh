#!/usr/bin/env bash
# Regenerate every number in the README from scratch.
#
# Roughly 25 min of compute on an M5 Air after the downloads finish, most of it
# the one-time VGGT geometry caching. Everything after that replays from cache.
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-python3}"
DATA=data/2011_09_26
D9=$DATA/2011_09_26_drive_0009_sync

echo "=== 0. sanity: geometry with no network involved ==="
$PY scripts/selftest_geometry.py

echo "=== 1. weights (5 GB, skipped if present) ==="
$PY scripts/fetch_weights.py

echo "=== 2. drives ==="
for d in 0009 0013 0014; do $PY scripts/fetch_kitti.py --drive "$d"; done

echo "=== 3. cache VGGT geometry (the only expensive step) ==="
$PY scripts/cache_geometry.py --frames $D9 --start 200 --count 105 --out cache/junction
$PY scripts/cache_geometry.py --frames $D9 --start 0   --count 99  --out cache/calib
$PY scripts/cache_geometry.py --frames $D9 --start 200 --count 105 \
    --target-h 224 --target-w 742 --out cache/junction_fov
$PY scripts/cache_geometry.py --frames $DATA/2011_09_26_drive_0013_sync \
    --start 6 --count 58  --out cache/d0013
$PY scripts/cache_geometry.py --frames $DATA/2011_09_26_drive_0014_sync \
    --start 0 --count 108 --out cache/d0014

echo "=== 4. shipped run: outputs/tier1 (pictures, video, blobs.json) ==="
$PY run.py --frames $D9 --start 200 --stride 2 --count 105 --chunk 8 --out outputs/tier1

echo "=== 5. control: is the signal real (AUC 0.813) ==="
$PY scripts/control_test.py --run outputs/kitti_busy --drive $D9 --calib $DATA \
  || echo "  (needs a --save-maps run; see README)"

echo "=== 6. Tier-1 ablation, both segments ==="
$PY scripts/ablate_tier1.py --cache cache --drive $D9 --calib-dir $DATA

echo "=== 7. depth vs classical flow baseline ==="
$PY scripts/compare_baselines.py --cache cache/junction --drive $D9 --calib-dir $DATA

echo "=== 8. HELD-OUT TEST, frozen config, run once ==="
$PY scripts/test_once.py --drives 0013,0014 --calib-dir $DATA

echo
echo "done. README tables correspond to steps 5-8."
