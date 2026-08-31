#!/usr/bin/env python3
"""Fetch one KITTI raw drive, its tracklets, and the calibration.

Only `image_02` (the left colour camera) is unpacked -- that is all the detector
reads, and it saves unzipping several GB of velodyne and right-camera data you
will never touch. The zip is deleted afterwards unless you say otherwise.

    python scripts/fetch_kitti.py --drive 0009

Drive 0009 is a city drive with a busy junction around raw frames 335-400,
which is where the interesting results are. Drives 0009, 0013 and 0014 all have
real traffic; residential drives are mostly parked cars and will look quiet.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

BASE = "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data"
DATA = Path(__file__).resolve().parent.parent / "data"
DATE = "2011_09_26"


def curl(url: str, dest: Path) -> bool:
    """curl rather than urllib: HF and S3 both drop long connections, and
    `--retry --continue-at` recovers without starting over."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  fetching {url.rsplit('/', 1)[-1]}")
    r = subprocess.run(
        ["curl", "-L", "--http1.1", "--retry", "8", "--retry-all-errors",
         "--retry-delay", "3", "-C", "-", "--progress-bar", "-o", str(dest), url]
    )
    return r.returncode == 0 and dest.exists()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--drive", default="0009", help="4-digit drive id, e.g. 0009")
    ap.add_argument("--keep-zip", action="store_true")
    args = ap.parse_args()

    name = f"{DATE}_drive_{args.drive}"
    sync_zip = DATA / f"{name}_sync.zip"
    track_zip = DATA / f"{name}_tracklets.zip"
    calib_zip = DATA / f"{DATE}_calib.zip"
    drive_dir = DATA / DATE / f"{name}_sync"

    if (drive_dir / "image_02" / "data").is_dir():
        n = len(list((drive_dir / "image_02" / "data").glob("*.png")))
        print(f"already have {drive_dir} ({n} frames)")
    else:
        if not curl(f"{BASE}/{name}/{name}_sync.zip", sync_zip):
            print("download failed", file=sys.stderr)
            return 1
        print("  unpacking image_02 only")
        with zipfile.ZipFile(sync_zip) as z:
            members = [m for m in z.namelist() if f"{name}_sync/image_02/" in m]
            z.extractall(DATA, members=members)
        if not args.keep_zip:
            sync_zip.unlink()
            print("  removed the zip")

    if not (drive_dir / "tracklet_labels.xml").exists():
        if curl(f"{BASE}/{name}/{name}_tracklets.zip", track_zip):
            with zipfile.ZipFile(track_zip) as z:
                z.extractall(DATA)
            track_zip.unlink()
        else:
            print("  note: no tracklets for this drive; validation will not run")

    if not (DATA / DATE / "calib_cam_to_cam.txt").exists():
        if curl(f"{BASE}/{DATE}_calib.zip", calib_zip):
            with zipfile.ZipFile(calib_zip) as z:
                z.extractall(DATA)
            calib_zip.unlink()

    frames = sorted((drive_dir / "image_02" / "data").glob("*.png"))
    print(f"\nready: {drive_dir}")
    print(f"  {len(frames)} frames")
    print(f"  tracklets:   {(drive_dir / 'tracklet_labels.xml').exists()}")
    print(f"  calibration: {(DATA / DATE / 'calib_cam_to_cam.txt').exists()}")
    print(f"\n  python run.py --frames {drive_dir} --start 200 --count 105 --stride 2")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
