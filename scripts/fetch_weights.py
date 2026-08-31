#!/usr/bin/env python3
"""Download the VGGT-1B backbone (5 GB) from Hugging Face.

That single file is everything the `rigid` and `identity` forecasters need --
the detector never trains anything.

The paper's own flow-matching checkpoint is a different story: it is published
through a OneDrive *folder share*, which needs an interactive login and cannot
be fetched from a script. Instructions for that are printed at the end.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
from pathlib import Path

REPO = "facebook/VGGT-1B"
URL = f"https://huggingface.co/{REPO}/resolve/main/model.safetensors?download=true"
DEST = Path(__file__).resolve().parent.parent / "weights" / "vggt_1b.safetensors"
EXPECTED_GB = 5.03

ONEDRIVE = ("https://1drv.ms/f/c/991ebd14386d50f5/"
            "IgAJa17DGykcT6LEAhV89viUAYbdSNHZm-woBHrWxHWZYbk?e=XLHhcc")


def _progress(done: int, total: int) -> None:
    if total <= 0:
        return
    pct = 100 * done / total
    bar = "#" * int(pct / 2.5)
    sys.stdout.write(f"\r  [{bar:<40}] {pct:5.1f}%  {done / 1e9:.2f}/{total / 1e9:.2f} GB")
    sys.stdout.flush()


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "hazard-detector/0.1"})
    with urllib.request.urlopen(req) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        done = 0
        with open(part, "wb") as fh:
            while chunk := resp.read(1 << 20):
                fh.write(chunk)
                done += len(chunk)
                _progress(done, total)
    print()
    shutil.move(part, dest)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="re-download even if present")
    args = ap.parse_args()

    if DEST.exists() and not args.force:
        gb = DEST.stat().st_size / 1e9
        print(f"already have {DEST} ({gb:.2f} GB)")
    else:
        print(f"downloading {REPO} -> {DEST}  (~{EXPECTED_GB} GB)")
        try:
            download(URL, DEST)
        except Exception as exc:
            print(f"\ndownload failed: {exc}", file=sys.stderr)
            print("\nfall back to curl, which resumes cleanly:", file=sys.stderr)
            print(f'  curl -L --http1.1 --retry 8 -C - -o "{DEST}" \\', file=sys.stderr)
            print(f'    "{URL}"', file=sys.stderr)
            return 1
        print(f"done: {DEST.stat().st_size / 1e9:.2f} GB")

    print(f"""
The learned forecaster from the paper is optional and must be fetched by hand:

  1. open  {ONEDRIVE}
  2. download  kitti_checkpoint.pt
  3. put it in  weights/
  4. pip install -r requirements-vggtworld.txt
  5. run with  --forecaster vggtworld --vggtworld-ckpt weights/kitti_checkpoint.pt

Everything else works without it.""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
