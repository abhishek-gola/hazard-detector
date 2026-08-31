#!/usr/bin/env python3
"""Fetch and extract the KittiMoSeg ground truth — the primary labels.

The extended KittiMoSeg release (Rashed's ~10x of Siam et al.'s MODNet dataset;
InstanceMotSeg is the instance-level version) lives in a public Google Drive
folder as 38 per-drive `.7z` archives, one per `2011_09_26` drive, ~8.4 GB total.

Only two directories per archive are unpacked -- the pixel motion masks and the
per-object text -- which is ~2 MB per drive instead of ~300 MB. The RGB copies
and one-hot class tensors are not used: frames come from KITTI raw via
`fetch_kitti.py`.

Needs gdown and py7zr, which should go somewhere they cannot disturb the runtime
environment:

    pip install --target .tools -r requirements-moseg.txt
    PYTHONPATH=.tools python scripts/fetch_moseg.py --drives 0009,0013,0014

    # or everything the release covers (8.4 GB download, ~1 h of extraction)
    PYTHONPATH=.tools python scripts/fetch_moseg.py --all

Manual alternative, if gdown will not authenticate: open
https://drive.google.com/drive/folders/1lGdLsoHHTYfLOex8Mci85EQNy74k39rq
download `2011_09_26_drive_NNNN_sync_output_Data.7z` into `data/kittimoseg/`,
then re-run this script -- it skips anything already downloaded.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
ARCHIVES = ROOT / "data" / "kittimoseg"
LABELS = ROOT / "data" / "moseg"
FOLDER = "https://drive.google.com/drive/folders/1lGdLsoHHTYfLOex8Mci85EQNy74k39rq"

# Every drive the release covers. All are 2011_09_26 -- the same recording
# session KITTI publishes tracklets for, which is a property of the benchmark
# rather than a choice made here.
ALL_DRIVES = [
    "0001", "0002", "0005", "0009", "0011", "0013", "0014", "0015", "0017",
    "0018", "0019", "0020", "0022", "0023", "0027", "0028", "0029", "0032",
    "0035", "0036", "0039", "0046", "0048", "0051", "0052", "0056", "0057",
    "0059", "0060", "0061", "0064", "0070", "0079", "0084", "0086", "0087",
    "0091", "0093",
]


def archive_for(drive: str) -> Path | None:
    """The release is inconsistent: drive 0070 is '_syn', everything else '_sync'."""
    for stem in (f"2011_09_26_drive_{drive}_sync_output_Data.7z",
                 f"2011_09_26_drive_{drive}_syn_output_Data.7z"):
        p = ARCHIVES / stem
        if p.exists():
            return p
    return None


def download_folder() -> bool:
    """Pull the whole Drive folder. gdown has no per-file selection for folders."""
    ARCHIVES.mkdir(parents=True, exist_ok=True)
    print(f"downloading the KittiMoSeg folder into {ARCHIVES} (~8.4 GB)")
    r = subprocess.run([sys.executable, "-m", "gdown", "--folder", FOLDER,
                        "-O", str(ARCHIVES)])
    return r.returncode == 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--drives", default="0009,0013,0014,0018,0051,0056,0059",
                    help="comma-separated drive ids to extract")
    ap.add_argument("--all", action="store_true", help="every drive in the release")
    ap.add_argument("--skip-download", action="store_true",
                    help="extract from archives already in data/kittimoseg/")
    args = ap.parse_args()

    try:
        import py7zr  # noqa: F401
    except ImportError:
        print("py7zr missing. pip install --target .tools -r requirements-moseg.txt\n"
              "then re-run with PYTHONPATH=.tools", file=sys.stderr)
        return 1

    drives = ALL_DRIVES if args.all else [d.strip() for d in args.drives.split(",")]

    if not args.skip_download and any(archive_for(d) is None for d in drives):
        try:
            import gdown  # noqa: F401
        except ImportError:
            print("gdown missing; see --skip-download or the manual path in the "
                  "docstring", file=sys.stderr)
            return 1
        if not download_folder():
            print("download failed; see the manual path in the docstring",
                  file=sys.stderr)
            return 1

    from hazard.kittimoseg import extract_drive, frames_available

    ok = 0
    for d in drives:
        out = LABELS / d
        if (out / "Text_data_Output").is_dir():
            print(f"  {d}: already extracted ({len(frames_available(out))} frames)")
            ok += 1
            continue
        arc = archive_for(d)
        if arc is None:
            print(f"  {d}: no archive found, skipped")
            continue
        n = extract_drive(arc, out)
        print(f"  {d}: extracted {n} label files "
              f"({len(frames_available(out))} frames)", flush=True)
        ok += 1

    print(f"\n{ok}/{len(drives)} drives ready in {LABELS}")
    print("labels are ~2 MB per drive; the archives can be deleted afterwards")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
