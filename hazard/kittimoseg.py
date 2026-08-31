"""KittiMoSeg / InstanceMotSeg labels: the ground truth that already existed.

Reader for the extended KittiMoSeg release (Siam et al.'s MODNet dataset, grown
~10x by Rashed to 12,919 frames; the instance-level version is InstanceMotSeg).
This replaces the self-derived tracklet labels in `kitti_labels.py`, which
reimplemented this dataset's own construction method.

Two things it buys that the derived labels cannot:

  1. **mIoU.** Published work on this task reports mean IoU over pixel motion
     masks. `binary_img_Output/*.png` is exactly that mask, so results here
     become comparable to the literature instead of to nothing.
  2. **38 drives** instead of the 7 with usable tracklets.

Two things it does not buy. It does not escape a single recording session --
every archive in the release is `2011_09_26`, the same afternoon KITTI publishes
tracklets for -- and **it ships no train/test split**. The only documentation is
`Data_representation.docx`, which covers the file format and nothing else. So
MODNet's Table II figures cannot be matched split-for-split from this release;
its numbers are on the original, smaller KITTI MOD (six raw sequences plus KITTI
scene flow), which is not what is distributed here.

Verified on load: 38 drives, **12,919 labelled frames**, exactly the count the
release claims.

Layout inside each per-drive `.7z`:

    binary_img_Output/NNNNNNNNNN.png     union of moving-object masks, 0/255
    np_array_moving_Output/*.npz         per-instance moving masks, (H, W, k)
    np_array_All_classes_Output/*.npz    one-hot over 7 classes, (H, W, 7)
    Text_data_Output/NNNNNNNNNN.txt      one line per object:
                                           instance_id moving y0 x0 y1 x1 class
                                         (field order per Data_representation.docx;
                                          the y0 x0 y1 x1 order was additionally
                                          confirmed against binary_img_Output)
    rgb_img_Output/*.jpg                 the frame, for reference

Everything is at 750x2484 -- exactly 2x KITTI's native 375x1242 -- so every
coordinate is halved on the way in.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

SCALE = 2.0  # annotations are at 2x KITTI native resolution
NATIVE_HW = (375, 1242)

# Class ids, verbatim from the release's own Data_representation.docx:
#   1 Car  2 Van  3 Truck  4 Pedestrian  5 Sitter  6 Cyclist  7 Tram  8 Misc
# An earlier version of this file guessed at this mapping and had classes 2-8
# wrong. It did not affect any box or mask metric, since nothing here filters by
# class, but it would have mislabelled every per-class breakdown.
CLASS_NAMES = {1: "Car", 2: "Van", 3: "Truck", 4: "Pedestrian",
               5: "Sitter", 6: "Cyclist", 7: "Tram", 8: "Misc"}

WANT_DIRS = ("binary_img_Output", "Text_data_Output")


@dataclass
class MoSegObject:
    track_id: int
    moving: bool
    box_native: tuple[float, float, float, float]  # x, y, w, h at 375x1242
    class_id: int

    @property
    def class_name(self) -> str:
        return CLASS_NAMES.get(self.class_id, f"class{self.class_id}")


def extract_drive(archive: str | Path, out_dir: str | Path,
                  dirs: tuple[str, ...] = WANT_DIRS) -> int:
    """Pull just the label directories out of a per-drive .7z.

    The full archive is ~300 MB per drive because it carries RGB copies and
    one-hot class tensors we do not need; the masks plus text are a fraction of
    that.
    """
    import py7zr

    archive, out_dir = Path(archive), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with py7zr.SevenZipFile(archive) as z:
        targets = [n for n in z.getnames()
                   if any(n.startswith(d + "/") for d in dirs)]
    with py7zr.SevenZipFile(archive) as z:
        z.extract(path=out_dir, targets=targets)
    return len(targets)


def frames_available(label_dir: str | Path) -> list[int]:
    d = Path(label_dir) / "Text_data_Output"
    if not d.is_dir():
        return []
    return sorted(int(p.stem) for p in d.glob("*.txt"))


def load_objects(label_dir: str | Path, frame: int) -> list[MoSegObject]:
    """Parse one frame's object list. Boxes returned at KITTI native scale."""
    p = Path(label_dir) / "Text_data_Output" / f"{frame:010d}.txt"
    if not p.exists():
        return []
    out: list[MoSegObject] = []
    for line in p.read_text().strip().splitlines():
        f = line.split()
        if len(f) < 7:
            continue
        tid, moving, y0, x0, y1, x1, cls = (float(v) for v in f[:7])
        # Some boxes run off-frame with negative or oversized coordinates;
        # clip in native space rather than dropping them, since a partially
        # visible vehicle is still a vehicle.
        x0, x1 = sorted((x0 / SCALE, x1 / SCALE))
        y0, y1 = sorted((y0 / SCALE, y1 / SCALE))
        x0, x1 = max(0.0, x0), min(float(NATIVE_HW[1]), x1)
        y0, y1 = max(0.0, y0), min(float(NATIVE_HW[0]), y1)
        if x1 - x0 < 1 or y1 - y0 < 1:
            continue
        out.append(MoSegObject(track_id=int(tid), moving=bool(int(moving)),
                               box_native=(x0, y0, x1 - x0, y1 - y0),
                               class_id=int(cls)))
    return out


def load_mask_native(label_dir: str | Path, frame: int) -> np.ndarray | None:
    """Union of moving-object masks, downsampled to KITTI native resolution."""
    p = Path(label_dir) / "binary_img_Output" / f"{frame:010d}.png"
    if not p.exists():
        return None
    m = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if m is None:
        return None
    if m.ndim == 3:
        m = m[..., 0]
    h, w = NATIVE_HW
    return (cv2.resize(m, (w, h), interpolation=cv2.INTER_AREA) > 127)


def crop_transform(target_h: int = 224, target_w: int = 448):
    """The same resize-then-centre-crop `hazard.data.load_frame` applies."""
    h, w = NATIVE_HW
    scale = max(target_h / h, target_w / w)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    left, top = (new_w - target_w) // 2, (new_h - target_h) // 2
    return scale, left, top, new_w, new_h


def box_to_cropped(box_native, target_h: int = 224, target_w: int = 448):
    """Native box -> cropped-image box, or None if it falls outside."""
    scale, left, top, _, _ = crop_transform(target_h, target_w)
    x, y, w, h = box_native
    x0, y0 = x * scale - left, y * scale - top
    x1, y1 = x0 + w * scale, y0 + h * scale
    x0, y0 = max(0.0, x0), max(0.0, y0)
    x1, y1 = min(float(target_w), x1), min(float(target_h), y1)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    return int(x0), int(y0), int(round(x1 - x0)), int(round(y1 - y0))


def mask_to_cropped(mask_native: np.ndarray, target_h: int = 224,
                    target_w: int = 448) -> np.ndarray:
    """Native mask -> cropped-image mask, matching the frame pipeline exactly."""
    scale, left, top, new_w, new_h = crop_transform(target_h, target_w)
    m = cv2.resize(mask_native.astype(np.uint8), (new_w, new_h),
                   interpolation=cv2.INTER_NEAREST)
    return m[top:top + target_h, left:left + target_w] > 0


def moving_boxes_cropped(label_dir: str | Path, frame: int,
                         target_h: int = 224, target_w: int = 448):
    """(box, moving, class_name, track_id) in cropped-image coordinates.

    The track id matters for evaluation, not just for bookkeeping: one moving
    car contributes ~20 highly correlated frames, so a confidence interval
    bootstrapped over instances would treat 20 views of one event as 20
    independent trials. Resampling over track ids keeps the unit of
    independence honest, and KittiMoSeg supplies real ids so there is no need
    to approximate them by spatial binning.
    """
    out = []
    for o in load_objects(label_dir, frame):
        b = box_to_cropped(o.box_native, target_h, target_w)
        if b:
            out.append((b, o.moving, o.class_name, o.track_id))
    return out
