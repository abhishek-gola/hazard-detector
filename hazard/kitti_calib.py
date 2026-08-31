"""KITTI raw calibration: velodyne 3D boxes -> image_02 pixels.

Needed only for validation. The detector itself never sees a calibration file,
a label, or a class name.

The full chain KITTI uses for the rectified left colour camera is

    x_img = P_rect_02 @ R_rect_00 @ Tr_velo_to_cam @ x_velo

and then the same resize-and-centre-crop the frames went through, so that boxes
land in 224x448 coordinates rather than the native 1242x375.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _read_calib(path: Path) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for line in Path(path).read_text().splitlines():
        if ":" not in line:
            continue
        key, val = line.split(":", 1)
        try:
            out[key.strip()] = np.array([float(x) for x in val.split()])
        except ValueError:
            pass  # calib_time and friends
    return out


class KittiCalib:
    """Everything needed to put a velodyne box on a cropped image."""

    def __init__(self, calib_dir: str | Path):
        calib_dir = Path(calib_dir)
        cam = _read_calib(calib_dir / "calib_cam_to_cam.txt")
        velo = _read_calib(calib_dir / "calib_velo_to_cam.txt")

        self.P2 = cam["P_rect_02"].reshape(3, 4)
        R_rect = np.eye(4)
        R_rect[:3, :3] = cam["R_rect_00"].reshape(3, 3)
        self.R_rect = R_rect

        Tr = np.eye(4)
        Tr[:3, :3] = velo["R"].reshape(3, 3)
        Tr[:3, 3] = velo["T"]
        self.Tr_velo_to_cam = Tr

        self.native_hw = tuple(int(x) for x in cam["S_rect_02"][::-1]) \
            if "S_rect_02" in cam else (375, 1242)

    def velo_to_image(self, pts_velo: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(N, 3) velodyne -> (N, 2) native pixels, plus a depth-positive mask."""
        hom = np.concatenate([pts_velo, np.ones((len(pts_velo), 1))], axis=1)
        cam = (self.R_rect @ self.Tr_velo_to_cam @ hom.T)  # (4, N)
        img = self.P2 @ cam  # (3, N)
        z = img[2]
        good = z > 1e-3
        with np.errstate(divide="ignore", invalid="ignore"):
            uv = (img[:2] / z).T
        return uv, good

    def crop_transform(self, target_h: int = 224, target_w: int = 448):
        """The exact resize+centre-crop `hazard.data.load_frame` applies."""
        h, w = self.native_hw
        scale = max(target_h / h, target_w / w)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        left, top = (new_w - target_w) // 2, (new_h - target_h) // 2
        return scale, left, top

    def velo_to_cropped(
        self, pts_velo: np.ndarray, target_h: int = 224, target_w: int = 448
    ) -> tuple[np.ndarray, np.ndarray]:
        uv, good = self.velo_to_image(pts_velo)
        scale, left, top = self.crop_transform(target_h, target_w)
        uv = uv * scale - np.array([left, top])
        return uv, good


def box_corners(center: np.ndarray, h: float, w: float, l: float, ry: float) -> np.ndarray:
    """Eight corners of a KITTI tracklet box, in velodyne coordinates.

    Tracklet origin sits on the ground at the box centre, z pointing up, so the
    box spans [0, h] in z rather than being centred on it.
    """
    x = np.array([l, l, -l, -l, l, l, -l, -l]) / 2
    y = np.array([w, -w, -w, w, w, -w, -w, w]) / 2
    z = np.array([0, 0, 0, 0, h, h, h, h], dtype=float)
    c, s = np.cos(ry), np.sin(ry)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return (R @ np.stack([x, y, z])).T + center


def project_box(
    calib: KittiCalib,
    center: np.ndarray,
    h: float,
    w: float,
    l: float,
    ry: float,
    target_h: int = 224,
    target_w: int = 448,
) -> tuple[int, int, int, int] | None:
    """2D axis-aligned box in cropped-image pixels, or None if not visible."""
    corners = box_corners(center, h, w, l, ry)
    uv, good = calib.velo_to_cropped(corners, target_h, target_w)
    if good.sum() < 4:
        return None
    uv = uv[good]
    x0, y0 = uv.min(axis=0)
    x1, y1 = uv.max(axis=0)
    # Clip into frame; reject if nothing meaningful is left.
    x0, x1 = max(0, int(x0)), min(target_w, int(np.ceil(x1)))
    y0, y1 = max(0, int(y0)), min(target_h, int(np.ceil(y1)))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    return x0, y0, x1 - x0, y1 - y0
