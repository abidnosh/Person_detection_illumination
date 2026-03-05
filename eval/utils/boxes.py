from __future__ import annotations

import numpy as np


def box_iou_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between sets of boxes in xyxy (absolute)."""
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)

    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0.0, None)
    inter = wh[:, :, 0] * wh[:, :, 1]

    area_a = np.clip(a[:, 2] - a[:, 0], 0.0, None) * np.clip(a[:, 3] - a[:, 1], 0.0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0.0, None) * np.clip(b[:, 3] - b[:, 1], 0.0, None)
    union = area_a[:, None] + area_b[None, :] - inter

    out = np.zeros_like(inter, dtype=np.float32)
    m = union > 0
    out[m] = inter[m] / union[m]
    return out

def coco_size_bin_xyxy(box: np.ndarray) -> str:
    """COCO area bins: small < 32^2, medium < 96^2, else large."""
    area = float(max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1]))
    if area < 32.0**2:
        return "small"
    if area < 96.0**2:
        return "medium"
    return "large"