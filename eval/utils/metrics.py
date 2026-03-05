from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Tuple

import numpy as np

from .boxes import box_iou_xyxy, coco_size_bin_xyxy

if TYPE_CHECKING:
    from .cache import CacheEntry


def greedy_match_xyxy(
    gt: np.ndarray, pr: np.ndarray, iou_thr: float
) -> Tuple[int, int, int, List[float], Dict[str, Tuple[int, int]]]:
    size_counts: Dict[str, Tuple[int, int]] = {"small": (0, 0), "medium": (0, 0), "large": (0, 0)}
    if gt.size:
        for g in gt:
            b = coco_size_bin_xyxy(g)
            gtc, tpc = size_counts[b]
            size_counts[b] = (gtc + 1, tpc)

    if gt.size == 0 and pr.size == 0:
        return 0, 0, 0, [], size_counts
    if gt.size == 0:
        return 0, int(pr.shape[0]), 0, [], size_counts
    if pr.size == 0:
        return 0, 0, int(gt.shape[0]), [], size_counts

    ious = box_iou_xyxy(gt, pr)
    matched_gt: set[int] = set()
    matched_pr: set[int] = set()
    tp_ious: List[float] = []

    while True:
        gi, pj = np.unravel_index(int(np.argmax(ious)), ious.shape)
        best = float(ious[gi, pj])
        if best < float(iou_thr):
            break
        matched_gt.add(int(gi))
        matched_pr.add(int(pj))
        tp_ious.append(best)
        ious[gi, :] = -1.0
        ious[:, pj] = -1.0

    tp = len(matched_gt)
    fp = int(pr.shape[0]) - len(matched_pr)
    fn = int(gt.shape[0]) - len(matched_gt)

    if tp:
        for gi in matched_gt:
            b = coco_size_bin_xyxy(gt[gi])
            gtc, tpc = size_counts[b]
            size_counts[b] = (gtc, tpc + 1)

    return tp, fp, fn, tp_ious, size_counts


def ap_from_pr(tp_flags: np.ndarray, fp_flags: np.ndarray, total_gt: int, pr_points: int = 101) -> float:
    if total_gt <= 0:
        return 0.0

    tp_cum = np.cumsum(tp_flags).astype(np.float64)
    fp_cum = np.cumsum(fp_flags).astype(np.float64)
    recall = tp_cum / float(total_gt)
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)

    prec_env = precision.copy()
    for i in range(len(prec_env) - 2, -1, -1):
        prec_env[i] = max(prec_env[i], prec_env[i + 1])

    recall_samples = np.linspace(0.0, 1.0, pr_points)
    sampled = np.zeros_like(recall_samples)
    for i, r in enumerate(recall_samples):
        inds = np.where(recall >= r)[0]
        sampled[i] = prec_env[inds[0]] if inds.size else 0.0
    return float(np.mean(sampled))


@dataclass
class OperatingPointMetrics:
    precision: float
    recall: float
    f1: float
    miss_rate: float
    fp_per_image: float
    mean_iou_tp: float
    tp: int
    fp: int
    fn: int
    gt_total: int
    images_total: int
    images_with_gt: int
    recall_by_size: Dict[str, float]


def evaluate_from_cache(cache: Dict[str, "CacheEntry"], score_thr: float, iou_thr: float) -> OperatingPointMetrics:
    tp = fp = fn = 0
    gt_total = 0
    images_with_gt = 0
    tp_ious_all: List[float] = []

    size_totals = {"small": 0, "medium": 0, "large": 0}
    size_tps = {"small": 0, "medium": 0, "large": 0}

    for e in cache.values():
        gt = e.gt
        if gt.shape[0] > 0:
            images_with_gt += 1
            gt_total += int(gt.shape[0])
            for g in gt:
                size_totals[coco_size_bin_xyxy(g)] += 1

        if e.scores.size:
            m = e.scores >= float(score_thr)
            pr = e.boxes[m] if m.any() else np.zeros((0, 4), dtype=np.float32)
        else:
            pr = np.zeros((0, 4), dtype=np.float32)

        tpi, fpi, fni, tp_ious, size_counts = greedy_match_xyxy(gt, pr, float(iou_thr))
        tp += tpi
        fp += fpi
        fn += fni
        tp_ious_all.extend(tp_ious)
        for k, (_, tpc) in size_counts.items():
            size_tps[k] += tpc

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / gt_total if gt_total else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    recall_by_size = {k: (size_tps[k] / size_totals[k] if size_totals[k] else 0.0) for k in size_totals}

    return OperatingPointMetrics(
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        miss_rate=float(1.0 - recall),
        fp_per_image=float(fp / len(cache) if cache else 0.0),
        mean_iou_tp=float(np.mean(tp_ious_all) if tp_ious_all else 0.0),
        tp=int(tp),
        fp=int(fp),
        fn=int(fn),
        gt_total=int(gt_total),
        images_total=int(len(cache)),
        images_with_gt=int(images_with_gt),
        recall_by_size=recall_by_size,
    )


def compute_ap_pr_from_cache(
    cache: Dict[str, "CacheEntry"], iou_thr: float, pr_points: int
) -> Tuple[float, Dict[str, List[float]]]:
    total_gt = sum(int(e.gt.shape[0]) for e in cache.values())
    if total_gt == 0:
        return 0.0, {"recall": [], "precision": [], "scores": []}

    dets: List[Tuple[str, float, int]] = []
    for stem, e in cache.items():
        for j, sc in enumerate(e.scores.tolist()):
            dets.append((stem, float(sc), int(j)))
    if not dets:
        return 0.0, {"recall": [], "precision": [], "scores": []}

    dets.sort(key=lambda x: x[1], reverse=True)
    matched = {stem: np.zeros((e.gt.shape[0],), dtype=bool) for stem, e in cache.items()}
    tp_flags = np.zeros((len(dets),), dtype=np.int32)
    fp_flags = np.zeros((len(dets),), dtype=np.int32)

    for i, (stem, _, j) in enumerate(dets):
        e = cache[stem]
        gt = e.gt
        if gt.shape[0] == 0:
            fp_flags[i] = 1
            continue

        box = e.boxes[j : j + 1]
        ious = box_iou_xyxy(gt, box).reshape(-1)
        if ious.size == 0:
            fp_flags[i] = 1
            continue

        for k in np.argsort(ious)[::-1]:
            if matched[stem][k]:
                continue
            if float(ious[k]) >= float(iou_thr):
                tp_flags[i] = 1
                matched[stem][k] = True
                break
        else:
            fp_flags[i] = 1

    tp_cum = np.cumsum(tp_flags).astype(np.float64)
    fp_cum = np.cumsum(fp_flags).astype(np.float64)
    recall = (tp_cum / float(total_gt)).tolist()
    precision = (tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)).tolist()
    scores = [s for _, s, _ in dets]

    ap = ap_from_pr(tp_flags, fp_flags, total_gt=total_gt, pr_points=int(pr_points))
    return ap, {"recall": recall, "precision": precision, "scores": scores}
