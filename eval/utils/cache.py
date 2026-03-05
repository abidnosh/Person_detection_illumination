from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from .gt import gt_for_image


@dataclass
class CacheEntry:
    gt: np.ndarray
    boxes: np.ndarray
    scores: np.ndarray


def _person_label_ids(model) -> List[int]:
    id2label = getattr(model.config, "id2label", {}) or {}
    return [int(i) for i, name in id2label.items() if (name or "").lower() == "person"]


@torch.no_grad()
def build_pred_cache(
    images: Sequence[str],
    gt_index: Dict[str, np.ndarray],
    model,
    processor,
    device: str,
    batch_size: int,
    max_dets: int,
    min_score_cache: float,
) -> Dict[str, CacheEntry]:
    """
    One-pass inference cache: per image stem -> (GT, person boxes, person scores).
    """
    person_ids = _person_label_ids(model)
    if not person_ids:
        raise SystemExit("Could not find 'person' label in model.config.id2label. Wrong checkpoint?")
    person_ids_np = np.array(person_ids, dtype=np.int64)

    cache: Dict[str, CacheEntry] = {}
    batch_paths: List[str] = []
    batch_imgs: List[Image.Image] = []

    def flush() -> None:
        if not batch_imgs:
            return

        target_sizes = [im.size[::-1] for im in batch_imgs]  # (h,w)
        inputs = processor(images=batch_imgs, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)
        posts = processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=float(min_score_cache)
        )

        for p, post in zip(batch_paths, posts):
            stem = Path(p).stem
            gt = gt_for_image(gt_index, p)

            boxes = post["boxes"].detach().cpu().numpy().astype(np.float32)
            scores = post["scores"].detach().cpu().numpy().astype(np.float32)
            labels = post["labels"].detach().cpu().numpy().astype(np.int64)

            keep = (scores >= float(min_score_cache)) & np.isin(labels, person_ids_np)
            if keep.any():
                boxes = boxes[keep]
                scores = scores[keep]
                if scores.size > max_dets:
                    idx = np.argsort(scores)[::-1][:max_dets]
                    boxes = boxes[idx]
                    scores = scores[idx]
            else:
                boxes = np.zeros((0, 4), dtype=np.float32)
                scores = np.zeros((0,), dtype=np.float32)

            cache[stem] = CacheEntry(gt=gt, boxes=boxes, scores=scores)

        batch_paths.clear()
        batch_imgs.clear()

    for p in tqdm(images, desc="Caching predictions", unit="img"):
        try:
            im = Image.open(p).convert("RGB")
        except Exception:
            stem = Path(p).stem
            cache[stem] = CacheEntry(
                gt=gt_for_image(gt_index, p),
                boxes=np.zeros((0, 4), dtype=np.float32),
                scores=np.zeros((0,), dtype=np.float32),
            )
            continue

        batch_paths.append(p)
        batch_imgs.append(im)
        if len(batch_imgs) >= int(batch_size):
            flush()

    flush()
    return cache
