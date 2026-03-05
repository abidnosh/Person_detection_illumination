from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

from .enhance import Enhancer, EnhancementConfig
from PIL import Image
import os

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
    enhance_low_conf: bool = False,
    low_conf_thr: float = 0.20,
    enhancer: Enhancer | None = None,
    gate_score_thr: float = 0.50,
    save_enhanced_dir: str = "",
    save_enhanced_max: int = 50,
    save_enhanced_every: int = 50,
    save_enhanced_side_by_side: bool = False,
) -> Dict[str, CacheEntry]:
    """
    One-pass inference cache: per image stem -> (GT, person boxes, person scores).
    """
    saved_enhanced = 0
    seen_enhanced = 0
    person_ids = _person_label_ids(model)
    if not person_ids:
        raise SystemExit("Could not find 'person' label in model.config.id2label. Wrong checkpoint?")
    person_ids_np = np.array(person_ids, dtype=np.int64)

    cache: Dict[str, CacheEntry] = {}
    batch_paths: List[str] = []
    batch_imgs: List[Image.Image] = []

    def infer_batch(batch_paths: List[str], batch_imgs: List[Image.Image]) -> List[dict]:
        target_sizes = [im.size[::-1] for im in batch_imgs]
        inputs = processor(images=batch_imgs, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)
        return processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=float(min_score_cache))

    def should_enhance_from_scores(
            scores: np.ndarray,
            *,
            good_thr: float = 0.5,
            low_conf_thr: float = 0.2,
            min_good: int = 1,
            topk: int = 5,
            topk_mean_thr: float = 0.25,
            use_topk_mean: bool = False,
        ) -> bool:
            if scores.size == 0:
                return False  # no detections => enhance

            mx = float(scores.max())
            good = int((scores >= good_thr).sum())
            if good < min_good:
                return True
            if mx < low_conf_thr:
                return True

            if use_topk_mean:
                k = min(int(topk), int(scores.size))
                top_mean = float(np.sort(scores)[-k:].mean())
                if top_mean < float(topk_mean_thr):
                    return True

            return False

    
    def maybe_save_enhanced(orig: Image.Image, enh: Image.Image, stem: str) -> None:
        nonlocal saved_enhanced
        if not save_enhanced_dir:
            return
        if saved_enhanced >= int(save_enhanced_max):
            return
        os.makedirs(save_enhanced_dir, exist_ok=True)

        if save_enhanced_side_by_side:
            w, h = orig.size
            canvas = Image.new("RGB", (w * 2, h))
            canvas.paste(orig, (0, 0))
            canvas.paste(enh, (w, 0))
            out = canvas
        else:
            out = enh

        out.save(os.path.join(save_enhanced_dir, f"{stem}.jpg"), quality=95)
        saved_enhanced += 1

    def store_posts(batch_paths: List[str], posts: List[dict]) -> None:
        for p, post in zip(batch_paths, posts):
            stem = Path(p).stem
            gt = gt_for_image(gt_index, p)
            boxes = post["boxes"].detach().cpu().numpy().astype(np.float32)    # [N,4]
            scores = post["scores"].detach().cpu().numpy().astype(np.float32)  # [N]
            labels = post["labels"].detach().cpu().numpy().astype(np.int64)    # [N]

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

    def flush(batch_paths: List[str], batch_imgs: List[Image.Image]) -> None:
        nonlocal seen_enhanced
        if not batch_imgs:
            return
        try:
            posts = infer_batch(batch_paths, batch_imgs)
            store_posts(batch_paths, posts)

            if enhance_low_conf:
                if enhancer is None:
                    raise ValueError("enhance_low_conf=True requires enhancer")

                low_paths: List[str] = []
                low_imgs: List[Image.Image] = []
                enhanced_count = 0
                for p, im in zip(batch_paths, batch_imgs):
                    stem = Path(p).stem
                    scores = cache[stem].scores
                    if should_enhance_from_scores(
                        scores,
                        good_thr=float(gate_score_thr),
                        low_conf_thr=float(low_conf_thr),
                        min_good=1,
                        use_topk_mean=True,
                    ):
                        enhanced_count += 1
                        low_paths.append(p)
                        seen_enhanced += 1 
                        enh = enhancer.apply(im)
                        assert enh.size == im.size, f"Enhancer changed image size: {im.size} -> {enh.size} for {p}"
                        if (seen_enhanced % int(save_enhanced_every) == 0):
                            maybe_save_enhanced(im, enh, stem)

                        low_imgs.append(enh)
                print(f"[INFO] Enhanced {enhanced_count}/{len(batch_paths)} in this flush")
                if low_imgs:
                    posts2 = infer_batch(low_paths, low_imgs)
                    store_posts(low_paths, posts2)
        finally:
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
            flush(batch_paths, batch_imgs)
            batch_paths, batch_imgs = [], []

    flush(batch_paths, batch_imgs)  # flush remaining
    return cache
