from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

_ALLOWED_CATS = {"person", "pedestrian"}


def _key_variants(name: str) -> List[str]:
    if not name:
        return []
    base = os.path.basename(name)
    stem = Path(base).stem
    keys = [name, base, stem]
    out: List[str] = []
    seen: set[str] = set()
    for k in keys:
        if k and k not in seen:
            seen.add(k)
            out.append(k)
    return out


def gt_for_image(index: Dict[str, np.ndarray], img_path: str) -> np.ndarray:
    p = Path(img_path)
    candidates = [
        img_path,
        str(p),
        str(p.absolute()),
        str(p.resolve()),
        os.path.realpath(img_path),
        f"{p.parent.name}/{p.name}",
        p.name,
        p.stem,
    ]
    for k in candidates:
        if k in index:
            arr = index[k]
            return arr if isinstance(arr, np.ndarray) else np.array(arr, dtype=np.float32)
    return np.zeros((0, 4), dtype=np.float32)


def _as_xyxy_from_xywh(b: List[float]) -> Optional[List[float]]:
    if not b or len(b) != 4:
        return None
    x, y, w, h = map(float, b)
    if w <= 0 or h <= 0:
        return None
    return [x, y, x + w, y + h]


def build_person_index(det_json_path: str) -> Dict[str, np.ndarray]:
    """
    Auto-detect and index GT boxes for PERSON.
    Supports:
      - BDD det_20 JSON (list of dicts with labels[].box2d)
      - COCO JSON (dict with images/annotations/categories; bbox is xywh)
    Returns: mapping of name variants -> [N,4] float32 xyxy
    """
    with open(det_json_path, "r") as f:
        data = json.load(f)

    # COCO
    if isinstance(data, dict) and "images" in data and "annotations" in data:
        cats = {c.get("id"): (c.get("name") or "").lower() for c in data.get("categories", [])}
        person_cat_ids = {cid for cid, name in cats.items() if name in _ALLOWED_CATS or name.startswith("human")}
        if not person_cat_ids:
            person_cat_ids = {1}

        img_id_to_name = {int(im["id"]): (im.get("file_name") or "") for im in data["images"]}

        img_id_to_boxes: Dict[int, List[List[float]]] = {}
        for ann in data["annotations"]:
            if ann.get("category_id") not in person_cat_ids:
                continue
            xyxy = _as_xyxy_from_xywh(ann.get("bbox"))
            if not xyxy:
                continue
            iid = int(ann.get("image_id"))
            img_id_to_boxes.setdefault(iid, []).append(xyxy)

        index: Dict[str, np.ndarray] = {}
        for iid, fn in img_id_to_name.items():
            pn = Path(fn)
            boxes = img_id_to_boxes.get(iid, [])
            arr = np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32)

            strong = [fn, str(pn), os.path.realpath(fn), f"{pn.parent.name}/{pn.name}"]
            for k in strong:
                if k:
                    index[k] = arr
            for k in _key_variants(fn):
                index[k] = arr

        uniq_stems = {Path(os.path.basename(k)).stem for k in index.keys() if k}
        print(
            f"[INFO] Built COCO GT index: images={len(data['images'])}, keys={len(index)}, ~{len(uniq_stems)} unique stems"
        )
        return index

    # BDD
    if not isinstance(data, list):
        raise ValueError("Unknown GT format. Expected BDD list or COCO dict.")

    index: Dict[str, np.ndarray] = {}
    for item in data:
        raw_name = item.get("name") or (item.get("image") or {}).get("name", "") or ""
        labels = item.get("labels") or item.get("objects") or []
        boxes: List[List[float]] = []
        for lab in labels:
            cat = (lab.get("category") or "").lower()
            if cat not in _ALLOWED_CATS:
                continue
            b = lab.get("box2d") or {}
            if not all(k in b for k in ("x1", "y1", "x2", "y2")):
                continue
            x1, y1, x2, y2 = float(b["x1"]), float(b["y1"]), float(b["x2"]), float(b["y2"])
            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])

        arr = np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32)
        for k in _key_variants(raw_name):
            index[k] = arr

    uniq_stems = {Path(os.path.basename(k)).stem for k in index.keys() if k}
    print(f"[INFO] Built BDD GT index: images={len(data)}, keys={len(index)}, ~{len(uniq_stems)} unique stems")
    return index
