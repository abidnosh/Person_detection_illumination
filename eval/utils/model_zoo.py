from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image


@dataclass
class Detector:
    """
    Unified detector interface.

    infer(images) returns list of dicts:
      {"boxes": (N,4) float32 xyxy abs pixels,
       "scores": (N,) float32,
       "labels": (N,) int64}
    """
    backend: str
    model_name: str
    device: str
    model: Any
    processor: Any = None  # HF only

    @torch.no_grad()
    def infer(self, images: Sequence[Image.Image], min_score: float) -> List[Dict[str, torch.Tensor]]:
        if self.backend == "hf":
            return _infer_hf(self.model, self.processor, images, self.device, min_score)
        if self.backend == "tv":
            return _infer_torchvision(self.model, images, self.device, min_score)
        raise ValueError(f"Unknown backend: {self.backend}")


def load_detector(backend: str, model_name: str, device: str) -> Detector:
    backend = backend.lower().strip()

    if backend == "hf":
        from transformers import AutoImageProcessor, AutoModelForObjectDetection

        try:
            processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
        except TypeError:
            processor = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModelForObjectDetection.from_pretrained(model_name).to(device).eval()
        return Detector(backend=backend, model_name=model_name, device=device, model=model, processor=processor)

    if backend == "tv":
        import torchvision

        model = _load_torchvision_detection_model(model_name)
        model = model.to(device).eval()
        return Detector(backend=backend, model_name=model_name, device=device, model=model, processor=None)

    raise ValueError(f"Unknown backend: {backend}")


# ------------------------- HF inference (DETR family) --------------------------
def _infer_hf(model, processor, images: Sequence[Image.Image], device: str, min_score: float):
    target_sizes = [im.size[::-1] for im in images]  # (h,w)
    inputs = processor(images=images, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model(**inputs)
    posts = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=float(min_score))
    return posts  # list of {"scores","labels","boxes"} tensors


# ------------------------- Torchvision inference -------------------------------
def _infer_torchvision(model, images: Sequence[Image.Image], device: str, min_score: float):
    # torchvision expects list[Tensor(C,H,W)] in [0,1]
    import torchvision.transforms.functional as F

    xs = [F.to_tensor(im).to(device) for im in images]
    outs = model(xs)  # list of dicts with boxes/scores/labels
    posts = []
    for o in outs:
        scores = o["scores"]
        keep = scores >= float(min_score)
        posts.append(
            {
                "boxes": o["boxes"][keep],
                "scores": o["scores"][keep],
                "labels": o["labels"][keep],
            }
        )
    return posts


def _load_torchvision_detection_model(name: str):
    """
    Torchvision model-name mapping.
    Use exactly these strings in --model_name for backend=tv.
    """
    import torchvision

    name = name.strip()
    fn = getattr(torchvision.models.detection, name, None)
    if fn is None:
        raise ValueError(
            f"Unknown torchvision detection model: {name}. "
            "Examples: fasterrcnn_resnet50_fpn, retinanet_resnet50_fpn, "
            "ssd300_vgg16, ssdlite320_mobilenet_v3_large"
        )
    # weights="DEFAULT" works on newer torchvision; fallback to pretrained=True on older
    try:
        return fn(weights="DEFAULT")
    except TypeError:
        return fn(pretrained=True)
