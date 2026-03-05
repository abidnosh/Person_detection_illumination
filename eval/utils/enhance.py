from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Sequence

import numpy as np
from PIL import Image

try:
    import cv2  # opencv-python
except Exception as e:  # pragma: no cover
    cv2 = None


def _require_cv2() -> None:
    if cv2 is None:
        raise ImportError("opencv-python is required for enhancement steps. Install: pip install opencv-python")


def pil_to_bgr(img: Image.Image) -> np.ndarray:
    arr = np.array(img.convert("RGB"), dtype=np.uint8)
    return arr[:, :, ::-1].copy()  # RGB->BGR


def bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    rgb = bgr[:, :, ::-1]
    return Image.fromarray(rgb.astype(np.uint8), mode="RGB")


def gray_world(bgr: np.ndarray) -> np.ndarray:
    """Simple gray-world color constancy."""
    b = bgr[:, :, 0].astype(np.float32)
    g = bgr[:, :, 1].astype(np.float32)
    r = bgr[:, :, 2].astype(np.float32)

    mb, mg, mr = float(b.mean()), float(g.mean()), float(r.mean())
    m = (mb + mg + mr) / 3.0
    sb = m / max(mb, 1e-6)
    sg = m / max(mg, 1e-6)
    sr = m / max(mr, 1e-6)

    out = bgr.astype(np.float32)
    out[:, :, 0] *= sb
    out[:, :, 1] *= sg
    out[:, :, 2] *= sr
    return np.clip(out, 0, 255).astype(np.uint8)


# def retinex_ssr(bgr: np.ndarray, sigma: float = 30.0) -> np.ndarray:
#     """
#     Simple Single-Scale Retinex (SSR) approximation.
#     This is not a full Retinex implementation, but works as a lightweight enhancer.
#     """
#     _require_cv2()
#     img = bgr.astype(np.float32) + 1.0
#     blur = cv2.GaussianBlur(img, ksize=(0, 0), sigmaX=sigma, sigmaY=sigma) + 1.0
#     ret = np.log(img) - np.log(blur)
#     ret = cv2.normalize(ret, None, 0, 255, cv2.NORM_MINMAX)
#     return ret.astype(np.uint8)

# Replacing retinex_ssr with msrcr (Multi-Scale Retinex with Color Restoration)

def _msr(bgr: np.ndarray, sigmas: Sequence[float]) -> np.ndarray:
    _require_cv2()
    img = bgr.astype(np.float32) + 1.0
    log_img = np.log(img)

    ret = np.zeros_like(img, dtype=np.float32)
    for s in sigmas:
        blur = cv2.GaussianBlur(img, ksize=(0, 0), sigmaX=float(s), sigmaY=float(s)) + 1.0
        ret += log_img - np.log(blur)

    ret /= float(len(sigmas))
    return ret


def retinex_msrcr(
    bgr: np.ndarray,
    sigmas: Sequence[float] = (15.0, 80.0, 250.0),
    alpha: float = 125.0,
    beta: float = 46.0,
    gain: float = 1.0,
    offset: float = 0.0,
) -> np.ndarray:
    """
    MSRCR Retinex (Multi-Scale Retinex + Color Restoration).
    Much less "washed-out gray fog" than SSR.

    Defaults are common in practice; you can tune alpha/beta if needed.
    """
    _require_cv2()

    img = bgr.astype(np.float32) + 1.0  # avoid log(0)
    msr = _msr(bgr, sigmas=sigmas)

    # Color restoration term
    sum_rgb = np.sum(img, axis=2, keepdims=True)
    c = beta * (np.log(alpha * img) - np.log(sum_rgb + 1.0))

    out = gain * (msr * c) + offset

    # Normalize per-channel to [0,255]
    out_norm = np.zeros_like(out, dtype=np.float32)
    for ch in range(3):
        out_norm[:, :, ch] = cv2.normalize(out[:, :, ch], None, 0, 255, cv2.NORM_MINMAX)

    return np.clip(out_norm, 0, 255).astype(np.uint8)

def clahe(bgr: np.ndarray, clip_limit: float = 2.0, tile_grid_size: int = 8) -> np.ndarray:
    """CLAHE on L-channel in LAB."""
    _require_cv2()
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe_op = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(int(tile_grid_size), int(tile_grid_size)))
    l2 = clahe_op.apply(l)
    lab2 = cv2.merge([l2, a, b])
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)


def fast_nlm_denoise(bgr: np.ndarray, h: float = 6.0, template_window: int = 7, search_window: int = 21) -> np.ndarray:
    """Fast Non-Local Means denoising (OpenCV)."""
    _require_cv2()
    return cv2.fastNlMeansDenoisingColored(
        bgr, None, h=float(h), hColor=float(h), templateWindowSize=int(template_window), searchWindowSize=int(search_window)
    )


# --- Hooks for learned enhancers (Zero-DCE++, DnCNN) --------------------------
class LearnedEnhancer:
    """
    Minimal interface for learned enhancement models.
    Implement __call__(pil_img)->pil_img.
    """

    def __call__(self, img: Image.Image) -> Image.Image:  # pragma: no cover
        raise NotImplementedError


@dataclass(frozen=True)
class EnhancementConfig:
    steps: Sequence[str]
    retinex_sigma: float = 30.0
    clahe_clip: float = 2.0
    clahe_grid: int = 8
    nlm_h: float = 6.0
    nlm_template: int = 7
    nlm_search: int = 21


class Enhancer:
    """
      - purely classical steps by default
      - optional learned model hooks (not enabled unless you pass them)
    """

    def __init__(self, cfg: EnhancementConfig, learned: List[Callable[[Image.Image], Image.Image]] | None = None) -> None:
        self.cfg = cfg
        self.learned = learned or []

    def apply(self, img: Image.Image) -> Image.Image:
        bgr = pil_to_bgr(img)

        for name in self.cfg.steps:
            n = name.strip().lower()
            if n in ("grayworld", "gray_world"):
                bgr = gray_world(bgr)
            # elif n in ("retinex", "ssr", "retinex_ssr"):
            #     bgr = retinex_ssr(bgr, sigma=float(self.cfg.retinex_sigma))
            elif n in ("msrcr", "retinex_msrcr"):
                bgr = retinex_msrcr(bgr)
            elif n == "nlm" or n == "fast_nlm":
                bgr = fast_nlm_denoise(
                    bgr,
                    h=float(self.cfg.nlm_h),
                    template_window=int(self.cfg.nlm_template),
                    search_window=int(self.cfg.nlm_search),
                )
            elif n == "clahe":
                bgr = clahe(
                    bgr,
                    clip_limit=float(self.cfg.clahe_clip),
                    tile_grid_size=int(self.cfg.clahe_grid),
                )
            elif n in ("zero_dce", "zero-dce", "zero_dce++", "zero-dce++", "dncnn"):
                # hook point: run learned enhancers on PIL, not numpy
                pil = bgr_to_pil(bgr)
                for fn in self.learned:
                    pil = fn(pil)
                bgr = pil_to_bgr(pil)
            else:
                raise ValueError(f"Unknown enhancement step: {name}")

        return bgr_to_pil(bgr)
