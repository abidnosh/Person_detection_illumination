# FILE: eval/utils/learned_enhancers.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional
import inspect
import numpy as np
import torch
from PIL import Image


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    """[1,3,H,W] float32 in [0,1]."""
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return t


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """Tensor [1,3,H,W] in [0,1] -> PIL."""
    t = t.detach().float().clamp(0.0, 1.0)[0].permute(1, 2, 0).cpu().numpy()
    return Image.fromarray((t * 255.0).round().astype(np.uint8), mode="RGB")


@dataclass(frozen=True)
class LearnedConfig:
    device: str
    # Zero-DCE++ (official repo layout)
    zerodcepp_repo_dir: str = ""
    zerodcepp_ckpt: str = ""  # if empty, uses <repo>/snapshots_Zero_DCE++/Epoch99.pth


def load_zerodcepp(cfg: LearnedConfig) -> Callable[[Image.Image], Image.Image]:
    """
    Loads official Zero-DCE++ pretrained weights (Epoch99.pth) from the authors' repo.
    Repo mentions: snapshots_Zero_DCE++/Epoch99.pth.  :contentReference[oaicite:2]{index=2}
    """
    repo = Path(cfg.zerodcepp_repo_dir).expanduser()
    if not repo.exists():
        raise FileNotFoundError(f"Zero-DCE++ repo_dir not found: {repo}")

    ckpt = Path(cfg.zerodcepp_ckpt).expanduser() if cfg.zerodcepp_ckpt else repo / "snapshots_Zero_DCE++" / "Epoch99.pth"
    if not ckpt.exists():
        raise FileNotFoundError(f"Zero-DCE++ checkpoint not found: {ckpt}")

    # Import authors' model.py dynamically from the repo folder
    import importlib.util

    spec = importlib.util.spec_from_file_location("zerodcepp_model", str(repo / "model.py"))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import Zero-DCE++ model.py from {repo}")

    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    # Authors typically expose an "enhance_net_nopool" network in model.py
    if not hasattr(m, "enhance_net_nopool"):
        raise AttributeError("Zero-DCE++ model.py missing enhance_net_nopool (repo mismatch)")

    ctor = m.enhance_net_nopool
    sig = inspect.signature(ctor)
    if "scale_factor" in sig.parameters:
        net = ctor(scale_factor=1).to(cfg.device).eval()
    else:
        net = ctor().to(cfg.device).eval()
    state = torch.load(str(ckpt), map_location=cfg.device)

    # handle common checkpoint formats
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}

    net.load_state_dict(state, strict=True)

    @torch.no_grad()
    def fn(img: Image.Image) -> Image.Image:
        x = pil_to_tensor(img).to(cfg.device)
        # Zero-DCE++ forward usually returns enhanced image directly
        y = net(x)
        return tensor_to_pil(y)

    return fn


def load_dncnn(cfg: LearnedConfig) -> Callable[[Image.Image], Image.Image]:
    """
    Loads pretrained DnCNN weights via deepinv (downloads from online weights hub).  :contentReference[oaicite:3]{index=3}
    """
    from deepinv.models import DnCNN

    model = DnCNN(in_channels=3, out_channels=3, pretrained="download", device=cfg.device).to(cfg.device).eval()

    @torch.no_grad()
    def fn(img: Image.Image) -> Image.Image:
        x = pil_to_tensor(img).to(cfg.device)
        y = model(x)
        return tensor_to_pil(y)

    return fn


def build_learned_enhancer(name: str, cfg: LearnedConfig) -> Callable[[Image.Image], Image.Image]:
    n = name.strip().lower()
    if n in ("zero_dce++", "zero-dce++", "zerodce++"):
        return load_zerodcepp(cfg)
    if n in ("dncnn",):
        return load_dncnn(cfg)
    raise ValueError(f"Unknown learned enhancer: {name}")
