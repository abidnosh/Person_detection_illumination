# FILE: eval/utils/zerodcepp_runtime.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W]


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    t = t.detach().float().clamp(0.0, 1.0)[0].permute(1, 2, 0).cpu().numpy()
    return Image.fromarray((t * 255.0).round().astype(np.uint8), mode="RGB")


@dataclass(frozen=True)
class ZeroDCEPPConfig:
    repo_dir: str
    ckpt_path: str = ""  # default: <repo_dir>/snapshots_Zero_DCE++/Epoch99.pth
    device: str = "cpu"
    scale_factor: int = 1


class ZeroDCEPP:
    """Loads Zero-DCE++ once per run and applies it to PIL images."""

    def __init__(self, cfg: ZeroDCEPPConfig) -> None:
        self.cfg = cfg
        self._net: Optional[torch.nn.Module] = None

    def _load(self) -> None:
        if self._net is not None:
            return

        repo = Path(self.cfg.repo_dir).expanduser().resolve()
        if not repo.exists():
            raise FileNotFoundError(f"Zero-DCE++ repo_dir not found: {repo}")

        model_py = repo / "model.py"
        if not model_py.exists():
            raise FileNotFoundError(f"Expected model.py in repo_dir, not found: {model_py}")

        ckpt = (
            Path(self.cfg.ckpt_path).expanduser().resolve()
            if self.cfg.ckpt_path
            else (repo / "snapshots_Zero_DCE++" / "Epoch99.pth")
        )
        if not ckpt.exists():
            raise FileNotFoundError(f"Zero-DCE++ checkpoint not found: {ckpt}")

        import importlib.util

        spec = importlib.util.spec_from_file_location("zerodcepp_model", str(model_py))
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not import {model_py}")

        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)

        if not hasattr(m, "enhance_net_nopool"):
            raise AttributeError("Zero-DCE++ model.py missing enhance_net_nopool (repo mismatch?)")

        net = m.enhance_net_nopool(scale_factor=int(self.cfg.scale_factor)).to(self.cfg.device).eval()

        state = torch.load(str(ckpt), map_location=self.cfg.device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
            state = {k.replace("module.", "", 1): v for k, v in state.items()}

        net.load_state_dict(state, strict=True)
        self._net = net

    @torch.no_grad()
    def __call__(self, img: Image.Image) -> Image.Image:
        self._load()
        assert self._net is not None
        x = pil_to_tensor(img).to(self.cfg.device)
        y = self._net(x)
        # print(type(y), len(y) if isinstance(y, (tuple, list)) else "tensor")
        # Some Zero-DCE++ forks return (enhanced, aux) or (enhanced, curves)
        if isinstance(y, (tuple, list)):
            y = y[0]
        return tensor_to_pil(y)
