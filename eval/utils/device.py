from __future__ import annotations

from typing import Optional

import torch


def pick_device(explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
