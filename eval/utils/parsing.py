from __future__ import annotations

from typing import List


def parse_iou_range(spec: str) -> List[float]:
    """'0.5' or '0.5:0.95:0.05'."""
    s = spec.strip()
    if ":" not in s:
        return [float(s)]
    a, b, c = s.split(":")
    start, end, step = float(a), float(b), float(c)
    out: List[float] = []
    v = start
    while v <= end + 1e-9:
        out.append(round(v, 4))
        v += step
    return out


def parse_sweep(spec: str) -> List[float]:
    """'0.3,0.5,0.7' or '0.1:0.9:0.1'."""
    s = spec.strip()
    if "," in s:
        return [float(x) for x in s.split(",") if x.strip()]
    if ":" in s:
        a, b, c = s.split(":")
        start, end, step = float(a), float(b), float(c)
        out: List[float] = []
        v = start
        while v <= end + 1e-9:
            out.append(round(v, 4))
            v += step
        return out
