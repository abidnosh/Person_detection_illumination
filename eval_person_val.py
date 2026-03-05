#!/usr/bin/env python3
from __future__ import annotations

from rtdetr_eval.cli import build_argparser
from rtdetr_eval.core import run_eval


def main() -> None:
    ap = build_argparser()
    args = ap.parse_args()
    run_eval(args)


if __name__ == "__main__":
    main()