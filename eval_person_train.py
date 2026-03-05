#!/usr/bin/env python3
from __future__ import annotations

from eval.cli import build_argparser
from eval.core import run_eval


def main() -> None:
    args = build_argparser().parse_args()
    run_eval(args)


if __name__ == "__main__":
    main()