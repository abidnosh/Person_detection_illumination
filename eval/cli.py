from __future__ import annotations
import argparse

def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list_file", required=True)
    ap.add_argument("--det_json", required=True)
    ap.add_argument("--model_name", default="PekingU/rtdetr_r50vd")
    ap.add_argument("--device", default="", help="cpu|cuda|mps (auto if empty)")

    ap.add_argument("--score_thr", type=float, default=0.5)
    ap.add_argument("--iou_thr", type=float, default=0.5)

    ap.add_argument("--ap_iou", default="0.5:0.95:0.05")
    ap.add_argument("--pr_points", type=int, default=101)

    ap.add_argument("--sweep", default="0.3,0.5,0.7")
    ap.add_argument("--recall_iou_set", default="0.5,0.75,0.9")

    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_dets", type=int, default=300)
    ap.add_argument("--min_score_cache", type=float, default=0.001)
    ap.add_argument("--save_json", default="")
    return ap
