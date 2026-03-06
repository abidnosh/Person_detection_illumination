from __future__ import annotations
from .utils.enhance import Enhancer, EnhancementConfig
from eval.utils.zerodcepp_runtime import ZeroDCEPP, ZeroDCEPPConfig

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import numpy as np
from transformers import AutoImageProcessor, AutoModelForObjectDetection

from .utils.cache import CacheEntry, build_pred_cache
from .utils.device import pick_device
from .utils.gt import build_person_index
from .utils.metrics import compute_ap_pr_from_cache, evaluate_from_cache
from .utils.parsing import parse_iou_range, parse_sweep


def run_eval(args) -> None:
    device = pick_device(args.device.strip() or None)
    print(f"[INFO] Using device: {device}", flush=True)

    images = [p.strip() for p in Path(args.list_file).read_text().splitlines() if p.strip()]
    if not images:
        raise SystemExit(f"Empty list_file: {args.list_file}")

    gt_index = build_person_index(args.det_json)
    stems = [Path(p).stem for p in images]
    stem_hits = sum(1 for s in stems if s in gt_index)
    if stem_hits == 0:
        print("[WARN] 0 stems from list_file matched GT index keys -> GT lookup mismatch likely.", flush=True)
    else:
        print(f"[INFO] GT key sanity: {stem_hits}/{len(stems)} image stems found in GT index", flush=True)

    print(f"[INFO] Loading model: {args.model_name}", flush=True)
    try:
        processor = AutoImageProcessor.from_pretrained(args.model_name, use_fast=True)
    except TypeError:
        processor = AutoImageProcessor.from_pretrained(args.model_name)
    model = AutoModelForObjectDetection.from_pretrained(args.model_name).to(device).eval()

    sweep_thresholds = parse_sweep(args.sweep)
    required_min_score = float(min([args.min_score_cache, args.score_thr, *sweep_thresholds]))
    min_score_cache = max(0.0, required_min_score)

    t0 = time.perf_counter()
    enhancer = None
    if args.enhance_low_conf:
        steps = [s.strip() for s in args.enhance_steps.split(",") if s.strip()]
        learned: list = []  # <-- define it
        if any(s.lower() in ("zero_dce++", "zero-dce++") for s in steps):
            if not args.zerodcepp_repo_dir:
                raise SystemExit("--zerodcepp_repo_dir is required when using zero_dce++")
            learned.append(
                ZeroDCEPP(
                    ZeroDCEPPConfig(
                        repo_dir=args.zerodcepp_repo_dir,
                        ckpt_path=args.zerodcepp_ckpt,
                        device=args.zerodcepp_device,
                        scale_factor=args.zerodcepp_scale_factor
                    )
                )
            )
        enhancer = Enhancer(EnhancementConfig(steps=steps), learned=learned)
        # enhancer = Enhancer(EnhancementConfig(steps=steps))
    cache: Dict[str, CacheEntry] = build_pred_cache(
        images=images,
        gt_index=gt_index,
        model=model,
        processor=processor,
        device=device,
        batch_size=int(args.batch_size),
        max_dets=int(args.max_dets),
        min_score_cache=float(min_score_cache),
        enhance_low_conf=bool(args.enhance_low_conf),
        low_conf_thr=float(args.low_conf_thr),
        enhancer=enhancer,  
        gate_score_thr=args.score_thr,  # only enhance if below operating point score thr
        save_enhanced_dir=args.save_enhanced_dir,
        save_enhanced_max=args.save_enhanced_max,
        save_enhanced_every=args.save_enhanced_every,
        save_enhanced_side_by_side=args.save_enhanced_side_by_side,
    )
    t1 = time.perf_counter()
    print(f"[INFO] Cache built for {len(cache)} images in {(t1 - t0):.1f}s", flush=True)

    # Operating point
    opm = evaluate_from_cache(cache, score_thr=float(args.score_thr), iou_thr=float(args.iou_thr))
    print("\n==== Operating Point Metrics ====")
    print(f"Images: {opm.images_total} | Images with GT: {opm.images_with_gt}")
    print(f"GT total: {opm.gt_total} | TP: {opm.tp} | FP: {opm.fp} | FN: {opm.fn}")
    print(f"Score thr: {args.score_thr:.3f} | IoU thr: {args.iou_thr:.2f}")
    print(f"Precision: {opm.precision:.3f}")
    print(f"Recall:    {opm.recall:.3f}")
    print(f"F1 score:  {opm.f1:.3f}")
    print(f"Miss Rate: {opm.miss_rate:.3f}")
    print(f"FP / Image: {opm.fp_per_image:.3f}")
    print(f"Mean IoU (TP matches): {opm.mean_iou_tp:.3f}")
    print("Recall by object size:")
    for k in ["small", "medium", "large"]:
        print(f"  {k}: {opm.recall_by_size.get(k, 0.0):.3f}")

    # Recall@multiple IoUs (fixed score_thr)
    recall_iou_list = [float(x) for x in args.recall_iou_set.split(",") if x.strip()]
    recall_multi: Dict[str, float] = {}
    for iou_t in recall_iou_list:
        m = evaluate_from_cache(cache, score_thr=float(args.score_thr), iou_thr=float(iou_t))
        recall_multi[f"{iou_t:.2f}"] = m.recall
    print("\nRecall@Multiple IoUs (fixed score thr):")
    for k in sorted(recall_multi.keys(), key=lambda s: float(s)):
        print(f"  IoU {k}: {recall_multi[k]:.3f}")

    # AP/PR
    ap_ious = parse_iou_range(args.ap_iou)
    ap_by_iou: Dict[str, float] = {}
    pr_curves: Dict[str, Dict[str, List[float]]] = {}
    print("\nComputing AP/PR from cache...", flush=True)
    for iou_t in ap_ious:
        ap_val, curve = compute_ap_pr_from_cache(cache, iou_thr=float(iou_t), pr_points=int(args.pr_points))
        ap_by_iou[f"{iou_t:.2f}"] = ap_val
        pr_curves[f"{iou_t:.2f}"] = curve

    ap50 = ap_by_iou.get("0.50", ap_by_iou.get("0.5", 0.0))
    map_5095 = float(np.mean(list(ap_by_iou.values()))) if ap_by_iou else 0.0
    print("==== AP Metrics ====")
    print(f"AP@0.50: {float(ap50):.3f}")
    if len(ap_ious) > 1:
        print(f"mAP@[{ap_ious[0]:.2f}:{ap_ious[-1]:.2f}]: {map_5095:.3f}")

    # Sweep
    sweep = {"thresholds": sweep_thresholds, "recall": [], "precision": [], "f1": []}
    print("\nRecall/Precision vs Score Threshold (fixed IoU thr):")
    for thr in sweep_thresholds:
        m = evaluate_from_cache(cache, score_thr=float(thr), iou_thr=float(args.iou_thr))
        sweep["recall"].append(m.recall)
        sweep["precision"].append(m.precision)
        sweep["f1"].append(m.f1)
        print(f"  thr {thr:.2f}: recall {m.recall:.3f} | prec {m.precision:.3f} | f1 {m.f1:.3f}")

    elapsed = time.perf_counter() - t0
    print(f"\n[INFO] Done. Total elapsed: {elapsed:.1f}s", flush=True)

    if args.save_json:
        payload = {
            "list_file": args.list_file,
            "det_json": args.det_json,
            "model_name": args.model_name,
            "device": device,
            "cache": {
                "batch_size": int(args.batch_size),
                "max_dets": int(args.max_dets),
                "min_score_cache": float(min_score_cache),
            },
            "operating_point": asdict(opm),
            "recall_multiple_ious": recall_multi,
            "ap": {
                "ap_iou_spec": args.ap_iou,
                "ap_by_iou": ap_by_iou,
                "ap50": float(ap50),
                "map_5095": map_5095,
                "pr_curves": pr_curves,
            },
            "score_sweep": sweep,
        }
        Path(args.save_json).write_text(json.dumps(payload, indent=2))
        print(f"[INFO] Saved JSON: {args.save_json}", flush=True)
