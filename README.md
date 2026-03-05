# Person Detection Evaluation (RT-DETR / Deformable-DETR) — BDD/nuImages

Clean, single-pass evaluation for **person** detection with HuggingFace object detection models.
Runs **one batched inference pass**, caches predictions, then computes:

* Operating point metrics: Precision / Recall / F1 / Miss Rate / FP per image / Mean IoU
* Recall at multiple IoU thresholds (fixed score threshold)
* COCO-style AP + PR curves over IoU range
* Score-threshold sweep (precision/recall/f1 vs threshold)

---

## Features

* **Single-pass caching**: no repeated inference for sweeps/AP/multi-IoU.
* **Supports GT formats**:

  * **BDD det_20** (labels with `box2d`)
  * **COCO** (images/annotations/categories, bbox in `xywh`)
* Filters predictions to **person** label only (from model config `id2label`).
* Outputs:

  * human-readable **console logs**
  * optional structured **JSON** for analysis/plots

---

## Repo structure

```text
person_detection/
  eval_person_train.py
  eval_person_val.py
  eval/
    __init__.py
    cli.py
    core.py
    utils/
      __init__.py
      cache.py
      boxes.py
      device.py
      gt.py
      metrics.py
      parsing.py
```

---

## Installation

Create an environment (recommended):

```bash
conda create -n persondet python=3.10 -y
conda activate persondet
python -m pip install -U torch transformers accelerate safetensors numpy pillow tqdm
```

Verify:

```bash
python -c "import torch, transformers; print(torch.__version__, transformers.__version__)"
```

---

## Quick start

### Train split evaluation

```bash
python -u eval_person_train.py \
  --list_file bdd100k/bdd100k/bdd_splits_train/calib_all.txt \
  --det_json /path/to/labels/det_20/det_train.json \
  --model_name SenseTime/deformable-detr \
  --device mps \
  --score_thr 0.5 \
  --iou_thr 0.5 \
  --sweep "0.05:0.95:0.05" \
  --save_json outputs/calib_all_train_deformable_detr.json
```

### Val/Test split evaluation

```bash
python -u eval_person_val.py \
  --list_file bdd100k/bdd100k/bdd_splits_val/calib_all.txt \
  --det_json /path/to/labels/det_20/det_val.json \
  --model_name SenseTime/deformable-detr \
  --device mps \
  --score_thr 0.5 \
  --iou_thr 0.5 \
  --sweep "0.05:0.95:0.05" \
  --save_json outputs/calib_all_val_deformable_detr.json
```

> For *test*, just pass a test list file via `--list_file` and choose a matching output json name.

---

## CLI arguments (core)

* `--list_file`: text file with absolute or relative image paths (one per line)
* `--det_json`: GT annotations (BDD det_20 JSON or COCO JSON)
* `--model_name`: HuggingFace model id or local checkpoint directory
* `--device`: `cpu|cuda|mps` (auto if empty)
* `--score_thr`: score threshold for operating point + recall@IoU
* `--iou_thr`: IoU threshold for operating point + sweep
* `--sweep`: thresholds list/range: `"0.3,0.5,0.7"` or `"0.05:0.95:0.05"`
* `--ap_iou`: IoU range for AP: `"0.5"` or `"0.5:0.95:0.05"`
* `--pr_points`: PR interpolation points (default 101)
* `--batch_size`: batched inference size
* `--max_dets`: cap detections per image kept in cache
* `--min_score_cache`: minimum score stored in cache (must be <= all thresholds you evaluate)
* `--save_json`: output metrics JSON path (optional)

---

## Output JSON

When `--save_json` is set, we write:

* `operating_point`: TP/FP/FN, precision/recall/f1, miss rate, fp/image, mean IoU, recall by size
* `recall_multiple_ious`: recall for each IoU in `--recall_iou_set`
* `ap`: AP-by-IoU, AP50, mAP@[range], PR curves (recall/precision/scores)
* `score_sweep`: precision/recall/f1 vs threshold

---

## Notes / gotchas

* **Person label must exist** in `model.config.id2label` with name `"person"`.
* GT lookup uses multiple filename keys (full path, basename, stem). If you see:

  * `[WARN] 0 stems ... matched GT index keys`
    then list_file paths and GT file naming don’t align.
* For HuggingFace post-processing, we pass `threshold=min_score_cache` to avoid default filtering.

---

## Development

Run import sanity checks:

```bash
python -c "from eval.cli import build_argparser; from eval.core import run_eval; print('imports ok')"
```

---

## License

Add your project license here (e.g., MIT/Apache-2.0).

---

## Acknowledgements

* HuggingFace Transformers (AutoModelForObjectDetection / AutoImageProcessor)
* RT-DETR / Deformable-DETR model authors
