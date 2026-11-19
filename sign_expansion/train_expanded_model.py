"""
SAFE expanded-model training script for Indian traffic signs.

STATUS (2026-11-16): PROVIDED BUT NOT RUN.
  - No multi-class labelled Indian traffic-sign dataset exists in this project.
  - This machine is CPU-only (torch 2.14.1+cpu). Training on CPU would take
    days for a meaningful run and is not attempted with sample-quality data.
  - The only local labelled dataset is the DataCluster GENERIC single-class
    sample, licensed for evaluation/academic use only. Training a commercial
    product model on it is NOT permitted without a licence.

Run ONLY after a properly labelled dataset exists (see dataset_tools.py to
build one), and it will:
  - never overwrite existing weights (new, timestamped run dir),
  - use the installed ultralytics version,
  - make epochs/batch/imgsz/confidence configurable,
  - check GPU/CPU before starting,
  - use early stopping + validation,
  - report per-class precision/recall/mAP50/mAP50-95, confusion matrix,
    and compare against best.pt and infrastructure_detector.pt.

Usage:
  python sign_expansion/train_expanded_model.py --data <data.yaml> --model yolov8s.pt \
      --epochs 100 --batch 16 --imgsz 640 --device cpu
"""

import argparse
import datetime
import json
import os
import shutil

from ultralytics import YOLO

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUNS_ROOT = os.path.join(ROOT, "sign_expansion", "training_runs")
PROTECTED_WEIGHTS = {
    os.path.join(ROOT, "best.pt"),
    os.path.join(ROOT, "best_14class_backup.pt"),
    os.path.join(ROOT, "road-sign-detection-yolo", "yolov8s-best.pt"),
    os.path.join(ROOT, "sih2026-sankalp", "models", "infrastructure", "infrastructure_detector.pt"),
}


def compare_models(data_yaml, new_weights, conf=0.25):
    """Evaluate new model + both existing models on the SAME validation set."""
    import torch

    results = {}
    device = 0 if torch.cuda.is_available() else "cpu"
    candidates = {
        "new_expanded_model": new_weights,
        "best.pt (14-class)": os.path.join(ROOT, "best.pt"),
        "infrastructure_detector.pt (57-class)": os.path.join(
            ROOT, "sih2026-sankalp", "models", "infrastructure", "infrastructure_detector.pt"
        ),
    }
    for name, w in candidates.items():
        try:
            model = YOLO(w)
            m = model.val(data=data_yaml, conf=conf, device=device, verbose=False)
            per_class = None
            try:
                names = m.names if hasattr(m, "names") else model.names
                per_class = {}
                for i in range(len(names)):
                    try:
                        per_class[names[i]] = {
                            "precision": round(float(m.box.p[i]), 4),
                            "recall": round(float(m.box.r[i]), 4),
                            "mAP50": round(float(m.box.ap50[i]), 4),
                            "mAP50-95": round(float(m.box.ap[i]), 4),
                        }
                    except (IndexError, TypeError):
                        break
            except Exception:
                per_class = None
            results[name] = {
                "mAP50": float(m.box.map50),
                "mAP50-95": float(m.box.map),
                "precision": float(m.box.mp),
                "recall": float(m.box.mr),
                "per_class": per_class,
            }
        except Exception as exc:  # noqa: BLE001
            results[name] = {"error": repr(exc)}
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="path to data.yaml")
    ap.add_argument("--model", default="yolov8s.pt", help="base model to start from")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25, help="val confidence")
    ap.add_argument("--device", default=None, help="0 / cpu (auto-detected if omitted)")
    ap.add_argument("--patience", type=int, default=20, help="early-stopping patience")
    args = ap.parse_args()

    import torch

    gpu = torch.cuda.is_available()
    device = args.device if args.device else ("0" if gpu else "cpu")
    print("=== Environment check ===")
    print("cuda available:", gpu)
    print("device:", device)
    if not gpu:
        print("WARNING: CPU-only training. Expect very long runtimes.")

    data_yaml = os.path.abspath(args.data)
    if not os.path.exists(data_yaml):
        raise SystemExit(f"data.yaml not found: {data_yaml}")

    run_name = "expanded_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(RUNS_ROOT, run_name)
    if os.path.exists(run_dir):
        raise SystemExit(f"run dir already exists (never overwrite): {run_dir}")
    os.makedirs(run_dir)
    print("run directory:", run_dir)

    base = YOLO(args.model)
    results = base.train(
        data=data_yaml,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=device,
        patience=args.patience,
        project=run_dir,
        name="train",
        exist_ok=False,  # never overwrite
        verbose=True,
    )
    best = os.path.join(run_dir, "train", "weights", "best.pt")
    print("trained weights:", best)

    report = {
        "run_dir": run_dir,
        "args": vars(args),
        "trained_best_weights": best,
        "comparison_on_same_val_set": compare_models(data_yaml, best, args.conf),
        "recommendation":
            "Only replace best.pt / infrastructure_detector.pt if the new model "
            "measurably outperforms BOTH on the same validation set with per-class "
            "metrics reviewed. Never overwrite the protected weights in place.",
    }
    with open(os.path.join(run_dir, "comparison_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report["comparison_on_same_val_set"], indent=2))


if __name__ == "__main__":
    main()
