"""Audit script: loads each model and dumps class names + metadata.

Created as part of the Indian traffic-sign expansion audit.
Read-only: never modifies any model file.
"""
import json
import os
import sys

from ultralytics import YOLO

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

MODELS = {
    "best.pt": os.path.join(PROJECT_ROOT, "best.pt"),
    "best_14class_backup.pt": os.path.join(PROJECT_ROOT, "best_14class_backup.pt"),
    "yolov8s-best.pt": os.path.join(PROJECT_ROOT, "road-sign-detection-yolo", "yolov8s-best.pt"),
    "infrastructure_detector.pt": os.path.join(
        PROJECT_ROOT, "sih2026-sankalp", "models", "infrastructure", "infrastructure_detector.pt"
    ),
    "indian_traffic.pt (referenced by indian_sign_test.py)": os.path.join(PROJECT_ROOT, "indian_traffic.pt"),
    "anpr best.pt": os.path.join(PROJECT_ROOT, "sih2026-sankalp", "models", "anpr", "best.pt"),
    "pedestrian_detector.pt": os.path.join(
        PROJECT_ROOT, "sih2026-sankalp", "models", "pedestrian", "pedestrian_detector.pt"
    ),
    "pothole.pt": os.path.join(PROJECT_ROOT, "sih2026-sankalp", "models", "pothole.pt"),
}

out = {}

for label, path in MODELS.items():
    entry = {"path": path, "exists": os.path.isfile(path)}
    if entry["exists"]:
        entry["size_mb"] = round(os.path.getsize(path) / (1024 * 1024), 2)
        try:
            model = YOLO(path)
            names = model.names
            entry["class_count"] = len(names)
            entry["classes"] = {int(k): str(v) for k, v in names.items()}
        except Exception as exc:  # noqa: BLE001
            entry["error"] = f"{type(exc).__name__}: {exc}"
    else:
        entry["error"] = "FILE NOT FOUND"
    out[label] = entry

dest = os.path.join(os.path.dirname(__file__), "model_class_audit.json")
with open(dest, "w", encoding="utf-8") as fh:
    json.dump(out, fh, indent=2, ensure_ascii=False)

for label, entry in out.items():
    print("=" * 70)
    print(label)
    if not entry["exists"]:
        print("  MISSING:", entry["path"])
        continue
    if "error" in entry:
        print("  ERROR:", entry["error"])
        continue
    print(f"  {entry['size_mb']} MB, {entry['class_count']} classes")
    for cid, name in entry["classes"].items():
        print(f"    {cid:>3} : {name}")

print("=" * 70)
print("Saved ->", dest)
