"""
Phase 1 verification: load every sign-related model, confirm class IDs/names,
and verify protected weights are unchanged (SHA-256).

Read-only. Does not modify any weights or source files.
"""

import hashlib
import json
import os
import sys

from ultralytics import YOLO

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODELS = {
    "best.pt": os.path.join(ROOT, "best.pt"),
    "best_14class_backup.pt": os.path.join(ROOT, "best_14class_backup.pt"),
    "yolov8s-best.pt": os.path.join(ROOT, "road-sign-detection-yolo", "yolov8s-best.pt"),
    "infrastructure_detector.pt": os.path.join(
        ROOT, "sih2026-sankalp", "models", "infrastructure", "infrastructure_detector.pt"
    ),
}

PROTECTED = {
    "best.pt": "C62E044A49BC095EDB3644794C7194062EBBE64D398A8141E41F29BE344AE721",
    "best_14class_backup.pt": "C62E044A49BC095EDB3644794C7194062EBBE64D398A8141E41F29BE344AE721",
    "yolov8s-best.pt": "C62E044A49BC095EDB3644794C7194062EBBE64D398A8141E41F29BE344AE721",
    "infrastructure_detector.pt": "7A71CDEE3C8E382DE5DEBE220ABD8EE499B6806E45255DBA80D69FB453C35837",
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def main():
    report = {"models": {}, "protected_hashes": {}}
    ok = True

    for name, path in MODELS.items():
        entry = {"path": path, "exists": os.path.exists(path)}
        if not entry["exists"]:
            entry["error"] = "FILE NOT FOUND"
            ok = False
        else:
            try:
                model = YOLO(path)
                entry["class_count"] = len(model.names)
                entry["classes"] = {str(k): v for k, v in model.names.items()}
                # Load test class names from the checkpoint task metadata too
                entry["model_type"] = getattr(model, "task", "unknown")
            except Exception as exc:  # noqa: BLE001
                entry["error"] = repr(exc)
                ok = False
        report["models"][name] = entry

    for name, expected in PROTECTED.items():
        path = MODELS[name]
        actual = sha256(path) if os.path.exists(path) else None
        match = actual == expected
        report["protected_hashes"][name] = {
            "expected": expected,
            "actual": actual,
            "match": match,
        }
        if not match:
            ok = False

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "verify_models_report.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print("STATUS:", "OK" if ok else "MISMATCH/ERROR")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
