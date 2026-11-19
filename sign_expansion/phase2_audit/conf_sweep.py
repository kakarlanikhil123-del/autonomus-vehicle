"""
Sweep confidence thresholds on available sample images for both models.
Read-only; writes annotated outputs into sign_expansion/phase2_audit/outputs/.
"""

import json
import os

import cv2
from ultralytics import YOLO

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

IMAGES = {
    "traffic_test": os.path.join(ROOT, "traffic_test", "test.jpg"),
    "stop_test": os.path.join(ROOT, "sih2026-sankalp", "model_test_images", "stop_test.jpg"),
}

MODELS = {
    "best14": os.path.join(ROOT, "best.pt"),
    "infra57": os.path.join(
        ROOT, "sih2026-sankalp", "models", "infrastructure", "infrastructure_detector.pt"
    ),
}

report = {}

for mkey, mpath in MODELS.items():
    model = YOLO(mpath)
    for ikey, ipath in IMAGES.items():
        img = cv2.imread(ipath)
        if img is None:
            report[f"{mkey}/{ikey}"] = {"error": "image not readable"}
            continue
        best = {"conf": None, "detections": []}
        for conf in (0.10, 0.15, 0.20, 0.30, 0.50):
            r = model(img, conf=conf, verbose=False)[0]
            dets = []
            if r.boxes is not None:
                for box in r.boxes:
                    dets.append(
                        {
                            "class": model.names[int(box.cls[0])],
                            "confidence": round(float(box.conf[0]), 3),
                            "box": [round(float(v)) for v in box.xyxy[0]],
                        }
                    )
            report[f"{mkey}/{ikey}"] = report.get(f"{mkey}/{ikey}", {})
            report[f"{mkey}/{ikey}"][f"conf_{conf}"] = dets
            if dets and best["conf"] is None:
                best = {"conf": conf, "detections": dets}
        # save annotated image at the lowest threshold
        r = model(img, conf=0.10, verbose=False)[0]
        cv2.imwrite(os.path.join(OUT_DIR, f"sweep_{mkey}_{ikey}.jpg"), r.plot())
        print(f"{mkey}/{ikey}: best={best['conf']} det={[d['class'] + ':' + str(d['confidence']) for d in best['detections']]}")

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "conf_sweep_report.json"), "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)
