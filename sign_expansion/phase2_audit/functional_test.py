"""
Functional test: run both model families on the real sample image
traffic_test/test.jpg and record detections. Read-only; outputs go to
sign_expansion/phase2_audit/outputs/.
"""

import json
import os
import time

import cv2
from ultralytics import YOLO

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IMG = os.path.join(ROOT, "traffic_test", "test.jpg")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

MODELS = {
    "best.pt": os.path.join(ROOT, "best.pt"),
    "infrastructure_detector.pt": os.path.join(
        ROOT, "sih2026-sankalp", "models", "infrastructure", "infrastructure_detector.pt"
    ),
}

report = {}

for name, path in MODELS.items():
    model = YOLO(path)
    img = cv2.imread(IMG)
    t0 = time.perf_counter()
    results = model(img, conf=0.25, verbose=False)
    dt = time.perf_counter() - t0
    r = results[0]
    dets = []
    if r.boxes is not None and len(r.boxes) > 0:
        for box in r.boxes:
            dets.append(
                {
                    "class_id": int(box.cls[0]),
                    "class_name": model.names[int(box.cls[0])],
                    "confidence": round(float(box.conf[0]), 4),
                    "box_xyxy": [round(float(v), 1) for v in box.xyxy[0]],
                }
            )
    annotated = r.plot()
    out_path = os.path.join(OUT_DIR, f"func_test_{os.path.splitext(name)[0]}.jpg")
    cv2.imwrite(out_path, annotated)
    report[name] = {
        "image": IMG,
        "detections": dets,
        "num_detections": len(dets),
        "inference_seconds_first_run": round(dt, 3),
        "annotated_output": out_path,
    }
    print(name, "->", len(dets), "detections")
    for d in dets:
        print("   ", d["class_name"], d["confidence"])

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "functional_test_report.json"), "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)
