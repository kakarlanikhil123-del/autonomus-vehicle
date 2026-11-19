"""
Headless webcam smoke test: open camera 0 (CAP_DSHOW), grab frames, run the
fused pipeline on a few frames, then release. Records timing + result honestly.
"""

import json
import os
import sys
import time

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "sign_expansion"))

from canonical_mapping import DetectionFuser, SignCatalogue  # noqa: E402
from ultralytics import YOLO  # noqa: E402

report = {"camera_index": 0, "backend": "CAP_DSHOW"}

cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
opened = cam.isOpened()
report["opened"] = opened

if opened:
    ret_probe, _ = cam.read()
    report["read_probe"] = bool(ret_probe)
    if ret_probe:
        models = {
            "best.pt": YOLO(os.path.join(ROOT, "best.pt")),
            "infrastructure_detector.pt": YOLO(
                os.path.join(ROOT, "sih2026-sankalp", "models", "infrastructure", "infrastructure_detector.pt")
            ),
        }
        fuser = DetectionFuser(catalogue=SignCatalogue())
        w, h = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        report["frame_size"] = [w, h]
        times = []
        dets_seen = []
        n_frames = 30
        ok_frames = 0
        for i in range(n_frames):
            ret, frame = cam.read()
            if not ret:
                continue
            ok_frames += 1
            t0 = time.perf_counter()
            raw = []
            for mname, model in models.items():
                r = model(frame, conf=0.35, verbose=False)[0]
                if r.boxes is not None:
                    for box in r.boxes:
                        raw.append({
                            "source_model": mname,
                            "source_class_id": int(box.cls[0]),
                            "raw_class_name": model.names[int(box.cls[0])],
                            "confidence": float(box.conf[0]),
                            "bbox": [float(v) for v in box.xyxy[0]],
                        })
            fused = fuser.fuse(raw)
            times.append(time.perf_counter() - t0)
            dets_seen.extend(f.display for f in fused)
        report["frames_ok"] = ok_frames
        if times:
            report["avg_ms_per_frame_fused"] = round(sum(times) / len(times) * 1000, 1)
            report["fps_fused"] = round(1 / (sum(times) / len(times)), 2)
        report["detections_during_test"] = dets_seen or []

cam.release()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "webcam_test_report.json")
with open(out, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)
print(json.dumps(report, indent=2))
