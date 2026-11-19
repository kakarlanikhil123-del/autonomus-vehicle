"""Phase 1 audit: dump class names of every YOLO weight file without running inference."""
import glob
import os
import sys

from ultralytics import YOLO

WEIGHTS = [
    r"C:\Users\kakar\OneDrive\Desktop\autonomus vehicle\best.pt",
    r"C:\Users\kakar\OneDrive\Desktop\autonomus vehicle\best_14class_backup.pt",
    r"C:\Users\kakar\OneDrive\Desktop\autonomus vehicle\road-sign-detection-yolo\yolov8s-best.pt",
    r"C:\Users\kakar\OneDrive\Desktop\autonomus vehicle\sih2026-sankalp\models\pothole.pt",
    r"C:\Users\kakar\OneDrive\Desktop\autonomus vehicle\sih2026-sankalp\models\pothole_previous.pt",
    r"C:\Users\kakar\OneDrive\Desktop\autonomus vehicle\sih2026-sankalp\models\anpr\best.pt",
    r"C:\Users\kakar\OneDrive\Desktop\autonomus vehicle\sih2026-sankalp\models\infrastructure\infrastructure_detector.pt",
    r"C:\Users\kakar\OneDrive\Desktop\autonomus vehicle\sih2026-sankalp\models\pedestrian\pedestrian_detector.pt",
]

for w in WEIGHTS:
    print("=" * 70)
    print("WEIGHT:", w)
    try:
        m = YOLO(w)
        names = m.names
        print("  classes:", len(names))
        for i, n in sorted(names.items()):
            print(f"   {i:>3}: {n}")
    except Exception as e:  # noqa: BLE001
        print("  ERROR:", type(e).__name__, e)
