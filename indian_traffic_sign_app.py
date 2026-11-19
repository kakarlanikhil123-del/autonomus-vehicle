"""
Indian Traffic Sign Recognition — integrated application.

Preserves sign_test.py (that file is untouched). This is a SEPARATE app:
  - webcam 0 (CAP_DSHOW), image, or video input
  - model selection: 14-class model, 57-class Indian model, or BOTH + fusion
  - canonical names, categories and driver advisories (config/indian_traffic_signs.json)
  - speed-limit OCR with per-detection cooldown (no OCR every frame)
  - duplicate detections removed by IoU fusion
  - UNKNOWN shown when confidence is insufficient; unknowns are never coerced
    into a known class
  - traffic-light states via temporal smoothing (RED/GREEN only; YELLOW is
    NOT supported by any trained model -> UNKNOWN)
  - models loaded once, outside the frame loop
  - keyboard controls, graceful camera/weight errors, configurable settings

Run:
  python indian_traffic_sign_app.py                     # webcam, both models fused
  python indian_traffic_sign_app.py --mode 14           # only best.pt
  python indian_traffic_sign_app.py --mode 57           # only infrastructure model
  python indian_traffic_sign_app.py --source image --input <path.jpg>
  python indian_traffic_sign_app.py --source video --input <path.mp4> --save out.mp4
Keys (window): q=quit, 1/2/3=model mode, +/-=confidence threshold, t=toggle OCR, s=save frame
"""

import argparse
import os
import sys
import time
from collections import defaultdict, deque

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "sign_expansion"))

from canonical_mapping import (  # noqa: E402
    UNKNOWN_ADVISORY,
    UNKNOWN_NAME,
    DetectionFuser,
    SignCatalogue,
)
from traffic_light import (  # noqa: E402
    TrafficLightMonitor,
    pick_strongest_light,
)

try:
    import pytesseract

    TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.exists(TESSERACT_PATH):
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
        OCR_AVAILABLE = True
    else:
        OCR_AVAILABLE = False
except ImportError:
    OCR_AVAILABLE = False

MODEL_14 = os.path.join(ROOT, "best.pt")
MODEL_57 = os.path.join(
    ROOT, "sih2026-sankalp", "models", "infrastructure", "infrastructure_detector.pt"
)

VALID_SPEEDS = {20, 30, 40, 50, 60, 65, 70, 80, 90, 100, 110, 120}

# ------------------------------------------------------------------ OCR


class SpeedOCRCache:
    """Runs Tesseract at most once per second per sign track position,
    and only for speed-limit detections."""

    def __init__(self, cooldown_seconds=1.0):
        self.cooldown = cooldown_seconds
        self._last = {}

    def maybe_read(self, key, crop):
        now = time.perf_counter()
        last = self._last.get(key)
        if last is not None and (now - last["t"]) < self.cooldown:
            return last["value"]
        value = self._read(crop) if OCR_AVAILABLE else None
        self._last[key] = {"t": now, "value": value}
        return value

    def _read(self, crop):
        if crop is None or crop.size == 0:
            return None
        try:
            big = cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
            gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (3, 3), 0)
            _, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            cfg = "--psm 7 -c tessedit_char_whitelist=0123456789"
            text = pytesseract.image_to_string(thr, config=cfg)
            digits = "".join(ch for ch in text if ch.isdigit())
            if digits and digits.isdigit():
                val = int(digits)
                if val in VALID_SPEEDS:  # accept only plausible Indian limits
                    return str(val)
            return None
        except Exception:
            return None


# ------------------------------------------------------------------ drawing

CATEGORY_COLORS = {
    "A_regulatory": (0, 0, 255),        # red
    "B_warning": (0, 200, 255),         # amber
    "C_informatory": (80, 220, 80),     # green
    "D_speed_limit": (0, 120, 255),     # orange
    "E_traffic_light": (255, 0, 255),   # magenta
    "U_unrecognised": (160, 160, 160),  # grey
    "unknown": (160, 160, 160),
}


def draw_fused(frame, fused, ocr_cache, ocr_enabled, font_scale=0.6):
    overlay = frame.copy()
    used_classes = set()
    for pred in fused:
        x1, y1, x2, y2 = [int(v) for v in pred.bbox]
        category = pred.category
        color = CATEGORY_COLORS.get(category, (255, 255, 255))

        name = pred.display_name
        speed_text = None

        if not pred.is_unknown and pred.canonical_class:
            if pred.canonical_class.startswith("speed_limit"):
                if pred.canonical_class == "speed_limit_unspecified":
                    key = (x1 // 40, y1 // 40)  # coarse position key for cache
                    if ocr_enabled:
                        crop = frame[max(0, y1):y2, max(0, x1):x2]
                        val = ocr_cache.maybe_read(("speed", key), crop)
                        if val:
                            speed_text = f"SPEED LIMIT {val} KM/H"
                            name = f"Speed Limit {val}"
                    if not speed_text:
                        name = "Speed Limit (OCR pending)"
                else:
                    kmh = pred.canonical_class.rsplit("_", 1)[-1]
                    speed_text = f"SPEED LIMIT {kmh} KM/H"
                    name = f"Speed Limit {kmh}"
            used_classes.add(pred.canonical_class)

        label = f"{name} {pred.confidence:.2f}"
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        ty = max(y1 - 8, 14)
        cv2.putText(overlay, label, (x1, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)
        # advisory line under the box
        adv = pred.advisory
        ay = min(y2 + 18, frame.shape[0] - 8)
        cv2.putText(overlay, adv, (x1, ay), cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.9, color, 2)
        if speed_text:
            sy = min(y2 + 36, frame.shape[0] - 8)
            cv2.putText(overlay, speed_text, (x1, sy), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)

    # best (highest-confidence) detection banner
    if fused:
        best = fused[0]
        banner = f"ADVISORY: {best.advisory}"
        cv2.putText(overlay, banner, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(overlay, f"SIGNS: {len(fused)}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    else:
        cv2.putText(overlay, "No traffic sign detected", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    return overlay, used_classes


# ------------------------------------------------------------------ engines

def load_engines(mode, conf):
    """Load models ONCE. Returns (models dict, catalogue, fuser)."""
    catalogue = SignCatalogue()
    fuser = DetectionFuser(catalogue=catalogue, unknown_threshold=conf)
    models = {}
    if mode in ("14", "both"):
        from ultralytics import YOLO

        if not os.path.exists(MODEL_14):
            raise FileNotFoundError(MODEL_14)
        models["best.pt"] = YOLO(MODEL_14)
    if mode in ("57", "both"):
        from ultralytics import YOLO

        if not os.path.exists(MODEL_57):
            raise FileNotFoundError(MODEL_57)
        models["infrastructure_detector.pt"] = YOLO(MODEL_57)
    return models, catalogue, fuser


def run_inference(models, frame, conf, iou_imgsz=None):
    """Run all loaded models on one frame -> raw detection dicts."""
    raw = []
    for mname, model in models.items():
        results = model(frame, conf=conf, verbose=False)
        r = results[0]
        if r.boxes is None:
            continue
        for box in r.boxes:
            raw.append({
                "source_model": mname,
                "source_class_id": int(box.cls[0]),
                "raw_class_name": model.names[int(box.cls[0])],
                "confidence": float(box.conf[0]),
                "bbox": [float(v) for v in box.xyxy[0]],
            })
    return raw


# ------------------------------------------------------------------ sources

def open_camera(index=0, width=1280, height=720):
    cam = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cam.isOpened():
        cam = cv2.VideoCapture(index)  # fallback backend
    if not cam.isOpened():
        raise RuntimeError(f"Camera {index} could not be opened")
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cam


def run_source(args):
    models, catalogue, fuser = load_engines(args.mode, args.conf)
    ocr_cache = SpeedOCRCache(args.ocr_cooldown)
    light_monitor = TrafficLightMonitor()
    conf = args.conf
    ocr_enabled = True
    save_frame_dir = os.path.join(ROOT, "sign_expansion", "results", "phase2_run", "saved_frames")
    os.makedirs(save_frame_dir, exist_ok=True)

    writer = None
    frame_iter = None
    cam = None

    if args.source == "camera":
        try:
            cam = open_camera(args.camera_index)
        except RuntimeError as exc:
            print("CAMERA ERROR:", exc)
            print("Tip: close other apps using the camera, or try --camera-index 1")
            return 1
        window = "Indian Traffic Sign Recognition - Webcam"
    elif args.source == "video":
        if not args.input or not os.path.exists(args.input):
            print("ERROR: --input video file not found")
            return 1
        cam = cv2.VideoCapture(args.input)
        if not cam.isOpened():
            print("ERROR: video could not be opened")
            return 1
        window = f"Indian Traffic Sign Recognition - {os.path.basename(args.input)}"
        if args.save:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            fps = cam.get(cv2.CAP_PROP_FPS) or 25
            w = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer = cv2.VideoWriter(args.save, fourcc, fps, (w, h))
    else:  # image
        if not args.input or not os.path.exists(args.input):
            print("ERROR: --input image file not found")
            return 1
        frame = cv2.imread(args.input)
        if frame is None:
            print("ERROR: image could not be read")
            return 1
        raw = run_inference(models, frame, args.conf)
        fused = fuser.fuse(raw)
        out, _ = draw_fused(frame, fused, ocr_cache, ocr_enabled)
        print(f"Detections: {len(fused)}")
        for f in fused:
            print(" -", f.display_name, f"{f.confidence:.2f}", "|", f.advisory,
                  "| sources:", [s["source_model"] for s in f.sources])
        out_path = args.output or os.path.join(
            save_frame_dir, os.path.splitext(os.path.basename(args.input))[0] + "_annotated.jpg"
        )
        cv2.imwrite(out_path, out)
        print("annotated image:", out_path)
        return 0

    print("Controls: q=quit | 1/2/3=model mode | +/-=confidence | t=OCR toggle | s=save frame")
    try:
        while True:
            ret, frame = cam.read()
            if not ret:
                if args.source == "video":
                    print("video ended")
                else:
                    print("ERROR: could not read camera frame")
                break

            raw = run_inference(models, frame, conf)
            fused = fuser.fuse(raw)

            # traffic-light smoothing: pick the strongest valid RED/GREEN
            # detection (Traffic_signal warning signs and unknowns never
            # qualify), then smooth with the SAME confidence threshold the
            # rest of the app uses (no hidden 0.45 dead zone).
            light_canon, light_conf = pick_strongest_light(fused)
            light_state = light_monitor.update(light_canon, light_conf, unknown_threshold=conf)

            out, _ = draw_fused(frame, fused, ocr_cache, ocr_enabled)
            cv2.putText(out, f"LIGHT STATE: {light_state}",
                        (10, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 255), 2)
            cv2.putText(out, light_monitor.advisory, (10, 106),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 255), 2)

            if writer is not None:
                writer.write(out)

            cv2.imshow(window, out)
            key = cv2.waitKey(1 if args.source != "video" else 30) & 0xFF
            if key == ord("q"):
                break
            elif key in (ord("1"), ord("2"), ord("3")):
                new_mode = {"1": "14", "2": "57", "3": "both"}[chr(key)]
                if new_mode != args.mode:
                    print("switching model mode ->", new_mode)
                    for m in models.values():
                        del m
                    models, catalogue, fuser = load_engines(new_mode, conf)
                    args.mode = new_mode
            elif key == ord("+") or key == ord("="):
                conf = min(0.95, conf + 0.05)
                fuser.unknown_threshold = conf
                print("confidence:", conf)
            elif key == ord("-"):
                conf = max(0.10, conf - 0.05)
                fuser.unknown_threshold = conf
                print("confidence:", conf)
            elif key == ord("t"):
                ocr_enabled = not ocr_enabled
                print("OCR:", "ON" if ocr_enabled else "OFF")
            elif key == ord("s"):
                path = os.path.join(save_frame_dir, f"frame_{int(time.time())}.jpg")
                cv2.imwrite(path, out)
                print("saved:", path)
    finally:
        if cam is not None:
            cam.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
    return 0


# ------------------------------------------------------------------ CLI

def main():
    ap = argparse.ArgumentParser(description="Indian Traffic Sign Recognition (integrated app)")
    ap.add_argument("--source", choices=["camera", "image", "video"], default="camera")
    ap.add_argument("--input", help="image or video path (for image/video sources)")
    ap.add_argument("--output", help="output path for annotated image")
    ap.add_argument("--save", help="output path for annotated video (video source)")
    ap.add_argument("--mode", choices=["14", "57", "both"], default="both",
                    help="model selection: 14-class, 57-class, or fused both")
    ap.add_argument("--camera-index", type=int, default=0)
    ap.add_argument("--conf", type=float, default=0.35, help="confidence / UNKNOWN threshold")
    ap.add_argument("--ocr-cooldown", type=float, default=1.0, help="seconds between OCR runs per sign")
    args = ap.parse_args()
    return run_source(args)


if __name__ == "__main__":
    sys.exit(main())
