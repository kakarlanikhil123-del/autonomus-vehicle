"""
Unit tests for the canonical mapping, fusion, traffic-light smoothing and
dataset tools. Run: python sign_expansion/tests_unit.py
"""

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "sign_expansion"))

from canonical_mapping import (  # noqa: E402
    UNKNOWN_NAME,
    DetectionFuser,
    SignCatalogue,
    iou,
)
from traffic_light import TrafficLightMonitor  # noqa: E402
from dataset_tools import validate_voc_record, read_voc_xml, stratified_split  # noqa: E402

PASS, FAIL = [], []


def check(name, cond):
    (PASS if cond else FAIL).append(name)
    print(("PASS" if cond else "FAIL"), name)


cat = SignCatalogue()
fuser = DetectionFuser(catalogue=cat)

# --- mapping
e = cat.resolve("best.pt", 13, "Stop")
check("map best.pt Stop -> stop", e and e["canonical_id"] == "stop")
e = cat.resolve("infrastructure_detector.pt", 42, "Stop_Sign")
check("map infra 42 Stop_Sign -> stop", e and e["canonical_id"] == "stop")
e = cat.resolve("infrastructure_detector.pt", 11, "Dangerous Rright Curve Ahead")
check("map infra typo 'Rright' -> right_curve", e and e["canonical_id"] == "right_curve")
e = cat.resolve("infrastructure_detector.pt", 29, "Pedestrain Crossing")
check("map infra typo 'Pedestrain' -> pedestrian_crossing", e and e["canonical_id"] == "pedestrian_crossing")
e = cat.resolve("infrastructure_detector.pt", 1, "50 mph speed limit")
check("infra mph class unmapped (never coerced to km/h)", e is None)
check("catalogue has 89 signs", len(cat.data["signs"]) == 89)
check("no duplicate canonical ids", len({s["canonical_id"] for s in cat.data["signs"]}) == len(cat.data["signs"]))
check("yellow light has NO model support",
      next(s for s in cat.data["signs"] if s["canonical_id"] == "traffic_light_yellow")["model_support"] == [])

# --- IoU
check("iou identical boxes = 1", abs(iou([0, 0, 10, 10], [0, 0, 10, 10]) - 1.0) < 1e-9)
check("iou disjoint = 0", iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0)

# --- fusion: duplicate boxes from two models of same sign
raw = [
    {"source_model": "best.pt", "source_class_id": 13, "raw_class_name": "Stop",
     "confidence": 0.90, "bbox": [100, 100, 200, 200]},
    {"source_model": "infrastructure_detector.pt", "source_class_id": 42, "raw_class_name": "Stop_Sign",
     "confidence": 0.70, "bbox": [105, 98, 198, 202]},
]
fused = fuser.fuse(raw)
check("two-model same sign fused to ONE box", len(fused) == 1)
check("fused keeps stronger conf", fused[0].confidence == 0.90)
check("fused carries both sources", len(fused[0].sources) == 2)
check("fused canonical = stop", fused[0].canonical_class == "stop")

# --- fusion: low-confidence prediction never overrides stronger
raw = [
    {"source_model": "best.pt", "source_class_id": 13, "raw_class_name": "Stop",
     "confidence": 0.85, "bbox": [100, 100, 200, 200]},
    {"source_model": "infrastructure_detector.pt", "source_class_id": 42, "raw_class_name": "Stop_Sign",
     "confidence": 0.30, "bbox": [100, 100, 200, 200]},
]
fused = fuser.fuse(raw)
check("low-conf same-class member does not change winner conf", fused[0].confidence == 0.85)

# --- fusion: below unknown threshold -> UNKNOWN, never coerced
raw = [
    {"source_model": "infrastructure_detector.pt", "source_class_id": 40,
     "raw_class_name": "Stack type Advance Direction sign",
     "confidence": 0.40, "bbox": [0, 0, 100, 100]},
]
fused = fuser.fuse(raw)
check("low-conf prediction becomes UNKNOWN", fused[0].is_unknown and fused[0].display_name == UNKNOWN_NAME)
check("unknown has no canonical_class coerced", fused[0].canonical_class is None)

# --- fusion: conflicting classes on same physical sign keep stronger, record conflict
raw = [
    {"source_model": "best.pt", "source_class_id": 13, "raw_class_name": "Stop",
     "confidence": 0.80, "bbox": [100, 100, 200, 200]},
    {"source_model": "infrastructure_detector.pt", "source_class_id": 25, "raw_class_name": "No Entry",
     "confidence": 0.55, "bbox": [100, 100, 200, 200]},
]
fused = fuser.fuse(raw)
check("conflict keeps stronger class", fused[0].canonical_class == "stop")
check("conflict recorded", "no_entry" in fused[0].conflicts)

# --- two separate signs are NOT merged
raw = [
    {"source_model": "best.pt", "source_class_id": 13, "raw_class_name": "Stop",
     "confidence": 0.9, "bbox": [0, 0, 100, 100]},
    {"source_model": "best.pt", "source_class_id": 9, "raw_class_name": "Speed Limit 60",
     "confidence": 0.8, "bbox": [400, 400, 500, 500]},
]
fused = fuser.fuse(raw)
check("distinct signs stay distinct", len(fused) == 2)

# --- traffic light smoothing
tl = TrafficLightMonitor(window=5, min_state_frames=3)
check("starts UNKNOWN", tl.state == "UNKNOWN")
for _ in range(3):
    tl.update("traffic_light_red", 0.9)
check("3 red frames -> RED", tl.state == "RED")
tl.update(None, None)  # one bad frame must not flip
check("one bad frame does not flip state", tl.state == "RED")
tl.update("traffic_light_green", 0.95)
check("single green does not flip state", tl.state == "RED")
for _ in range(3):
    tl.update("traffic_light_green", 0.95)
check("sustained green flips state", tl.state == "GREEN")
check("yellow maps to UNKNOWN (no model)", tl.update("traffic_light_yellow", 0.9) in ("GREEN", "RED", "UNKNOWN"))

# --- dataset tools
rec = {"filename": "x.jpg", "width": "100", "height": "100", "objects": []}
check("empty voc record valid", validate_voc_record(rec) == [])
rec_bad = {"filename": None, "width": "100", "height": "100", "objects": []}
check("voc record missing filename flagged", validate_voc_record(rec_bad) != [])
idxs = list(range(30))
tr, va, te = stratified_split(
    [{"objects": [{"name": "traffic_sign"}]} for _ in range(30)], 0.2, 0.1, 42,
    key=lambda r: r["objects"][0]["name"])
check("splits disjoint (no leakage)", not (set(tr) & set(va)) and not (set(tr) & set(te)) and not (set(va) & set(te)))
check("splits cover all items", set(tr) | set(va) | set(te) == set(range(30)))

# --- xml parse of real data file
real = os.path.join(ROOT, "combined_training", "indian_dataset", "Annotations", "Annotations",
                    "Datacluster Traffic Sign (1).xml")
rec = read_voc_xml(real)
check("real VOC file parsed with size", rec["width"] == "3120" and len(rec["objects"]) == 1)

print()
print(f"PASSED: {len(PASS)}  FAILED: {len(FAIL)}")
if FAIL:
    print("FAILURES:", FAIL)
    sys.exit(1)
