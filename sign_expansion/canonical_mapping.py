"""
Canonical mapping and safe multi-model prediction fusion for Indian
traffic signs.

Design rules (see sign_expansion/results/phase2_run/IMPLEMENTATION_PLAN.md):
  - Original class IDs are preserved; every prediction carries
    source_model, source_class_id, raw_class_name, canonical_class.
  - Fusion never lets a low-confidence prediction override a stronger one.
  - Overlapping boxes from different models are the SAME physical sign
    (IoU-based dedupe), not two signs.
  - Below the unknown threshold a sign is UNKNOWN / UNRECOGNISED; it is
    never coerced into a random known class.
  - Normalisation fixes spelling so catalogues match, but it does NOT
    improve model accuracy.
"""

import json
import os
import threading

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CATALOGUE = os.path.join(ROOT, "config", "indian_traffic_signs.json")

UNKNOWN_ID = "unrecognised_sign"
UNKNOWN_NAME = "UNKNOWN / UNRECOGNISED SIGN"
UNKNOWN_ADVISORY = "SIGN NOT RECOGNISED"

# Spelling fixes used only for name lookup (class IDs are authoritative)
_NAME_FIXES = {
    "rright": "right",
    "curv": "curve",
    "pedestrain": "pedestrian",
    "rout": "route",
    "kmph": "km/h",
    "no_": "no ",
    "_over_taking": " overtaking",
    "_sign": " sign",
    "stop_sign": "stop sign",
}


def _normalize_name(name):
    """Light normalisation for lookup only. Does not change model outputs."""
    n = name.lower().strip()
    for bad, good in _NAME_FIXES.items():
        n = n.replace(bad, good)
    n = " ".join(n.split())
    return n


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


class SignCatalogue:
    """Loads and queries the versioned canonical catalogue."""

    def __init__(self, catalogue_path=None):
        self.path = catalogue_path or DEFAULT_CATALOGUE
        with open(self.path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.version = self.data.get("catalogue_version", "0")
        self._by_id = {s["canonical_id"]: s for s in self.data["signs"]}
        # (model, class_id) -> canonical entry
        self._by_model_class = {}
        for s in self.data["signs"]:
            for m in s["model_support"]:
                self._by_model_class[(m["source_model"], m["source_class_id"])] = s
        # normalised raw name -> canonical (first match wins; IDs take priority)
        self._by_name = {}
        for s in self.data["signs"]:
            self._by_name.setdefault(_normalize_name(s["canonical_name"]), s)
            for alt in s.get("alternative_names", []):
                self._by_name.setdefault(_normalize_name(alt), s)
        for model_meta in self.data.get("models", {}).values():
            for cid, raw in model_meta.get("classes", {}).items():
                n = _normalize_name(raw)
                if n not in self._by_name:
                    self._by_name[n] = None  # known-but-unmapped (e.g. '50 mph speed limit')

    def entry(self, canonical_id):
        return self._by_id.get(canonical_id)

    def resolve(self, source_model, source_class_id=None, raw_name=None):
        """Resolve a model prediction to a canonical entry (or None)."""
        if source_class_id is not None:
            hit = self._by_model_class.get((source_model, int(source_class_id)))
            if hit is not None:
                return hit
        if raw_name:
            return self._by_name.get(_normalize_name(raw_name))
        return None

    def advisory(self, canonical_id):
        e = self.entry(canonical_id)
        return e["advisory"] if e else UNKNOWN_ADVISORY

    def category(self, canonical_id):
        e = self.entry(canonical_id)
        return e["category"] if e else "unknown"

    def display_name(self, canonical_id):
        e = self.entry(canonical_id)
        return e["canonical_name"] if e else UNKNOWN_NAME


class FusedPrediction:
    """One physical sign after fusion."""

    __slots__ = ("canonical_class", "confidence", "bbox", "sources", "agreement", "conflicts",
                 "is_unknown", "display", "category", "advisory")

    def __init__(self, canonical_class, confidence, bbox, sources, agreement, conflicts, is_unknown):
        self.canonical_class = canonical_class
        self.confidence = confidence
        self.bbox = bbox
        self.sources = sources  # [{source_model, source_class_id, raw_class_name, confidence, bbox}]
        self.agreement = agreement  # canonical classes of all sources
        self.conflicts = conflicts  # sources whose class disagreed with the winner
        self.is_unknown = is_unknown

    @property
    def display_name(self):
        if self.is_unknown:
            return UNKNOWN_NAME
        return self.display

    def to_dict(self):
        return {
            "canonical_class": self.canonical_class,
            "confidence": round(self.confidence, 4),
            "bbox": [round(float(v), 1) for v in self.bbox],
            "sources": self.sources,
            "agreement": self.agreement,
            "conflicts": self.conflicts,
            "is_unknown": self.is_unknown,
        }


class DetectionFuser:
    """Fuses predictions from one or more YOLO models per frame.

    Parameters are configurable (see indian_traffic_sign_app.py):
      iou_dedupe_threshold : IoU at which two boxes are the same physical sign
      unknown_threshold    : below this confidence the sign becomes UNKNOWN
      conflicting_class_rule: when two models disagree on class for the same
                              physical sign, keep the higher-confidence
                              prediction and record the loser as a conflict.
                              A lower-confidence prediction NEVER overrides.
    """

    def __init__(self, catalogue=None, iou_dedupe_threshold=0.60, unknown_threshold=0.45):
        self.catalogue = catalogue or SignCatalogue()
        self.iou_dedupe_threshold = iou_dedupe_threshold
        self.unknown_threshold = unknown_threshold
        self._lock = threading.Lock()

    def fuse(self, raw_detections):
        """raw_detections: list of dicts from any number of models:
            {source_model, source_class_id, raw_class_name, confidence, bbox}
        Returns list[FusedPrediction].
        """
        with self._lock:
            resolved = []
            for d in raw_detections:
                entry = self.catalogue.resolve(d["source_model"], d.get("source_class_id"), d.get("raw_class_name"))
                resolved.append({
                    **d,
                    "canonical_id": entry["canonical_id"] if entry else None,
                    "canonical_name": entry["canonical_name"] if entry else None,
                    "category": entry["category"] if entry else "unknown",
                    "advisory": entry["advisory"] if entry else UNKNOWN_ADVISORY,
                })

            # Sort by confidence: strong predictions first; weak ones never win.
            resolved.sort(key=lambda d: d["confidence"], reverse=True)

            clusters = []
            for d in resolved:
                placed = False
                for cluster in clusters:
                    # Same physical sign if boxes overlap strongly AND classes agree,
                    # or boxes overlap extremely strongly (>= 0.85) regardless of class
                    # (same sign, one model may mislabel at low conf).
                    if d["canonical_id"] == cluster["canonical_id"] and iou(d["bbox"], cluster["bbox"]) >= self.iou_dedupe_threshold:
                        cluster["members"].append(d)
                        placed = True
                        break
                    if iou(d["bbox"], cluster["bbox"]) >= 0.85:
                        cluster["members"].append(d)
                        cluster["conflicting_members"].append(d)
                        placed = True
                        break
                if not placed:
                    clusters.append({
                        "canonical_id": d["canonical_id"],
                        "canonical_name": d["canonical_name"],
                        "category": d["category"],
                        "advisory": d["advisory"],
                        "bbox": d["bbox"],
                        "members": [d],
                        "conflicting_members": [],
                    })

            fused = []
            for c in clusters:
                members = c["members"] + c["conflicting_members"]
                winner = max(members, key=lambda d: d["confidence"])
                is_unknown = (
                    winner["confidence"] < self.unknown_threshold
                    or c["canonical_id"] in (None, UNKNOWN_ID)
                )
                if is_unknown:
                    canonical_id_out = None
                    display = UNKNOWN_NAME
                    category = "unknown"
                    advisory = UNKNOWN_ADVISORY
                else:
                    canonical_id_out = c["canonical_id"]
                    display = c["canonical_name"]
                    category = c["category"]
                    advisory = c["advisory"]
                fp = FusedPrediction(
                    canonical_class=canonical_id_out,
                    confidence=winner["confidence"],
                    bbox=winner["bbox"],
                    sources=[
                        {k: d[k] for k in ("source_model", "source_class_id", "raw_class_name", "confidence")}
                        for d in members
                    ],
                    agreement=sorted({d["canonical_id"] for d in members if d["canonical_id"]}),
                    conflicts=sorted({d["canonical_id"] for d in c["conflicting_members"] if d["canonical_id"]}),
                    is_unknown=is_unknown,
                )
                fp.display = display
                fp.category = category
                fp.advisory = advisory
                fused.append(fp)
            fused.sort(key=lambda f: f.confidence, reverse=True)
            return fused
