"""
Traffic-light state detection — a SEPARATE task from sign detection.

Facts (verified 2026-11-16 by loading best.pt with ultralytics):
  - best.pt classes: 0=Green Light, 1=Red Light. NO Yellow/Amber class.
  - infrastructure_detector.pt class 47 = "Traffic_signal" is the WARNING
    SIGN "Traffic Signals Ahead" (canonical id: traffic_signals_ahead,
    category B_warning) — never a live light state.
  - No labelled traffic-light dataset exists in this project.

Consequences enforced by this module:
  - YELLOW recognition is NOT claimed and CANNOT be produced by the current
    models. STATE_YELLOW is kept as a distinct state (interface stability)
    but NO code path can produce it until a validated yellow-recognition
    method with labelled data exists (YELLOW_RECOGNITION_AVAILABLE = False).
    Until then a light that cannot be classified is UNKNOWN.
  - UNKNOWN / missing / low-confidence detections NEVER become YELLOW (or
    any other state); they can only decay the display to UNKNOWN.

Temporal smoothing:
  - A state must be observed in >= min_state_frames of the last `window`
    processed frames before it is displayed; a single noisy frame cannot
    change the state (MIN_STATE_FRAMES / window).
  - A DIFFERENT state must hold a strict majority of non-UNKNOWN votes in
    the window to replace the currently displayed state (no tie flips).
  - Conversely, after unknown_confirm_frames CONSECUTIVE missing/low-conf
    frames the display decays to UNKNOWN — the old bug where a latched
    state stayed on screen forever with no detection at all is fixed.

BUG FIXED 2026-11-16 (sticky state): TrafficLightMonitor.update() used to
return self._displayed whenever no non-UNKNOWN candidate reached
min_state_frames, so once RED/GREEN was latched it was displayed forever
even when every subsequent frame had no light detection. Verified before
the fix: 3 RED frames then 10 empty frames still displayed "RED".
"""

from collections import Counter, deque

STATE_RED = "RED"
STATE_GREEN = "GREEN"
STATE_YELLOW = "YELLOW / AMBER"
STATE_UNKNOWN = "UNKNOWN"

# All four states remain distinct (see tests_traffic_light.py).
ALL_STATES = (STATE_RED, STATE_GREEN, STATE_YELLOW, STATE_UNKNOWN)

# Honest capability flag: no validated yellow detector / labelled yellow
# data exists in this project. YELLOW stays reserved; do not flip this
# flag until a yellow-recognition method has been validated on data.
YELLOW_RECOGNITION_AVAILABLE = False

ADVISORIES = {
    STATE_RED: "STOP",
    STATE_YELLOW: "CAUTION - PREPARE TO STOP WHERE SAFE",
    STATE_GREEN: "PROCEED ONLY IF INTERSECTION AND PATH ARE CLEAR",
    STATE_UNKNOWN: "NO RELIABLE SIGNAL STATE",
}

# canonical_id -> raw state. ONLY classes that actually exist in best.pt.
# Deliberately NO traffic_light_yellow entry: the trained model cannot
# produce it, so inventing a mapping would fabricate predictions.
_CANON_TO_STATE = {
    "traffic_light_red": STATE_RED,
    "traffic_light_green": STATE_GREEN,
}

# Canonical classes that a live traffic-light state may be derived from.
# The infrastructure model's "Traffic_signal" warning sign
# (traffic_signals_ahead) is intentionally NOT in this set.
LIGHT_CANONICAL_CLASSES = frozenset(_CANON_TO_STATE)


def pick_strongest_light(detections):
    """Select the strongest valid traffic-light detection for one frame.

    detections: iterable of fused predictions (objects exposing
    .canonical_class and .confidence, and optionally .is_unknown).

    Rules:
      - Only canonical classes in LIGHT_CANONICAL_CLASSES qualify; warning
        signs (e.g. traffic_signals_ahead) and other signs never qualify.
      - UNKNOWN predictions never qualify.
      - Highest confidence wins; on exact ties the earlier detection wins
        (deterministic, fusion sorts strongest-first).
    Returns (canonical_class, confidence) or (None, 0.0) if no valid light.
    """
    best = None
    best_conf = 0.0
    for p in detections or ():
        canon = getattr(p, "canonical_class", None)
        if canon not in LIGHT_CANONICAL_CLASSES:
            continue
        if getattr(p, "is_unknown", False):
            continue
        conf = getattr(p, "confidence", None)
        if not isinstance(conf, (int, float)) or conf < 0.0:
            continue
        if best is None or conf > best_conf:
            best, best_conf = canon, float(conf)
    return (best, best_conf) if best is not None else (None, 0.0)


class TrafficLightMonitor:
    """Smooths per-frame light detections into a stable displayed state.

    A state must be observed in >= min_state_frames of the last `window`
    processed frames before it replaces the displayed state; this stops one
    noisy detection from flickering the UI. A different state must hold a
    strict majority over the displayed state. After unknown_confirm_frames
    consecutive missing/low-confidence frames the display returns to
    UNKNOWN (it never invents a state, and never invents YELLOW).
    """

    def __init__(self, window=7, min_state_frames=3, unknown_confirm_frames=2):
        self.window = window
        self.min_state_frames = min_state_frames
        self.unknown_confirm_frames = unknown_confirm_frames
        self._votes = deque(maxlen=window)
        self._unknown_streak = 0
        self._displayed = STATE_UNKNOWN

    def update(self, canonical_class, confidence, unknown_threshold=0.45):
        """Feed one frame's strongest light detection (or None if no light)."""
        if canonical_class is None or confidence is None or confidence < unknown_threshold:
            vote = STATE_UNKNOWN
        else:
            vote = _CANON_TO_STATE.get(canonical_class, STATE_UNKNOWN)
        self._votes.append(vote)

        if vote == STATE_UNKNOWN:
            self._unknown_streak += 1
        else:
            self._unknown_streak = 0

        counts = Counter(v for v in self._votes if v != STATE_UNKNOWN)
        candidate, n = counts.most_common(1)[0] if counts else (None, 0)

        if vote == STATE_UNKNOWN and self._unknown_streak >= self.unknown_confirm_frames:
            # Sustained missing / low-confidence evidence: the light is not
            # being detected, so the display decays to UNKNOWN and the stale
            # vote history is invalidated (it cannot re-latch a state later).
            # UNKNOWN can never become YELLOW (or RED/GREEN) by itself; a
            # real state must be re-detected with min_state_frames votes.
            self._votes.clear()
            self._displayed = STATE_UNKNOWN
        elif candidate is not None and n >= self.min_state_frames:
            current_count = counts.get(self._displayed, 0) if self._displayed != STATE_UNKNOWN else 0
            if self._displayed == STATE_UNKNOWN or candidate == self._displayed or n > current_count:
                self._displayed = candidate

        return self._displayed

    @property
    def state(self):
        return self._displayed

    @property
    def advisory(self):
        return ADVISORIES[self._displayed]

    def reset(self):
        self._votes.clear()
        self._unknown_streak = 0
        self._displayed = STATE_UNKNOWN
