"""
Traffic-light state tests (the traffic-light issue ONLY).

Covers: red, green, yellow (reserved -> UNKNOWN), no detection, low
confidence, conflicting detections, state transitions, UNKNOWN handling,
temporal stability (no single-frame flips), Traffic_signal warning-sign
exclusion, and trained-weight preservation.

Run:  python sign_expansion/tests_traffic_light.py
"""

import hashlib
import os
import sys
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "sign_expansion"))

from traffic_light import (  # noqa: E402
    ADVISORIES,
    ALL_STATES,
    LIGHT_CANONICAL_CLASSES,
    STATE_GREEN,
    STATE_RED,
    STATE_UNKNOWN,
    STATE_YELLOW,
    TrafficLightMonitor,
    pick_strongest_light,
)

PASS, FAIL = [], []


def check(name, cond):
    (PASS if cond else FAIL).append(name)
    print(("PASS" if cond else "FAIL"), name)


def pred(canonical_class, confidence, is_unknown=False):
    return SimpleNamespace(canonical_class=canonical_class, confidence=confidence,
                           is_unknown=is_unknown)


# ------------------------------------------------------------------ states
check("four distinct states exist",
      len(set(ALL_STATES)) == 4 and STATE_RED != STATE_GREEN != STATE_YELLOW != STATE_UNKNOWN)
check("every state has an advisory", all(s in ADVISORIES for s in ALL_STATES))
check("yellow recognition is not claimed", __import__("traffic_light").YELLOW_RECOGNITION_AVAILABLE is False)

# ------------------------------------------------------------------ red
tl = TrafficLightMonitor()
check("starts UNKNOWN", tl.state == STATE_UNKNOWN)
tl.update("traffic_light_red", 0.9)
check("single red frame does not flip (temporal stability)", tl.state == STATE_UNKNOWN)
tl.update("traffic_light_red", 0.9)
check("two red frames do not flip", tl.state == STATE_UNKNOWN)
tl.update("traffic_light_red", 0.9)
check("three red frames -> RED", tl.state == STATE_RED)

# ------------------------------------------------------------------ green
tl = TrafficLightMonitor()
for _ in range(3):
    tl.update("traffic_light_green", 0.9)
check("three green frames -> GREEN", tl.state == STATE_GREEN)
tl.update("traffic_light_red", 0.9)
check("single red while GREEN does not flip", tl.state == STATE_GREEN)
for _ in range(2):
    tl.update("traffic_light_red", 0.9)
check("two red frames do not flip GREEN (needs majority)", tl.state == STATE_GREEN)
tl.update("traffic_light_red", 0.9)
check("fourth red out-votes green in window -> RED", tl.state == STATE_RED)

# ------------------------------------------------------------------ yellow (reserved)
tl = TrafficLightMonitor()
for _ in range(7):
    tl.update("traffic_light_yellow", 0.99)  # class exists in catalogue, NOT in any model
check("yellow detections never display YELLOW", tl.state != STATE_YELLOW)
check("sustained yellow-only input decays to UNKNOWN", tl.state == STATE_UNKNOWN)

tl = TrafficLightMonitor()
for _ in range(3):
    tl.update("traffic_light_red", 0.9)
for _ in range(7):
    tl.update("traffic_light_yellow", 0.99)
check("yellow after RED never shows YELLOW", tl.state != STATE_YELLOW)

# ------------------------------------------------------------------ no detection
tl = TrafficLightMonitor()
for _ in range(3):
    tl.update("traffic_light_red", 0.9)
check("RED displayed before misses", tl.state == STATE_RED)
tl.update(None, None)
check("one missed frame does not flip (temporal stability)", tl.state == STATE_RED)
tl.update(None, None)
check("two consecutive missed frames -> UNKNOWN (decay)", tl.state == STATE_UNKNOWN)
tl.update("traffic_light_green", 0.9)
check("UNKNOWN does not become GREEN from one frame", tl.state == STATE_UNKNOWN)
check("UNKNOWN can never become YELLOW by itself", tl.state != STATE_YELLOW)
tl.update("traffic_light_green", 0.95)   # 2nd green vote since decay
check("two green votes after decay are not enough (full evidence required)", tl.state == STATE_UNKNOWN)
tl.update("traffic_light_green", 0.95)   # 3rd green vote since decay
check("three green votes after decay -> GREEN", tl.state == STATE_GREEN)

# ------------------------------------------------------------------ low confidence
tl = TrafficLightMonitor()
tl.update("traffic_light_red", 0.30)   # below default 0.45 threshold
tl.update("traffic_light_red", 0.30)
tl.update("traffic_light_red", 0.30)
check("low-confidence red never becomes RED (default threshold)", tl.state == STATE_UNKNOWN)

tl = TrafficLightMonitor()
tl.update("traffic_light_red", 0.40)
tl.update("traffic_light_red", 0.44)
check("one below-threshold + one at-threshold frame still UNKNOWN", tl.state == STATE_UNKNOWN)

tl = TrafficLightMonitor()
for _ in range(3):
    tl.update("traffic_light_red", 0.40, unknown_threshold=0.35)
check("explicit lower threshold accepts valid red", tl.state == STATE_RED)

# ------------------------------------------------------------------ conflicting detections
r = pick_strongest_light([pred("traffic_light_red", 0.40), pred("traffic_light_green", 0.80)])
check("conflicting red+green: higher confidence wins", r == ("traffic_light_green", 0.80))
r = pick_strongest_light([pred("traffic_light_green", 0.60), pred("traffic_light_red", 0.60)])
check("exact tie resolves to first (deterministic)", r == ("traffic_light_green", 0.60))
r = pick_strongest_light([pred("traffic_light_red", 0.60), pred("traffic_light_green", 0.90)])
tl = TrafficLightMonitor()
for _ in range(5):
    tl.update(*r)
check("conflict winner drives state", tl.state == STATE_GREEN)

# ------------------------------------------------------------------ warning-sign exclusion
r = pick_strongest_light([pred("traffic_signals_ahead", 0.99)])
check("Traffic_signal WARNING sign is never a light state", r == (None, 0.0))
r = pick_strongest_light([pred("traffic_signals_ahead", 0.99), pred("traffic_light_red", 0.55)])
check("warning sign ignored, real red still selected", r == ("traffic_light_red", 0.55))
r = pick_strongest_light([pred("traffic_signals_ahead", 0.99), pred("traffic_light_red", 0.55, is_unknown=True)])
check("UNKNOWN-flagged light prediction never qualifies", r == (None, 0.0))
r = pick_strongest_light([pred("stop", 0.99), pred("speed_limit_60", 0.95)])
check("non-light signs never qualify", r == (None, 0.0))
r = pick_strongest_light([])
check("no detections -> (None, 0.0)", r == (None, 0.0))
check("warning sign canonical id not in light classes", "traffic_signals_ahead" not in LIGHT_CANONICAL_CLASSES)

# catalogue: infra class 47 'Traffic_signal' must resolve to the warning sign,
# never to a light state
from canonical_mapping import SignCatalogue  # noqa: E402
cat = SignCatalogue()
e = cat.resolve("infrastructure_detector.pt", 47, "Traffic_signal")
check("infra 47 maps to warning sign, not a light",
      e is not None and e["canonical_id"] == "traffic_signals_ahead"
      and e["canonical_id"] not in LIGHT_CANONICAL_CLASSES)
e = cat.resolve("best.pt", 0, "Green Light")
check("best.pt class 0 -> green light", e and e["canonical_id"] == "traffic_light_green")
e = cat.resolve("best.pt", 1, "Red Light")
check("best.pt class 1 -> red light", e and e["canonical_id"] == "traffic_light_red")

# ------------------------------------------------------------------ transitions
tl = TrafficLightMonitor()
seq = [
    (("traffic_light_red", 0.9), 3, STATE_RED, "UNKNOWN->RED"),
    (("traffic_light_green", 0.95), 4, STATE_GREEN, "RED->GREEN (strict window majority)"),
    (("traffic_light_red", 0.95), 4, STATE_RED, "GREEN->RED (strict window majority)"),
    ((None, None), 2, STATE_UNKNOWN, "RED->UNKNOWN (light lost)"),
]
ok = True
for (canon, conf), frames, expected, label in seq:
    for _ in range(frames):
        tl.update(canon, conf)
    if tl.state != expected:
        ok = False
        print("   transition failed:", label, "->", tl.state)
check("full transition sequence UNKNOWN->RED->GREEN->RED->UNKNOWN", ok)

# ------------------------------------------------------------------ weights preserved
protected = os.path.join(ROOT, "sign_expansion", "backups", "PROTECTED_WEIGHTS_SHA256.txt")
with open(protected, "r", encoding="utf-8") as f:
    expected_hashes = {}
    for line in f:
        parts = line.split()
        if len(parts) == 2:
            expected_hashes[os.path.basename(parts[1])] = parts[0]
ok = True
for name, expected in expected_hashes.items():
    path = os.path.join(ROOT, name)
    if not os.path.exists(path):
        continue  # optional weight for this machine
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    if h.hexdigest().upper() != expected.upper():
        ok = False
        print("   WEIGHT CHANGED:", name)
check("trained weights bit-identical to protected hashes", ok)

print()
print(f"PASSED: {len(PASS)}  FAILED: {len(FAIL)}")
if FAIL:
    print("FAILURES:", FAIL)
    sys.exit(1)
