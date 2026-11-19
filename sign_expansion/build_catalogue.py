"""
Build the canonical Indian Traffic Sign catalogue:
  config/indian_traffic_signs.json  (versioned)

Sources used:
  - Verified class arrays of the two existing models (loaded 2026-11-16, see
    sign_expansion/phase2_audit/verify_models_report.json).
  - Indian road-sign standards reference: MoRTH "Code of Practice for Road
    Signs" (IRC:67-2012, 3rd revision) which defines three categories:
    Mandatory/Regulatory (A), Cautionary/Warning (B), Informatory (C).
    Per-sign verification_status flags which entries are well-standardised
    versus merely proposed and needing verification. NO invented official
    names or sign codes.

The mapping preserves every model's ORIGINAL class IDs; canonical entries
reference them as {source_model, source_class_id}.
"""

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "config", "indian_traffic_signs.json")

CAT_A = "A_regulatory"
CAT_B = "B_warning"
CAT_C = "C_informatory"
CAT_D = "D_speed_limit"
CAT_E = "E_traffic_light"
CAT_U = "U_unrecognised"

# verification levels
STD = "standardized (MoRTH/IRC:67-2012 - verify exact variant)"
PROP = "proposed - not yet verified against IRC:67-2012"

# training data status
TD_MODEL_ONLY = "model-supported (original training dataset missing from project)"
TD_NONE = "no labelled training data in project"
TD_MODEL = "model-supported with measured performance"


def sign(cid, name, cat, alts, adv, models, verify=STD, td=TD_NONE, dup=None, note=None):
    return {
        "canonical_id": cid,
        "canonical_name": name,
        "category": cat,
        "alternative_names": alts,
        "advisory": adv,
        "model_support": models,  # list of {source_model, source_class_id, source_class_name}
        "training_data_status": td,
        "verification_status": verify,
        "duplicate_of": dup,
        "note": note,
    }


def ms(model, cid):
    return {"source_model": model, "source_class_id": cid}


B14 = "best.pt"
INF = "infrastructure_detector.pt"

SIGNS = []

# ---------------------------------------------------------------- A. Regulatory
SIGNS += [
    sign("stop", "Stop", CAT_A, ["stop sign"], "STOP",
         [ms(B14, 13), ms(INF, 33), ms(INF, 42)],
         td=TD_MODEL, note="Duplicate infra classes 33/42 map to same canonical sign."),
    sign("give_way", "Give Way", CAT_A, ["yield"], "GIVE WAY / SLOW DOWN",
         [ms(INF, 16)], td=TD_MODEL_ONLY),
    sign("no_entry", "No Entry", CAT_A, ["entry prohibited"], "DO NOT ENTER",
         [ms(INF, 25)], td=TD_MODEL_ONLY),
    sign("one_way", "One Way", CAT_A, ["one way traffic"], "FOLLOW ONE-WAY DIRECTION",
         [], note="Distinct from No Entry; verify variant."),
    sign("no_parking", "No Parking", CAT_A, ["parking prohibited"], "DO NOT PARK",
         []),
    sign("no_stopping", "No Stopping", CAT_A, ["stopping prohibited"], "DO NOT STOP",
         []),
    sign("height_limit", "Maximum Height", CAT_A, ["height limit", "height restriction"],
         "DO NOT PROCEED IF VEHICLE EXCEEDS HEIGHT", [ms(INF, 19)], td=TD_MODEL_ONLY),
    sign("width_limit", "Maximum Width", CAT_A, ["width limit"], "DO NOT PROCEED IF VEHICLE EXCEEDS WIDTH", []),
    sign("weight_limit", "Maximum Weight", CAT_A, ["axle load limit", "weight restriction"],
         "DO NOT PROCEED IF VEHICLE EXCEEDS WEIGHT", []),
    sign("no_overtaking", "No Overtaking", CAT_A, ["overtaking prohibited"],
         "DO NOT OVERTAKE", [ms(INF, 26)], td=TD_MODEL_ONLY),
    sign("no_truck_overtaking", "No Overtaking By Trucks", CAT_A,
         ["overtaking by trucks is prohibited"], "TRUCKS MUST NOT OVERTAKE",
         [ms(INF, 28)], td=TD_MODEL_ONLY),
    sign("no_horn", "No Horn", CAT_A, ["horn prohibited", "silence zone"], "DO NOT HORN", []),
    sign("no_u_turn", "No U-Turn", CAT_A, ["u-turn prohibited"], "NO U-TURN", []),
    sign("no_right_turn", "No Right Turn", CAT_A, [], "DO NOT TURN RIGHT", []),
    sign("no_left_turn", "No Left Turn", CAT_A, [], "DO NOT TURN LEFT", []),
    sign("keep_left", "Keep Left", CAT_A, [], "KEEP LEFT", [ms(INF, 20)], td=TD_MODEL_ONLY),
    sign("keep_right", "Keep Right", CAT_A, [], "KEEP RIGHT", [ms(INF, 21)], td=TD_MODEL_ONLY),
    sign("compulsory_go_straight", "Compulsory Go Straight", CAT_A,
         ["straight ahead only", "one way ahead"], "GO STRAIGHT", [ms(INF, 43)], td=TD_MODEL_ONLY),
    sign("go_straight_or_turn_left", "Go Straight Or Turn Left", CAT_A, [],
         "GO STRAIGHT OR TURN LEFT ONLY", [ms(INF, 18)], td=TD_MODEL_ONLY),
    sign("go_straight_or_turn_right", "Go Straight Or Turn Right", CAT_A, [],
         "GO STRAIGHT OR TURN RIGHT ONLY", [ms(INF, 17)], td=TD_MODEL_ONLY),
    sign("turn_left_ahead", "Turn Left Ahead", CAT_A, ["compulsory turn left ahead"],
         "TURN LEFT AHEAD", [ms(INF, 49)], td=TD_MODEL_ONLY),
    sign("turn_right_ahead", "Turn Right Ahead", CAT_A, ["compulsory turn right ahead"],
         "TURN RIGHT AHEAD", [ms(INF, 50)], td=TD_MODEL_ONLY),
    sign("compulsory_roundabout", "Compulsory Roundabout", CAT_A,
         ["roundabout", "go around"], "GIVE WAY AT ROUNDABOUT", [ms(INF, 32)], td=TD_MODEL_ONLY),
    sign("end_of_restrictions", "End Of All Restrictions", CAT_A,
         ["end of all speed and passing limits"], "RESTRICTIONS END",
         [ms(INF, 12)], td=TD_MODEL_ONLY),
    sign("pedestrian_prohibited", "Pedestrian Prohibited", CAT_A, [], "PEDESTRIANS MUST NOT ENTER", []),
    sign("bullock_cart_prohibited", "Bullock Cart And Hand Cart Prohibited", CAT_A,
         ["bullock and hand cart prohibited"], "BULLOCK/HAND CARTS NOT PERMITTED", [ms(INF, 6)], td=TD_MODEL_ONLY),
    sign("cycle_prohibited", "Cycle Prohibited", CAT_A, ["bicycles prohibited"],
         "CYCLES NOT PERMITTED", [ms(INF, 9)], td=TD_MODEL_ONLY),
    sign("tractor_prohibited", "Tractor Prohibited", CAT_A, [], "TRACTORS NOT PERMITTED", [ms(INF, 45)], td=TD_MODEL_ONLY),
    sign("two_wheeler_prohibited", "Two Wheeler Prohibited", CAT_A, [], "TWO-WHEELERS NOT PERMITTED", [ms(INF, 51)], td=TD_MODEL_ONLY),
    sign("truck_prohibited", "Truck Prohibited", CAT_A, ["trucks prohibited"],
         "TRUCKS NOT PERMITTED", [ms(INF, 48)], td=TD_MODEL_ONLY),
]

# ---------------------------------------------------------------- B. Warning
SIGNS += [
    sign("left_curve", "Left Curve Ahead", CAT_B,
         ["dangerous left curve ahead", "left hand curve"], "SLOW DOWN - LEFT CURVE AHEAD",
         [ms(INF, 10), ms(INF, 22)], td=TD_MODEL_ONLY,
         note="Duplicate infra classes 10/22."),
    sign("right_curve", "Right Curve Ahead", CAT_B,
         ["dangerous right curve ahead", "right hand curve"], "SLOW DOWN - RIGHT CURVE AHEAD",
         [ms(INF, 11), ms(INF, 31)], td=TD_MODEL_ONLY,
         note="Infra class 11 has typo 'Rright'; duplicate 11/31."),
    sign("hairpin_bend", "Hairpin Bend", CAT_B, [], "SLOW DOWN - HAIRPIN BEND", [], verify=PROP),
    sign("steep_ascent", "Steep Ascent", CAT_B, ["steep hill upwards"], "USE LOW GEAR", []),
    sign("steep_descent", "Steep Descent", CAT_B, ["steep hill downwards"], "USE LOW GEAR - DO NOT RIDE BRAKES", []),
    sign("narrow_road", "Narrow Road", CAT_B, ["road narrows"], "SLOW DOWN - NARROW ROAD", []),
    sign("road_narrows_right", "Road Narrows On Right", CAT_B, ["road narrows on right"],
         "SLOW DOWN - ROAD NARROWS AHEAD", [ms(INF, 0)], td=TD_MODEL_ONLY),
    sign("narrow_bridge", "Narrow Bridge", CAT_B, ["narrow bridge ahead"],
         "SLOW DOWN - NARROW BRIDGE AHEAD", [ms(INF, 24)], td=TD_MODEL_ONLY),
    sign("slippery_road", "Slippery Road", CAT_B, ["slippery road ahead"],
         "REDUCE SPEED", [ms(INF, 37)], td=TD_MODEL_ONLY),
    sign("uneven_road", "Uneven Road", CAT_B, [], "REDUCE SPEED", [ms(INF, 52)], td=TD_MODEL_ONLY),
    sign("hump_ahead", "Hump / Speed Breaker Ahead", CAT_B, ["speed breaker", "hump or rough road"],
         "REDUCE SPEED - SPEED BREAKER", [], verify=PROP),
    sign("school_ahead", "School Ahead", CAT_B, [], "SLOW DOWN - WATCH FOR CHILDREN", [ms(INF, 34)], td=TD_MODEL_ONLY),
    sign("children_ahead", "Children Ahead", CAT_B, ["beware of children"],
         "SLOW DOWN - WATCH FOR CHILDREN", [ms(INF, 4)], td=TD_MODEL_ONLY),
    sign("pedestrian_crossing", "Pedestrian Crossing", CAT_B, ["pedestrian crossing ahead"],
         "WATCH FOR PEDESTRIANS", [ms(INF, 29), ms(INF, 30)], td=TD_MODEL_ONLY,
         note="Infra class 29 has typo 'Pedestrain'; duplicate 29/30."),
    sign("cycle_crossing", "Cycle Crossing Ahead", CAT_B, ["cycle route ahead warning"],
         "WATCH FOR CYCLISTS", [ms(INF, 8)], td=TD_MODEL_ONLY,
         note="Model class is a cycle-route warning; exact variant to verify."),
    sign("cattle_ahead", "Cattle Ahead", CAT_B, ["loose cattle"], "SLOW DOWN - LOOSE CATTLE", [], verify=PROP),
    sign("men_at_work", "Men At Work", CAT_B, ["road works", "road work ahead"], "CAUTION - ROAD WORKS", [], verify=PROP),
    sign("falling_rocks", "Falling Rocks", CAT_B, ["falling rock zone"], "CAUTION - FALLING ROCKS", [], verify=PROP),
    sign("cross_roads", "Cross Roads Ahead", CAT_B, ["major cross road"], "SLOW DOWN - CROSS ROADS", [], verify=PROP),
    sign("side_road_right", "Side Road Right", CAT_B, ["side road junction right"],
         "SLOW DOWN - JUNCTION AHEAD", [ms(INF, 35)], td=TD_MODEL_ONLY),
    sign("roundabout_ahead", "Roundabout Ahead", CAT_B, [], "SLOW DOWN - ROUNDABOUT AHEAD", [], verify=PROP),
    sign("traffic_signals_ahead", "Traffic Signals Ahead", CAT_B, ["traffic signal ahead"],
         "SLOW DOWN - SIGNALS AHEAD", [ms(INF, 47)], td=TD_MODEL_ONLY,
         note="Class name 'Traffic_signal' is ambiguous; treated as the warning sign, NOT a live light state."),
    sign("railway_crossing", "Railway Crossing Ahead", CAT_B, ["level crossing"],
         "SLOW DOWN - RAILWAY CROSSING", [], verify=PROP),
    sign("dangerous_intersection", "Dangerous Intersection", CAT_B, [],
         "SLOW DOWN - DANGEROUS INTERSECTION", [], verify=PROP),
    sign("zigzag_left", "Zigzag Road Left", CAT_B, ["left zig zag traffic"],
         "SLOW DOWN - ZIGZAG AHEAD", [ms(INF, 23)], td=TD_MODEL_ONLY),
    sign("chevron", "Chevron Board", CAT_B, ["single chevron", "direction chevron"],
         "FOLLOW CHEVRON DIRECTION", [ms(INF, 36)], td=TD_MODEL_ONLY),
    sign("object_hazard_right", "Object Hazard Right", CAT_B, [], "CAUTION - OBSTRUCTION", [ms(INF, 27)], td=TD_MODEL_ONLY),
]

# ---------------------------------------------------------------- C. Informatory
SIGNS += [
    sign("parking", "Parking", CAT_C, ["parking place"], "PARKING AREA", []),
    sign("bus_stop", "Bus Stop", CAT_C, [], "BUS STOP AHEAD", [ms(INF, 7)], td=TD_MODEL_ONLY),
    sign("hospital", "Hospital", CAT_C, ["first aid post"], "HOSPITAL / FIRST AID NEARBY", []),
    sign("first_aid", "First Aid", CAT_C, [], "FIRST AID NEARBY", []),
    sign("petrol_pump", "Petrol Pump", CAT_C, ["filling station"], "PETROL PUMP AHEAD", [ms(INF, 15)], td=TD_MODEL_ONLY),
    sign("restaurant", "Restaurant", CAT_C, [], "RESTAURANT AHEAD", []),
    sign("telephone", "Telephone", CAT_C, [], "TELEPHONE AHEAD", []),
    sign("toilet", "Public Toilet", CAT_C, ["toilet"], "TOILET AHEAD", [ms(INF, 44)], td=TD_MODEL_ONLY),
    sign("railway_station", "Railway Station", CAT_C, [], "RAILWAY STATION AHEAD", []),
    sign("airport", "Airport", CAT_C, [], "AIRPORT AHEAD", []),
    sign("advance_direction", "Advance Direction Sign", CAT_C, ["advance direction"],
         "ROUTE INFORMATION", [ms(INF, 2)], td=TD_MODEL_ONLY),
    sign("stack_advance_direction", "Stack Type Advance Direction Sign", CAT_C,
         ["stack type advance direction sign"], "ROUTE INFORMATION", [ms(INF, 40)], td=TD_MODEL_ONLY),
    sign("state_highway_marker", "State Highway Route Marker", CAT_C, [],
         "STATE HIGHWAY ROUTE MARKER", [ms(INF, 41)], td=TD_MODEL_ONLY),
    sign("expressway_route_marking", "Expressway Route Marking", CAT_C,
         ["expressway rout marking"], "EXPRESSWAY ROUTE MARKER", [ms(INF, 13)], td=TD_MODEL_ONLY),
    sign("built_up_area", "Built Up Area", CAT_C, [], "ENTERING BUILT-UP AREA", [ms(INF, 5)], td=TD_MODEL_ONLY),
    sign("expressway_entry_ramp", "Expressway Entry Ramp", CAT_C, ["fs 31 entry ramp for expressway"],
         "EXPRESSWAY ENTRY AHEAD", [ms(INF, 14)], td=TD_MODEL_ONLY),
]

# ---------------------------------------------------------------- D. Speed limits
SPEEDS = [(20, 5, [38, 55]), (30, 6, [39]), (40, 7, []), (50, 8, []),
          (60, 9, []), (70, 10, []), (80, 11, [56]), (90, 12, []),
          (100, 2, []), (110, 3, []), (120, 4, [])]
for kmh, bid, infra_ids in SPEEDS:
    SIGNS.append(
        sign(
            f"speed_limit_{kmh}", f"Speed Limit {kmh}", CAT_D,
            [f"speed limit {kmh} kmph", f"speed limit {kmh} km/h"],
            f"OBSERVE {kmh} KM/H LIMIT",
            [ms(B14, bid)] + [ms(INF, i) for i in infra_ids],
            td=TD_MODEL if infra_ids else TD_MODEL_ONLY,
        )
    )
SIGNS.append(sign(
    "speed_limit_unspecified", "Speed Limit (value unspecified)", CAT_D,
    ["speed limit"], "DISPLAY THE DETECTED SPEED LIMIT (OCR REQUIRED)",
    [ms(INF, 54)], td=TD_MODEL,
    note="Generic class; number must be read by OCR. Infra class 1 '50 mph speed limit' "
         "appears to be a non-Indian variant and is NOT mapped to a km/h class."))

# ---------------------------------------------------------------- E. Traffic lights
SIGNS += [
    sign("traffic_light_red", "Red Traffic Light", CAT_E, ["red light"], "STOP",
         [ms(B14, 1)], td=TD_MODEL_ONLY),
    sign("traffic_light_green", "Green Traffic Light", CAT_E, ["green light"],
         "PROCEED ONLY IF INTERSECTION AND PATH ARE CLEAR", [ms(B14, 0)], td=TD_MODEL_ONLY),
    sign("traffic_light_yellow", "Yellow / Amber Traffic Light", CAT_E, ["yellow light", "amber light"],
         "CAUTION - PREPARE TO STOP WHERE SAFE", [],
         note="NO trained model class and NO labelled data. Interface exists; recognition NOT claimed."),
]

# ---------------------------------------------------------------- U. Unrecognised bucket
SIGNS.append(sign("unrecognised_sign", "Unrecognised Sign", CAT_U,
                  ["other", "traffic sign not visibles", "unknown"],
                  "SIGN NOT RECOGNISED",
                  [ms(INF, 46), ms(INF, 53)], td=TD_MODEL_ONLY,
                  note="Model 'other'/'not visible' classes deliberately bucketed as UNKNOWN, never coerced to a real sign."))

# ---------------------------------------------------------------- assemble
MODELS_META = {
    B14: {
        "path": "best.pt",
        "sha256": "C62E044A49BC095EDB3644794C7194062EBBE64D398A8141E41F29BE344AE721",
        "class_count": 14,
        "classes": {"0": "Green Light", "1": "Red Light", "2": "Speed Limit 100",
                    "3": "Speed Limit 110", "4": "Speed Limit 120", "5": "Speed Limit 20",
                    "6": "Speed Limit 30", "7": "Speed Limit 40", "8": "Speed Limit 50",
                    "9": "Speed Limit 60", "10": "Speed Limit 70", "11": "Speed Limit 80",
                    "12": "Speed Limit 90", "13": "Stop"},
        "note": "best_14class_backup.pt and road-sign-detection-yolo/yolov8s-best.pt are byte-identical copies.",
    },
    INF: {
        "path": "sih2026-sankalp/models/infrastructure/infrastructure_detector.pt",
        "sha256": "7A71CDEE3C8E382DE5DEBE220ABD8EE499B6806E45255DBA80D69FB453C35837",
        "class_count": 57,
        "classes": {"0": "-Road narrows on right", "1": "50 mph speed limit", "2": "Advance Direction",
                    "3": "Attention Please-", "4": "Beware of children", "5": "Built Up Area",
                    "6": "Bullock and Hand Cart Prohibited", "7": "Bus Stop",
                    "8": "CYCLE ROUTE AHEAD WARNING", "9": "Cycle Prohibited",
                    "10": "Dangerous Left Curve Ahead", "11": "Dangerous Rright Curve Ahead",
                    "12": "End of all speed and passing limits", "13": "Expressway Rout Marking",
                    "14": "FS 31 Entry Ramp for Expressway", "15": "Filling Station",
                    "16": "Give Way", "17": "Go Straight or Turn Right", "18": "Go straight or turn left",
                    "19": "Height Limit", "20": "Keep-Left", "21": "Keep-Right", "22": "Left Hand Curv",
                    "23": "Left Zig Zag Traffic", "24": "Narrow Bridge", "25": "No Entry",
                    "26": "No_Over_Taking", "27": "Object Hazard Right",
                    "28": "Overtaking by trucks is prohibited", "29": "Pedestrain Crossing",
                    "30": "Pedestrian Crossing", "31": "Right Hand Curv", "32": "Round-About",
                    "33": "STOP Sign", "34": "School Ahead", "35": "Side Road Right",
                    "36": "Single Chevron", "37": "Slippery Road Ahead", "38": "Speed Limit 20 KMPh",
                    "39": "Speed Limit 30 KMPh", "40": "Stack type Advance Direction sign",
                    "41": "State Highway Route Marker", "42": "Stop_Sign", "43": "Straight Ahead Only",
                    "44": "Toilet", "45": "Tractor Prohibited", "46": "Traffic sign Not visibles",
                    "47": "Traffic_signal", "48": "Truck traffic is prohibited",
                    "49": "Turn left ahead", "50": "Turn right ahead", "51": "Two Wheeler Prohibited",
                    "52": "Uneven Road", "53": "other", "54": "speed limit", "55": "speed limit 20",
                    "56": "speed limit 80"},
    },
    "indian_traffic.pt": {
        "path": "indian_traffic.pt",
        "sha256": None,
        "class_count": None,
        "classes": {},
        "note": "Referenced by indian_sign_test.py / check_indian_model.py but FILE NOT FOUND. "
                "Original 14/15-class and 57-class training datasets also not found in project.",
    },
}

# reverse map: (model, class_id) -> canonical_id
MAPPING = {}
for s in SIGNS:
    for m in s["model_support"]:
        MAPPING[(m["source_model"], m["source_class_id"])] = s["canonical_id"]

catalogue = {
    "catalogue_version": "1.0.0",
    "created": "2026-11-16",
    "description": ("Canonical Indian traffic-sign catalogue. Recognition catalogue only; "
                    "not a claim that every sign is officially standardised. "
                    "Each entry records verification status honestly."),
    "standards_reference": {
        "primary": "MoRTH / IRC:67-2012 Code of Practice for Road Signs (3rd revision) - "
                   "three categories: Mandatory/Regulatory, Cautionary/Warning, Informatory",
        "status": "Category-level reference. Per-entry verification_status must still be "
                  "checked against the official IRC:67-2012 sign plates before operational use.",
        "speed_limits_note": "Indian speed-limit signage includes 20/30/40/50/60/70/80 km/h on "
                             "urban roads and higher limits on expressways. Not every limit is "
                             "used in every jurisdiction; classes here exist only because a "
                             "trained model already has them, or as catalogue proposals.",
    },
    "models": MODELS_META,
    "reverse_mapping": {
        f"{m}::{cid}": canon for (m, cid), canon in sorted(MAPPING.items(), key=lambda kv: (kv[0][0], kv[0][1]))
    },
    "unmapped_model_classes": {
        "infrastructure_detector.pt::1": {
            "raw_name": "50 mph speed limit",
            "reason": "Non-Indian variant (mph); deliberately not mapped to a km/h class.",
        },
        "infrastructure_detector.pt::3": {
            "raw_name": "Attention Please-",
            "reason": "Ambiguous class name with no clear canonical sign; left unmapped.",
        },
    },
    "signs": SIGNS,
}


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(catalogue, f, indent=2, ensure_ascii=False)

    # quick integrity checks
    ids = [s["canonical_id"] for s in SIGNS]
    assert len(ids) == len(set(ids)), "duplicate canonical ids"
    by_model = {B14: set(), INF: set()}
    for s in SIGNS:
        for m in s["model_support"]:
            by_model[m["source_model"]].add(m["source_class_id"])
    print("catalogue entries:", len(SIGNS))
    print("regulatory:", sum(1 for s in SIGNS if s["category"] == CAT_A))
    print("warning:", sum(1 for s in SIGNS if s["category"] == CAT_B))
    print("informatory:", sum(1 for s in SIGNS if s["category"] == CAT_C))
    print("speed limits:", sum(1 for s in SIGNS if s["category"] == CAT_D))
    print("traffic lights:", sum(1 for s in SIGNS if s["category"] == CAT_E))
    print("unrecognised bucket:", sum(1 for s in SIGNS if s["category"] == CAT_U))
    print("best.pt classes mapped:", len(by_model[B14]), "/ 14")
    print("infra classes mapped:", len(by_model[INF]), "/ 57")
    unmapped_14 = set(range(14)) - by_model[B14]
    unmapped_57 = set(range(57)) - by_model[INF]
    print("unmapped best.pt:", unmapped_14)
    print("unmapped infra:", sorted(unmapped_57))
    print("written:", OUT)


if __name__ == "__main__":
    main()
