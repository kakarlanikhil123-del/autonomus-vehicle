"""
Reusable dataset pipeline for Indian traffic-sign data.

Capabilities (all honest, all reproducible):
  1. Read Pascal VOC XML annotations.
  2. Read YOLO TXT annotations.
  3. Validate class IDs and bounding boxes.
  4. Convert annotations ONLY when class mappings are valid (a mapping
     table must cover every label used; otherwise conversion is refused).
  5. Detect missing images / annotations.
  6. Detect duplicate images (byte hash).
  7. Detect invalid or unknown labels.
  8. Create reproducible stratified train/val/test splits.
  9. Generate a correct data.yaml.
 10. Produce a per-class image summary.
 11. Prevent leakage (a file appears in exactly one split; splits stored).
 12. Write everything to a NEW directory; never destroys source data.

Known honest limitation enforced by design:
  - combined_training/indian_dataset is a DataCluster Labs SAMPLE where all
    194 boxes share the single generic label "traffic_sign" (20 XMLs empty).
    It is imported AS ONE GENERIC CLASS and is NEVER expanded into STOP /
    GIVE WAY / etc. It is also evaluation/academic licensed: do NOT train
    commercial models on it without a licence.
"""

import hashlib
import json
import os
import random
import shutil
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict

# --------------------------------------------------------------- VOC reader

def read_voc_xml(xml_path):
    """Read one Pascal VOC XML -> {filename, width, height, objects:[{name,xmin..}]}.
    Raises ValueError on malformed XML so callers can count bad files."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    out = {
        "annotation_file": os.path.basename(xml_path),
        "filename": root.findtext("filename"),
        "width": root.findtext("size/width"),
        "height": root.findtext("size/height"),
        "objects": [],
    }
    for obj in root.iter("object"):
        bb = obj.find("bndbox")
        if bb is None:
            raise ValueError(f"{xml_path}: object without bndbox")
        out["objects"].append({
            "name": obj.findtext("name"),
            "xmin": float(bb.findtext("xmin")),
            "ymin": float(bb.findtext("ymin")),
            "xmax": float(bb.findtext("xmax")),
            "ymax": float(bb.findtext("ymax")),
        })
    return out


def validate_voc_record(rec):
    """Validate one VOC record. Returns list of issue strings (empty = OK)."""
    issues = []
    if not rec.get("filename"):
        issues.append("missing filename")
    if not rec.get("width") or not rec.get("height"):
        issues.append("missing width/height")
    try:
        w, h = int(rec["width"]), int(rec["height"])
        if w <= 0 or h <= 0:
            issues.append("non-positive image size")
        for o in rec["objects"]:
            x1, y1, x2, y2 = o["xmin"], o["ymin"], o["xmax"], o["ymax"]
            if x2 <= x1 or y2 <= y1:
                issues.append(f"invalid box {o['name']}: {x1},{y1},{x2},{y2}")
            elif x2 > w + 1 or y2 > h + 1:
                issues.append(f"box outside image for {o['name']}")
    except (TypeError, ValueError):
        issues.append("unparseable size/box values")
    return issues


# --------------------------------------------------------------- YOLO reader

def read_yolo_txt(txt_path, class_names=None):
    """Read one YOLO TXT (class cx cy w h normalised). Returns records list."""
    recs = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            parts = line.split()
            if not parts:
                continue
            if len(parts) != 5:
                raise ValueError(f"{txt_path}:{ln}: expected 5 values, got {len(parts)}")
            cls = int(parts[0])
            cx, cy, w, h = map(float, parts[1:])
            if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1):
                raise ValueError(f"{txt_path}:{ln}: normalised box out of range")
            if class_names is not None and not (0 <= cls < len(class_names)):
                raise ValueError(f"{txt_path}:{ln}: class id {cls} outside class list")
            recs.append({"class_id": cls, "cx": cx, "cy": cy, "w": w, "h": h})
    return recs


# --------------------------------------------------------------- dataset audit

def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def audit_image_annotation_pairs(images_dir, annotations, image_exts=(".jpg", ".jpeg", ".png")):
    """Detect missing images and missing annotations, plus duplicate images."""
    images = sorted(
        os.path.join(images_dir, f) for f in os.listdir(images_dir)
        if f.lower().endswith(image_exts)
    )
    by_stem = {os.path.splitext(f)[0]: f for f in (os.path.basename(a) for a in annotations)}
    img_stems = {os.path.splitext(os.path.basename(p))[0]: p for p in images}

    missing_images = [a for stem, a in by_stem.items() if stem not in img_stems]
    missing_annotations = [os.path.basename(p) for stem, p in img_stems.items() if stem not in by_stem]

    hashes = defaultdict(list)
    for p in images:
        hashes[file_sha256(p)].append(os.path.basename(p))
    duplicates = {h: names for h, names in hashes.items() if len(names) > 1}

    return {
        "num_images": len(images),
        "num_annotations": len(annotations),
        "missing_images": missing_images,
        "missing_annotations": missing_annotations,
        "duplicate_images": duplicates,
    }


# --------------------------------------------------------------- conversion

def voc_record_to_yolo_lines(rec, class_to_id):
    """Convert VOC record to YOLO lines. REFUSES unknown labels."""
    w, h = int(rec["width"]), int(rec["height"])
    lines = []
    for o in rec["objects"]:
        if o["name"] not in class_to_id:
            raise ValueError(f"unknown label '{o['name']}' - conversion refused")
        x1, y1, x2, y2 = o["xmin"], o["ymin"], o["xmax"], o["ymax"]
        cx = ((x1 + x2) / 2.0) / w
        cy = ((y1 + y2) / 2.0) / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        lines.append(f"{class_to_id[o['name']]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return lines


def stratified_split(records, val_frac, test_frac, seed, key):
    """Reproducible stratified split by a key (e.g. dominant class).
    Each record index appears in EXACTLY one split (no leakage)."""
    rng = random.Random(seed)
    by_group = defaultdict(list)
    for idx, rec in enumerate(records):
        by_group[key(rec)].append(idx)
    val, test, train = [], [], []
    for group in sorted(by_group):
        idxs = sorted(by_group[group])
        rng.shuffle(idxs)
        n = len(idxs)
        n_val = round(n * val_frac)
        n_test = round(n * test_frac)
        val += idxs[:n_val]
        test += idxs[n_val:n_val + n_test]
        train += idxs[n_val + n_test:]
    return sorted(train), sorted(val), sorted(test)


# --------------------------------------------------------------- build pipeline

def build_dataset_from_voc(
    voc_dir,
    images_dir,
    out_dir,
    class_to_id,           # explicit mapping; every label used MUST be covered
    dataset_name,
    val_frac=0.2,
    test_frac=0.1,
    seed=42,
    license_note=None,
    keep_empty_annotation_files=True,
):
    """Full pipeline: read VOC -> validate -> audit -> convert -> split -> write.
    Writes into a NEW out_dir; refuses to overwrite existing data there."""
    os.makedirs(out_dir, exist_ok=True)
    if os.listdir(out_dir):
        raise RuntimeError(f"{out_dir} already exists and is not empty - refusing (safety)")

    xmls = sorted(
        os.path.join(voc_dir, f) for f in os.listdir(voc_dir) if f.lower().endswith(".xml")
    )
    records, bad_xml = [], []
    for x in xmls:
        try:
            rec = read_voc_xml(x)
            records.append(rec)
        except (ET.ParseError, ValueError) as exc:
            bad_xml.append({"file": os.path.basename(x), "error": str(exc)})

    # per-record validation
    invalid_records = []
    for rec in records:
        issues = validate_voc_record(rec)
        if issues:
            invalid_records.append({"file": rec["annotation_file"], "issues": issues})
    invalid_names = {r["file"] for r in invalid_records}

    pairs = audit_image_annotation_pairs(images_dir, xmls)

    # label inventory
    label_counts = Counter()
    for rec in records:
        for o in rec["objects"]:
            label_counts[o["name"]] += 1
    unknown_labels = sorted(set(label_counts) - set(class_to_id))

    if unknown_labels:
        raise RuntimeError(
            f"Refusing conversion: labels not covered by class_to_id: {unknown_labels}. "
            "Extend the mapping explicitly (never auto-map generic labels)."
        )

    # stratify on dominant label (generic dataset -> one bucket, still reproducible)
    def key(rec):
        if not rec["objects"]:
            return "<empty>"
        return Counter(o["name"] for o in rec["objects"]).most_common(1)[0][0]

    usable = [r for r in records if r["annotation_file"] not in invalid_names]
    train_idx, val_idx, test_idx = stratified_split(usable, val_frac, test_frac, seed, key)

    split_dirs = {s: (os.path.join(out_dir, "images", s), os.path.join(out_dir, "labels", s))
                  for s in ("train", "val", "test")}
    for img_dir, lbl_dir in split_dirs.values():
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)

    stem_to_split = {}
    for split, idxs in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        for i in idxs:
            stem_to_split[os.path.splitext(usable[i]["annotation_file"])[0]] = split

    img_dir_src = images_dir
    split_counts = Counter()
    boxes_per_class = {s: Counter() for s in ("train", "val", "test")}
    for stem, split in stem_to_split.items():
        # find source image
        src_img = None
        for ext in (".jpg", ".jpeg", ".png"):
            cand = os.path.join(img_dir_src, stem + ext)
            if os.path.exists(cand):
                src_img = cand
                break
        if src_img is None:
            continue
        shutil.copy2(src_img, os.path.join(split_dirs[split][0], os.path.basename(src_img)))
        rec = next(r for r in usable if os.path.splitext(r["annotation_file"])[0] == stem)
        lines = voc_record_to_yolo_lines(rec, class_to_id)  # refuses unknown labels
        with open(os.path.join(split_dirs[split][1], stem + ".txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))
        split_counts[split] += 1
        for o in rec["objects"]:
            boxes_per_class[split][class_to_id[o["name"]]] += 1

    names = sorted(class_to_id, key=lambda k: class_to_id[k])
    yaml_lines = [
        f"# Auto-generated by sign_expansion/dataset_tools.py - do not hand-edit class ids",
        f"path: {os.path.abspath(out_dir)}",
        f"train: images/train",
        f"val: images/val",
        f"test: images/test",
        f"names:",
    ]
    for cid, nm in enumerate(names):
        yaml_lines.append(f"  {cid}: {nm}")
    yaml_path = os.path.join(out_dir, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write("\n".join(yaml_lines) + "\n")

    summary = {
        "dataset_name": dataset_name,
        "created": "2026-11-16",
        "seed": seed,
        "license_note": license_note,
        "class_to_id": class_to_id,
        "num_xml_files": len(xmls),
        "bad_xml_files": bad_xml,
        "invalid_or_issue_records": invalid_records,
        "pair_audit": pairs,
        "label_counts_raw": dict(label_counts),
        "unknown_labels": unknown_labels,
        "split_sizes": dict(split_counts),
        "boxes_per_class_per_split": {s: dict(c) for s, c in boxes_per_class.items()},
        "empty_annotation_files_note":
            "Files with zero objects are copied as background images with empty labels "
            "(valid for YOLO negative samples).",
        "leakage_check": "Each file assigned to exactly one split (train/val/test disjoint by construction).",
        "multi_class_warning":
            "This pipeline NEVER converts a generic label into specific sign classes.",
    }
    with open(os.path.join(out_dir, "dataset_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


# --------------------------------------------------------------- CLI

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Indian traffic-sign dataset tools")
    p.add_argument("--audit-voc", help="VOC dir to audit (read-only)")
    p.add_argument("--images-dir", help="images dir for audit")
    p.add_argument("--out", help="print JSON report to this file instead of stdout")
    args = p.parse_args()

    if args.audit_voc:
        xmls = sorted(os.path.join(args.audit_voc, f) for f in os.listdir(args.audit_voc) if f.lower().endswith(".xml"))
        labels = Counter(); bad = []; recs = []
        for x in xmls:
            try:
                recs.append(read_voc_xml(x))
            except (ET.ParseError, ValueError) as e:
                bad.append({"file": os.path.basename(x), "error": str(e)})
        for r in recs:
            for o in r["objects"]:
                labels[o["name"]] += 1
        empty = [r["annotation_file"] for r in recs if not r["objects"]]
        report = {
            "voc_dir": args.audit_voc,
            "num_xml": len(xmls),
            "bad_xml": bad,
            "unique_labels": dict(labels),
            "empty_annotation_files": empty,
            "honest_interpretation": None,
        }
        if list(labels) == ["traffic_sign"]:
            report["honest_interpretation"] = (
                "Single generic label 'traffic_sign': this is NOT a labelled multi-class "
                "dataset and must not be used to train individual sign classes."
            )
        text = json.dumps(report, indent=2, ensure_ascii=False)
        if args.out:
            os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
            with open(args.out, "w", encoding="utf-8") as f:
                f.write(text)
            print("written:", args.out)
        else:
            print(text)
    else:
        p.print_help()
