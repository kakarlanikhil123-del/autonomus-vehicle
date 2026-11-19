# Indian Traffic Sign Recognition — Implementation Plan (written 2026-11-16)

## Phase 1 audit findings (verified, not assumed)

| Item | Finding |
|---|---|
| `best.pt` | 14 classes (Green/Red Light, Speed Limits 20–120, Stop). Loads OK. Detects Stop @0.927 on stop_test.jpg. SHA-256 matches protected hash. |
| `best_14class_backup.pt` | Byte-identical to best.pt (same SHA-256). Untouched. |
| `road-sign-detection-yolo/yolov8s-best.pt` | Byte-identical to best.pt. Untouched. |
| `infrastructure_detector.pt` | 57 classes (Indian signs, several misspellings: "Rright", "Curv", "Pedestrain"). Loads OK. Detects Stop_Sign @0.671, Traffic_signal @0.476. SHA-256 matches protected hash. |
| `sign_test.py` | Working webcam app, best.pt, OCR via Tesseract. PRESERVED — not modified. |
| `combined_training/indian_dataset` | DataCluster Labs **sample**: 150 images, 150 VOC XMLs, all 194 boxes share ONE generic label `traffic_sign`. 20 XMLs have zero objects. NOT multi-class labelled data. License: evaluation/academic only, no commercial training. |
| `combined_training/merged_dataset` | EMPTY (0 images, 0 labels, no data.yaml). |
| `Indian-Traffic-Sign-Image-Dataset` | 17 unannotated sample JPGs + README. No labels. |
| Original 57-class training dataset | NOT FOUND anywhere in the project — only the trained weights exist. |
| Original 14-class training dataset | NOT FOUND anywhere in the project — only the trained weights exist. |
| Environment | Python 3.13.0, ultralytics 8.4.173, torch 2.14.1+**cpu** — NO GPU. |

## Existing 57-class per-class metrics on record (sih2026-sankalp report)

`speed limit` mAP50=0.995, `Speed Limit 20 KMPh`=0.635, `STOP Sign`=0.421, `Stack type Advance Direction`=0.495; `No_Over_Taking`, `Speed Limit 30 KMPh`, `Expressway Rout Marking` = 0.0 on the evaluation subset. Several classes have no measured performance.

## Plan

1. **P2+P3 — Catalogue & canonical mapping** (no data needed):
   - `config/indian_traffic_signs.json` — versioned canonical catalogue (categories A/B/C/D, alternative names, model support with source_model + source_class_id, advisories, training-data status, standards reference).
   - `sign_expansion/canonical_mapping.py` — normalizes both models' raw names (incl. typos) to canonical IDs; fusion of multi-model predictions (IoU dedupe, confidence rules, UNKNOWN when below threshold).
2. **P4 — Dataset tools** (`sign_expansion/dataset_tools.py`): VOC/YOLO readers, validation (class IDs, boxes, missing/duplicate images, unknown labels), reproducible stratified splits with leakage prevention, data.yaml, per-class summary. New output dir `sign_expansion/dataset_generated/`. Run honestly on the 150-image sample; report that it is single-generic-class.
3. **P5 — Training**: NOT POSSIBLE honestly today: no multi-class labelled data + CPU-only machine. Provide `sign_expansion/train_expanded_model.py` (configurable, safe run dirs, early stopping, per-class metrics, comparison report) marked NOT RUN. No fake training.
4. **P6 — Integrated app** `indian_traffic_sign_app.py`: DSHOW webcam + image/video, model or model-combination selection, fusion, advisories, OCR with caching/dedupe, UNKNOWN handling, keyboard controls, graceful errors. `sign_test.py` untouched.
5. **P7 — Traffic lights**: `sign_expansion/traffic_light.py` — RED/GREEN via best.pt (verified), YELLOW explicitly unsupported → UNKNOWN; temporal smoothing; separate module, no confusion with Traffic_signal warning sign.
6. **P8 — Tests**: py_compile all new files; unit tests for mapping/fusion/dataset tools; image inference tests; record results in `sign_expansion/results/phase2_run/`.
7. **P9 — Final report** `sign_expansion/results/phase2_run/FINAL_REPORT.md`.

## Guardrails honoured
No weights modified (hashes re-verified at end), no datasets downloaded, no packages installed, sign_test.py untouched, all new artifacts in `config/`, `sign_expansion/`, or new root-level app file.
