# FINAL REPORT — Indian Traffic Sign Recognition Expansion (2026-11-16)

## 1. Files inspected
- `best.pt`, `best_14class_backup.pt`, `road-sign-detection-yolo/yolov8s-best.pt` — all loaded, all **byte-identical copies of one 14-class model** (same SHA-256 `C62E044A…`)
- `sih2026-sankalp/models/infrastructure/infrastructure_detector.pt` — 57-class Indian model, loaded OK
- `sign_test.py`, `indian_sign_test.py`, `check_indian_model.py`, `check_model.py`, `traffic.py`, `camera.py`, `main.py`
- `sih2026-sankalp/src/detection/detect_infrastructure.py` + its `infrastructure_57_class_metrics.json`
- `combined_training/indian_dataset` (150 images + 150 VOC XMLs), `combined_training/merged_dataset`, `combined_training/scripts`
- `Indian-Traffic-Sign-Image-Dataset/` (17 unannotated sample JPGs)
- `road-sign-detection-yolo/` (training/eval code), `sign_expansion/` prior audit artifacts, `traffic_test/test.jpg`, `sih2026-sankalp/model_test_images/stop_test.jpg`
- Existing records: `infrastructure_57_class_metrics.json` (per-class mAP of the 57-class model)

## 2. Files created (all new; nothing replaced)
| File | Purpose |
|---|---|
| `config/indian_traffic_signs.json` | **Versioned canonical catalogue v1.0.0** |
| `sign_expansion/build_catalogue.py` | Reproducible catalogue generator |
| `sign_expansion/canonical_mapping.py` | Canonical mapping + safe prediction fusion |
| `sign_expansion/dataset_tools.py` | VOC/YOLO readers, validation, splits, data.yaml, summaries |
| `sign_expansion/train_expanded_model.py` | Safe training script (provided, **NOT run**) |
| `sign_expansion/traffic_light.py` | Separate traffic-light state module with smoothing |
| `indian_traffic_sign_app.py` | Integrated webcam/image/video application |
| `sign_expansion/tests_unit.py` | Unit tests (31 tests, all passing) |
| `sign_expansion/phase2_audit/*` | verify_models.py, functional_test.py, conf_sweep.py, webcam_test.py + reports |
| `sign_expansion/dataset_generated/traffic_sign_generic_sample/` | New validated dataset dir (105/30/15 split) |
| `sign_expansion/results/phase2_run/*` | Implementation plan, audits, test reports, annotated samples |
| `sign_expansion/backups/phase2_start_20261116/` | Fresh backups (see §4) |

## 3. Files modified
- `sign_expansion/dataset_tools.py` — bug fix during development (VOC reader now reads `size/width`, `size/height` correctly; dataset regenerated fresh into its own new directory)
- No other existing file was modified. **`sign_test.py` verified byte-identical to its backup after all work.**

## 4. Backups created
`sign_expansion/backups/phase2_start_20261116/`: sign_test.py, indian_sign_test.py, check_indian_model.py, check_model.py, traffic.py, camera.py, main.py, detect_infrastructure.py (all verified unmodified at finish).

## 5. Existing models preserved (re-verified at end)
| Weights | SHA-256 status |
|---|---|
| best.pt | match (unchanged) |
| best_14class_backup.pt | match (unchanged) |
| yolov8s-best.pt | match (unchanged) |
| infrastructure_detector.pt | match (unchanged) |

## 6–7. Canonical Indian sign catalogue
`config/indian_traffic_signs.json` — **89 canonical signs**: 30 regulatory, 27 warning, 16 informatory, 12 speed-limit classes, 3 traffic-light states, 1 UNRECOGNISED bucket. Each entry has canonical id/name, category, alternative names, advisory, model support (`source_model` + original `source_class_id` preserved), training-data status, and an honest `verification_status` (`standardized` vs `proposed`). Standards reference: MoRTH/IRC:67-2012 at category level; per-variant verification is explicitly still required — no invented sign codes.

## 8. Classes supported by existing trained models
- best.pt: **14/14 classes mapped** (Speed 20–120, Stop, Red/Green Light)
- infrastructure_detector.pt: **55/57 classes mapped**; 2 documented exceptions (`50 mph speed limit` — non-Indian variant; `Attention Please-` — ambiguous)
- Duplicate 57-class classes identified and de-duplicated at canonical level: 2× stop (33/42), 2× speed-20 (38/55), 2× pedestrian crossing (29 typo/30), 2× each curve (10/22, 11/31). Typos recorded: "Rright", "Curv", "Pedestrain", "Rout".

## 9–10. Training data status
- Classes with validated (labelled, multi-class) training data available **locally: 0**. The only local annotated dataset is the DataCluster sample with **one generic label `traffic_sign`** (194 boxes, 20 empty files, 1 byte-identical duplicate pair). It is imported honestly as a single generic class and is **evaluation/academic licensed only — no commercial training**.
- Classes still requiring labelled data: all catalogue entries not already inside the two trained models (≈ 45 of 89), plus YELLOW traffic light (0 labelled examples).
- The original 14-class and 57-class training datasets are **missing from the project** — only trained weights exist.

## 11–12. New model trained?
**No — honestly not trained.** No multi-class labelled data exists locally, and this machine is CPU-only (torch 2.14.1+cpu). `train_expanded_model.py` is ready (configurable epochs/batch/imgsz/conf, early stopping, new timestamped run dirs, per-class metrics + same-validation-set comparison vs both existing models) but unrun. No fake metrics are reported. Known measured performance on record for infrastructure_detector.pt: `speed limit` mAP50=0.995, `Speed Limit 20`=0.635, `STOP Sign`=0.421, `Stack Advance Direction`=0.495; several classes 0.0.

## 13. Webcam test results (measured, not claimed)
| Test | Result |
|---|---|
| Camera 0 via CAP_DSHOW | opened, 30/30 frames read, 640×480 |
| Fused both-models inference on webcam | 243 ms/frame ≈ **4.1 fps** (CPU) |
| Fused both-models on 3120×4160 still | 155 ms/frame ≈ 6.5 fps |
| 14-class model only | 122 ms/frame ≈ 8.2 fps |
| stop_test.jpg, both models fused | **Stop 0.93** (two models merged to ONE box) + Traffic Signals Ahead 0.48 |
| stop_test.jpg at conf 0.6 | 0.48 prediction correctly suppressed |
| traffic_test/test.jpg | 0 detections (honest: no qualifying sign) |
| Live UNKNOWN path | low-confidence webcam detections displayed as UNKNOWN, never coerced |
| Unit tests | 31/31 PASS |
| Syntax checks | all files compile |

## 14. Known limitations
1. Recognition coverage = the union of the two existing models: **58 distinct canonical signs reachable** (14 via best.pt; 45 additional via the 57-class model, de-duplicated). The other ~31 catalogue entries have **no trained support**.
2. Speed-limit OCR needs a clear, reasonably large sign; it accepts only plausible Indian limits (20–120 set) and otherwise reports "OCR pending" rather than a wrong number.
3. YELLOW traffic light: no trained class, no data — displayed state stays UNKNOWN. RED/GREEN work through best.pt with temporal smoothing.
4. 57-class model has weak measured classes (e.g. `STOP Sign` mAP50 0.42; several 0.0) — fusion mitigates but does not fix this.
5. Catalogue "proposed" entries need verification against IRC:67-2012 plates before operational use.
6. CPU-only inference ≈ 4 fps fused on webcam; GPU recommended for real-time.
7. The generic sample dataset is single-class and evaluation-licensed; it cannot produce sign-specific training.

## 15. Exact commands to run
```bat
:: webcam (fused both models — default)
python indian_traffic_sign_app.py

:: only one model
python indian_traffic_sign_app.py --mode 14
python indian_traffic_sign_app.py --mode 57

:: image / video
python indian_traffic_sign_app.py --source image --input "sih2026-sankalp\model_test_images\stop_test.jpg"
python indian_traffic_sign_app.py --source video --input <video.mp4> --save annotated.mp4

:: options: --camera-index 0  --conf 0.35  --ocr-cooldown 1.0
:: keys: q quit | 1/2/3 model mode | +/- confidence | t OCR toggle | s save frame

:: classic app (untouched, still works)
python sign_test.py

:: tests / audits
python sign_expansion\tests_unit.py
python sign_expansion\phase2_audit\verify_models.py
python "sign_expansion\dataset_tools.py" --audit-voc "combined_training\indian_dataset\Annotations\Annotations"
```

## 16. Steps to add more labelled signs later
1. Collect/choose a properly labelled Indian traffic-sign dataset (multi-class bounding boxes). Candidate to evaluate: the full DataCluster Indian Traffic Sign dataset (commercial licence — sales@datacluster.ai) or another licensed source you approve. **No data has been downloaded.**
2. Verify the licence permits your use (academic vs commercial).
3. Put images + VOC XML (or YOLO TXT) into a new folder; make sure every image has an annotation file and labels are specific sign names.
4. Extend `config/indian_traffic_signs.json` (via `sign_expansion/build_catalogue.py`) with any new canonical entries and their advisories.
5. Build a training dataset:
   `python -c` call to `sign_expansion.dataset_tools.build_dataset_from_voc(...)` with an explicit `class_to_id` covering **every** label — unknown labels are refused, splits are reproducible and leakage-free, output goes to a NEW directory.
6. Train safely:
   `python sign_expansion\train_expanded_model.py --data <new data.yaml> --model yolov8s.pt --epochs 100 --batch 16 --imgsz 640`
   (GPU required for a practical runtime; CPU-only will be extremely slow.)
7. Review `comparison_report.json` — adopt the new weights ONLY if they beat BOTH existing models per-class on the same validation set. Protected weights are never overwritten; new weights live in `sign_expansion/training_runs/<timestamp>/`.

## Clear distinction (requested)
- **Catalogue entries:** 89 (proposed recognition catalogue, not a claim of official standardisation per variant)
- **Model-supported classes:** 58 distinct canonical signs reachable today (14 via best.pt, 45 additional via the 57-class model, deduplicated)
- **Classes tested successfully:** Stop (0.93 fused, both models), Red/Green traffic light (live webcam, smoothed), generic sign behaviour + UNKNOWN handling, OCR pipeline (enabled, accepts only plausible limits), fusion/dedup, dataset pipeline
- **Classes requiring training data:** ~31 catalogue entries with no model support at all, plus several reachable classes with weak measured metrics (No Overtaking 0.0, Speed-30 0.0 mAP50) and YELLOW light (none)
