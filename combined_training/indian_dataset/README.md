---
license: other
license_name: datacluster-commercial-sample
license_link: LICENSE
tags:
- object-detection
- image-classification
- traffic-sign
- self-driving
- autonomous-vehicles
- generative-ai
- visual-question-answering
- computer-vision
- sample-dataset
pretty_name: Indian Traffic Sign Image Dataset (Sample)
size_categories:
- n<1K
task_categories:
- object-detection
- image-classification
---

# Indian Traffic Sign Image Dataset (Sample)

> ⚠️ **This is a free sample subset for evaluation purposes only.**  
> The full dataset (2,000+ HD images) is available for commercial licensing.  
> **Contact:** [sales@datacluster.ai](mailto:sales@datacluster.ai) · [datacluster.ai](https://datacluster.ai)

---

## Dataset Summary

This dataset is an extremely challenging collection of original Indian traffic sign images, crowdsourced from over 400 urban and rural areas. Every image is manually reviewed and verified by computer vision professionals at DataCluster Labs.

Optimized for **Generative AI, Visual Question Answering, Image Classification, and LMM development**, this dataset provides a strong basis for robust model performance in traffic sign detection and related tasks.

## Dataset Features (Full Dataset)

| Feature | Detail |
|---|---|
| **Dataset size** | 2,000+ images |
| **Captured by** | 400+ crowdsource contributors |
| **Resolution** | 100% HD and above (1920x1080+) |
| **Location** | 400+ cities across India |
| **Diversity** | Day, night, varied distances, viewpoints |
| **Device** | Mobile phones, captured 2020–2021 |
| **Usage** | Traffic sign detection, self-driving systems, traffic detection, sign detection |
| **Ownership** | Exclusively owned by DataCluster Labs — not scraped from the internet |

## Sample vs. Full Dataset

| | Sample (this repo) | Full Dataset |
|---|---|---|
| Images | Small representative subset | 2,000+ |
| Annotation formats | Pascal VOC (XML) | COCO, YOLO, Pascal VOC, TF-Record |
| Resolution | HD and above | HD and above (1920x1080+) |
| Commercial use | ❌ Not permitted | ✅ With license |
| Redistribution | ❌ Not permitted | Per license terms |

**To license the full dataset:** [sales@datacluster.ai](mailto:sales@datacluster.ai)

## Dataset Structure

```
Indian_Traffic_Sign_Image_Dataset/
├── images/
│   ├── img001.jpg
│   └── ...
└── annotations/
    └── voc/
        ├── img001.xml
        └── ...
```

Each XML file contains bounding-box annotations in Pascal VOC format, with filenames matching their corresponding image.

## Data Collection

- **Source:** Crowdsourced original photography — not downloaded from the internet
- **Contributors:** 400+ crowdsource contributors across India
- **Conditions:** Day, night, varied lighting, distances, and viewpoints
- **Use cases:** Traffic sign detection, self-driving/autonomous vehicle systems, ADAS, generative AI, VQA, LMM training

## How to Use

### Download

```bash
huggingface-cli download Dataclusterlabspvtltd/Indian_Traffic_Sign_Image_Dataset --repo-type dataset --local-dir ./Indian_Traffic_Sign_Image_Dataset
```

Or clone directly:

```bash
git lfs install
git clone https://huggingface.co/datasets/Dataclusterlabspvtltd/Indian_Traffic_Sign_Image_Dataset
```

### Convert VOC to YOLO or COCO

```python
from pylabel import importer

# VOC → YOLO
dataset = importer.ImportVOC(path="annotations/voc")
dataset.export.ExportToYoloV5(output_path="annotations/yolo")

# VOC → COCO
dataset.export.ExportToCoco(output_path="annotations/coco/annotations.json")
```

## License

This sample dataset is released under a **restricted commercial-sample license** — see the `LICENSE` file in this repository.

Key points:
- ✅ Free to download and evaluate
- ✅ Free for academic research with attribution
- ❌ No commercial use without a license from DataCluster Labs
- ❌ No redistribution or re-upload
- ❌ No use in training commercial ML models

The images are exclusively owned by DataCluster Labs. To access the full dataset for research or commercial purposes, a license can be purchased.

**Contact:** [sales@datacluster.ai](mailto:sales@datacluster.ai)

## Citation

```bibtex
@misc{datacluster_indian_traffic_sign_sample,
  title        = {Indian Traffic Sign Image Dataset (Sample)},
  author       = {DataCluster Labs},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/datasets/Dataclusterlabspvtltd/Indian_Traffic_Sign_Image_Dataset}},
  note         = {Sample subset. Full dataset (2,000+ images) available for commercial licensing at sales@datacluster.ai}
}
```

## About DataCluster Labs

DataCluster Labs specializes in managed crowd-sourced data collection and annotation — images, videos, audio, text, and surveys — through our Dailydata platform. We deliver custom datasets for computer vision, NLP, and ML use cases.

📧 **Sales / Full Dataset Access:** sales@datacluster.ai  
🌐 **Website:** [datacluster.ai](https://datacluster.ai)