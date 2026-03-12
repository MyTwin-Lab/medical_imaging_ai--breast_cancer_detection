# Breast Cancer Detection — Mammography AI

A two-stage deep learning pipeline for lesion detection and BI-RADS classification on mammograms, trained on the INbreast dataset.

**Team**: Patricia, Shriya, Asmaa, Shane, Duncan, Govind, Kunal

---

## Overview

The system works in two sequential stages:

1. **Lesion Detection** — Faster R-CNN (ResNet-50 FPN) detects and localizes lesions in a full mammogram, outputting bounding boxes and lesion type (Mass, Calcification, Asymmetry).
2. **BI-RADS Classification** — EfficientNet-B4 classifies each detected ROI crop by BI-RADS score (binary: benign vs malignant, or multiclass: 8 levels).

```
DICOM → Detector → ROI crops → Classifier → BI-RADS label + overall assessment
```

---

## Dataset — INbreast

The project uses the [INbreast](https://www.academicradiology.org/article/S1076-6332(11)00451-X/abstract) dataset.

```
INbreast/
├── AllDICOMs/          # 412 anonymized mammogram DICOM files
├── AllXML/             # 343 OsiriX polygon annotation files
├── AllROI/             # 343 corresponding .roi files
├── MedicalReports/     # 343 radiologist reports
├── PectoralMuscle/     # Pectoral muscle boundary annotations
├── INbreast.csv        # Metadata: patient ID, laterality, view, ACR density, BI-RADS
└── INbreast.xls        # Excel version of metadata
```

**BI-RADS scale**: 1 (negative) · 2 (benign) · 3 (probably benign) · 4a/4b/4c (suspicious) · 5 (likely malignant) · 6 (known malignancy)

**Lesion types** extracted from XML annotations:
| XML label | Model class |
|---|---|
| Mass, Spiculated Region, Distortion | Mass (1) |
| Calcification, Cluster | Calcification (2) |
| Asymmetry | Asymmetry (3) |

---

## Project Structure

```
.
├── src/
│   ├── data/
│   │   ├── xml_parser.py          # Parse OsiriX plist XMLs → bounding boxes
│   │   ├── inbreast_dataset.py    # Dataset for full-image BI-RADS classification
│   │   ├── detection_dataset.py   # Dataset for Faster R-CNN training
│   │   └── crop_dataset.py        # Dataset for ROI crop classification (Step 2)
│   ├── models/
│   │   ├── lesion_detector.py     # Faster R-CNN wrapper
│   │   └── birads_classifier.py   # EfficientNet-B4 wrapper
│   └── utils/
│       └── metrics.py             # Accuracy, AUC, F1, confusion matrix
├── train_detector.py              # Train the lesion detector
├── train_birads.py                # Train the BI-RADS classifier
├── inference_pipeline.py          # Run full two-stage inference on a DICOM
├── visualize_annotations.py       # Visualize GT annotations and model predictions
├── outputs/
│   ├── detector/                  # best_detector.pt + training_log.csv
│   ├── birads/                    # best_model.pt + training_log.csv
│   └── visualizations/            # Saved visualization PNGs
├── INbreast/                      # Dataset (see above)
├── GUIDE.md                       # Contribution guide and branching workflow
└── ROADMAP.md                     # Project phases and team assignments
```

---

## Setup

### 1. Prerequisites

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- A Kaggle account with API credentials (to download INbreast)

Create a `.env` file in the project root with your Kaggle credentials:

```
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_api_key
```

> Get your key from https://www.kaggle.com/settings → API → Create New Token.

### 2. Run setup

```bash
bash setup.sh
```

This will:
- Create the `mamography` conda environment (Python 3.11, PyTorch 2.7 with CUDA 12.8)
- Download the INbreast dataset into `INbreast/`

Options:
```
--skip-env        Skip env creation (already done)
--skip-download   Skip dataset download (already present)
```

### 3. Activate the environment

```bash
conda activate mamography
```

All training and inference commands below assume the environment is active.

---

## Data Splits

### Why patient-aware splitting matters

Each patient in INbreast has up to 4 images: CC and MLO views for the left and right breast. A naive random split can put the CC view of a patient in the training set and the MLO view of the same patient in the validation set. The model then "sees" the patient during training, making validation metrics over-optimistic — this is **data leakage**.

Both training scripts use a patient-aware split (`src/data/patient_split.py`) that keeps all images from the same patient in the same partition.

### How patient groups are reconstructed

The public INbreast release has the Patient ID column anonymized (set to `"removed"`). We reconstruct patient groups from two structural properties of the dataset:

1. **Same acquisition date** — all images from one patient visit share the same `YYYYMM` acquisition date.
2. **Consecutive file-name IDs** — within a date, images from the same patient have sequential integer file names with small gaps (~19–28). A gap larger than 200 between consecutive file names indicates a new patient.

This reliably recovers the ~115 patient groups across the 410 images.

### Split sizes

| Script | Train | Val | Test |
|---|---|---|---|
| `train_detector.py` | ~80% of patients | ~20% | — |
| `train_birads.py` | ~70% of patients | ~15% | ~15% |

Fractions are over **patients**, not images, so the actual image counts may differ slightly due to varying numbers of images per patient.

---

## Training

### Step 1 — Train the lesion detector

```bash
python train_detector.py \
  --img-size 1024 \
  --epochs 30 \
  --batch-size 4 \
  --lr 5e-4
```

Saves `outputs/detector/best_detector.pt` and `training_log.csv`.

Key options:
- `--val-split` — validation fraction (default 0.2)
- `--no-pretrain` — disable COCO pretrained weights

### Step 2 — Train the BI-RADS classifier

**Option A: full-image classification**
```bash
python train_birads.py \
  --mode binary \
  --img-size 512 \
  --epochs 40 \
  --batch-size 8 \
  --lr 1e-4
```

**Option B: ROI crop classification** (uses detected bounding boxes as training regions)
```bash
python train_birads.py --crop-mode --crop-size 224
```

Saves `outputs/birads/best_model.pt` and `training_log.csv`.

Key options:
- `--mode` — `binary` (benign/malignant) or `multiclass` (8 BI-RADS levels)
- `--dropout` — classifier head dropout (default 0.4)

Training uses a two-phase strategy: backbone frozen for first `epochs//6` epochs, then full fine-tuning with a lower learning rate.

---

## Inference

Run the full two-stage pipeline on a single DICOM:

```bash
python inference_pipeline.py \
  --dicom path/to/image.dcm \
  --detector outputs/detector/best_detector.pt \
  --classifier outputs/birads/best_model.pt \
  --score-thresh 0.4
```

Output:
- Prints each detected lesion with type, confidence score, and BI-RADS prediction
- Prints an overall malignancy assessment
- Saves annotated image to `outputs/inference/{filename}_result.png`

---

## Visualization

View ground truth annotations (and optionally model predictions) for dataset images:

```bash
# Random sample of 9 images
python visualize_annotations.py --count 9

# Single image
python visualize_annotations.py --file 22678622

# Filter by lesion type, overlay model predictions
python visualize_annotations.py \
  --lesion Mass \
  --detector outputs/detector/best_detector.pt \
  --score-thresh 0.4
```

Output PNGs are saved to `outputs/visualizations/`.

---

## Model Architecture

| Component | Architecture | Pretrained on |
|---|---|---|
| Lesion Detector | Faster R-CNN + ResNet-50 FPN | COCO |
| BI-RADS Classifier | EfficientNet-B4 | ImageNet |

**Preprocessing**:
- DICOM window/level normalization
- CLAHE contrast enhancement
- Stratified train/val/test splits
- Class-weighted loss to handle BI-RADS imbalance
- Augmentation via albumentations (with bounding box support for detection)

**Evaluation metrics**: ROC-AUC, F1 (classifier) · mAP@0.5, per-class AP (detector)

---

## Outputs

Pre-trained checkpoints from the current training run:

| File | Size | Description |
|---|---|---|
| `outputs/detector/best_detector.pt` | 159 MB | Faster R-CNN checkpoint |
| `outputs/birads/best_model.pt` | 68 MB | EfficientNet-B4 checkpoint |

---

## References

- **INbreast dataset**: Moreira et al., [*INbreast: Toward a Full-field Digital Mammographic Database*](https://www.academicradiology.org/article/S1076-6332(11)00451-X/abstract), Academic Radiology, 2012.
- **Faster R-CNN**: Ren et al., *Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks*, NeurIPS 2015.
- **EfficientNet**: Tan & Le, *EfficientNet: Rethinking Model Scaling for CNNs*, ICML 2019.
- **MONAI**: [monai.io](https://monai.io) — Medical Open Network for AI.
