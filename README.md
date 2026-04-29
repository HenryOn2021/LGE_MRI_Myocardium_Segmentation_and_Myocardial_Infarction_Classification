# LGE Cardiac MRI AI Pipeline

Industry-facing research repository for **Late Gadolinium Enhancement (LGE) cardiac MRI segmentation and binary myocardial infarction (MI) classification**.

This project was developed from a PhD research pipeline using the public EMIDEC dataset and an in-house Imperial/NHS dataset. It demonstrates a reproducible medical imaging workflow covering data preparation, segmentation, classification, cross-validation, ensemble inference, and standardised metric reporting.

> **Research use only.** This repository is not a certified medical device and is not intended for clinical deployment.

---

## Key Features

### Segmentation
- 2D, 3D and Cascaded3D U-Net style segmentation workflows.
- MONAI/PyTorch-based training.
- Fold-aware model training.
- Native-space inference and metric reporting.
- Dice and HD95 evaluation.
- Optional fold ensemble inference.

### MI Classification
- Binary classification of MI presence/absence from LGE MRI.
- 2D slice-based classification using DenseNet and ResNet variants.
- 3D volume-based classification using DenseNet variants.
- Stratified hold-out testing and 5-fold cross-validation.
- Standardised outputs including metrics, ROC/PR curves and confusion matrices.
- Optional 2D GradCAM/GuidedGradCAM analysis.

---

## Repository Structure

```text
.
├── main.py                         # Unified CLI entry point
├── dataset.py                      # Dataset utilities
├── dataset_preparer.py             # Dataset split preparation
├── segmentation_myo.py             # Segmentation training
├── segmentation_inference_myo.py   # Segmentation inference/evaluation
├── LGE_Classification_2D.py        # 2D MI classification model zoo
├── LGE_Classification_test5_3D.py  # 3D MI classification model zoo
├── transforms.py                   # Transform utilities
├── utils.py                        # Common utilities
├── configs/                        # Example reproducible configs
├── data/README.md                  # Dataset format documentation
├── docs/                           # Additional usage notes
├── scripts/                        # Example shell commands
└── tests/                          # Lightweight import tests
```

---

## Installation

```bash
conda create -n lge-ai python=3.10 -y
conda activate lge-ai
pip install -r requirements.txt
```

For GPU use, install a PyTorch build matching your CUDA version before installing the remaining packages.

---

## Dataset Policy

Raw data are **not included** in this repository.

The repository is designed for:
- EMIDEC LGE MRI data
- In-house LGE MRI data with matching labels

See [`data/README.md`](data/README.md) for the expected folder and CSV format.

---

## Unified CLI Usage

### 1. Data preparation

```bash
python main.py \
  --stage data_prep \
  --root_dir /path/to/project
```

---

### 2. Segmentation training

```bash
python main.py \
  --stage seg_train \
  --mode 3D \
  --root_dir /path/to/project \
  --fold 1 \
  --epochs 500 \
  --train_bs 8 \
  --roi_3d 224,224,18 \
  --spacing_3d 1.0,1.0,8.0
```

---

### 3. Segmentation evaluation

```bash
python main.py \
  --stage seg_eval \
  --mode 3D \
  --root_dir /path/to/project \
  --date_str 100825 \
  --n_folds 5
```

---

### 4. 2D MI classification

```bash
python main.py \
  --stage classif_full \
  --mode 2D \
  --classif_model resnet50 \
  --images_dir_emidec /path/to/emidec_2d_images \
  --images_dir_imperial /path/to/imperial_2d_images \
  --labels_csv_emidec /path/to/emidec_labels.csv \
  --labels_csv_imperial /path/to/imperial_labels.csv \
  --out_dir outputs/classification_2d \
  --spatial_size_2d 224,224 \
  --classif_batch_size 8 \
  --lr 3e-4
```

Supported 2D models:
- `densenet121`, `densenet169`, `densenet201`, `densenet264`
- `resnet18`, `resnet34`, `resnet50`, `resnet101`, `resnet152`

Optional CAM analysis:

```bash
python main.py \
  --stage classif_full \
  --mode 2D \
  --classif_model resnet50 \
  --enable_cams \
  --infarct_dir_emidec /path/to/emidec_infarct_masks \
  --infarct_dir_imperial /path/to/imperial_infarct_masks \
  ...
```

---

### 5. 3D MI classification

```bash
python main.py \
  --stage classif_full \
  --mode 3D \
  --classif_model densenet121 \
  --images_dir_emidec /path/to/emidec_3d_images \
  --images_dir_imperial /path/to/imperial_3d_images \
  --labels_csv_emidec /path/to/emidec_labels.csv \
  --labels_csv_imperial /path/to/imperial_labels.csv \
  --out_dir outputs/classification_3d \
  --spatial_size_3d 224,224,32 \
  --classif_batch_size 8 \
  --lr 3e-4
```

Supported 3D models:
- `densenet121`, `densenet169`, `densenet201`, `densenet264`

---

## Output Examples

Classification outputs are saved under:

```text
outputs/classification_2d/<model_name>/
outputs/classification_3d/<model_name>/
```

Typical outputs include:
- trained fold checkpoints
- metrics summaries in JSON/CSV
- ROC curves
- precision-recall curves
- confusion matrices
- optional CAM visualisations for 2D classification

Segmentation outputs are saved under the project root folders used by the original pipeline, including model checkpoints and prediction folders.

---

## Default Showcase Hyperparameters

The CLI defaults are aligned with the portfolio version of the project:

```text
batch size:      8
2D patch size:   224 × 224
3D ROI:          224 × 224 × 18 for segmentation
3D class size:   224 × 224 × 32 for classification
learning rate:   3e-4
folds:           5
```

---

## Why This Repository Is Portfolio-Ready

This repository highlights skills relevant to medical imaging AI and production-oriented ML engineering:

- Modular Python project structure.
- Unified command-line interface.
- Reproducible cross-validation and hold-out testing.
- Multi-model experimentation.
- Standardised metric and figure generation.
- Separation of source code, configuration, data documentation, and outputs.
- Clear exclusion of private data, trained weights, and generated artefacts.

---

## Citation

If using this repository academically, please cite the associated PhD thesis work:

```text
Henry On. PhD Thesis, Imperial College London. Cardiac MRI AI for LGE segmentation and MI classification.
```
