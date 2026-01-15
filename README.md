Strep Throat Classification – Deep Learning

This repository contains a deep learning solution for classifying throat images as strep-positive or strep-negative, with optional integration of structured clinical symptom features.

The goal of this project is to demonstrate model design, evaluation judgment, and clear communication under small-data constraints.

Project Overview

- Dataset: 100 throat images (50 positive / 50 negative)
- Optional inputs: 7 binary clinical symptoms
- Primary task: Binary classification (strep vs non-strep)
- Framework: PyTorch
- Backbone: ResNet18 (ImageNet-pretrained)

Repository Structure

```
.
├── data/
│   ├── images/                # 100 throat images (JPEG)
│   └── sample_dataset_100.csv # Labels + clinical symptoms
├── src/
│   ├── dataset.py             # Dataset + transforms
│   ├── model.py               # Image-only and multimodal models
│   ├── train.py               # Training + validation
│   └── evaluate.py            # Evaluation utilities
├── outputs/                   # Saved metrics, plots, checkpoints
├── requirements.txt
└── README.md
```

Installation

```bash
pip install -r requirements.txt
```

Recommended (for reproducibility)

Due to known binary conflicts in some global Conda environments, results were generated using a local virtual environment:

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install -U pip
./.venv/bin/python -m pip install -r requirements.txt
```

Usage

All commands should be run from the repository root.

Train image-only model

```bash
python src/train.py \
  --mode image_only \
  --epochs 15 \
  --batch_size 8 \
  --lr 1e-4 \
  --seed 42 \
  --csv_path data/sample_dataset_100.csv \
  --images_dir data/images \
  --output_dir outputs/image_only
```

Train image + clinical features model

```bash
python src/train.py \
  --mode with_clinical \
  --epochs 15 \
  --batch_size 8 \
  --lr 1e-4 \
  --seed 42 \
  --csv_path data/sample_dataset_100.csv \
  --images_dir data/images \
  --output_dir outputs/with_clinical
```

Model Architecture

- Image encoder: ResNet18 pretrained on ImageNet
- Image features: 512-dimensional embedding
- Clinical features: 7 binary inputs (optional)
- Fusion: Concatenation of image + clinical embeddings
- Classifier: Fully connected layers with dropout
- Output: Two logits (Negative / Positive)

Training Details

- Train/validation split: 80/20 (stratified)
- Data augmentation: Horizontal flip, small rotations, color jitter
- Optimizer: Adam (lr = 1e-4)
- Loss: CrossEntropyLoss
- Epochs: 15
- Batch size: 8
- Hardware: CPU or GPU (auto-detected)

Hyperparameters were intentionally kept conservative to reduce overfitting given the small dataset size.

Evaluation Metrics

The following metrics are reported on the validation set:

- Accuracy
- Precision
- Recall
- F1 score
- ROC-AUC (threshold-independent)

Metrics and plots are saved automatically to the specified output_dir.

Key Results (15 epochs)

| Model | Accuracy | F1 | Recall | ROC-AUC |
|-------|----------|----|--------|---------|
| Image-only | 60.0% | 69.23% | 81.82% | 53.54% |
| Image + clinical | 55.0% | 60.87% | 63.64% | 51.52% |

Summary:

- The image-only model performed best across all metrics
- Adding clinical features did not improve performance on this dataset
- High recall in the image-only model suggests good sensitivity to positive cases

Interpretation

The lack of improvement from clinical features is likely due to:

- Very small dataset size (100 samples)
- Coarse, binary symptom features
- Increased variance from additional model parameters

This result highlights that additional features do not always improve performance, especially under data constraints.

Limitations

- Extremely small dataset → high variance
- Single train/validation split
- Limited clinical feature granularity
- No external validation cohort

Future Work

With more data or time:

- Stratified k-fold cross-validation
- Larger and more diverse dataset
- Richer clinical features (severity, duration, demographics)
- Model interpretability (Grad-CAM)
- Calibration and robustness analysis

Reproducibility Notes

- Random seed supported via `--seed`
- All dependencies pinned in requirements.txt
- Metrics and outputs saved to disk for inspection
