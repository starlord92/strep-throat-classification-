Strep Throat Classification – Deep Learning

This repository contains a deep learning solution for classifying throat images as strep-positive or strep-negative, with optional integration of structured clinical symptom features.

The goal of this project is to demonstrate model design, evaluation judgment, and clear communication under severe small-data constraints.

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
│   ├── train.py               # Training + validation (single split + k-fold CV)
│   └── evaluate.py            # Evaluation utilities
├── outputs/                   # Saved metrics, plots, checkpoints (created by train.py --output_dir)
│   ├── image_only/            # Single-split example
│   │   ├── best_model.pth
│   │   ├── final_metrics.csv
│   │   ├── training_metrics.csv
│   │   ├── confusion_matrix.png
│   │   └── training_curves.png
│   └── image_only_cv/         # K-fold example (k=5)
│       ├── cv_summary.csv
│       ├── cv_fold_metrics.csv
│       ├── fold_1/
│       │   ├── best_model.pth
│       │   ├── final_metrics.csv
│       │   ├── training_metrics.csv
│       │   ├── confusion_matrix.png
│       │   └── training_curves.png
│       ├── fold_2/
│       ├── fold_3/
│       ├── fold_4/
│       └── fold_5/             (each same as fold_1)
├── requirements.txt
├── EXECUTIVE_SUMMARY.md       # Optional 1–2 page report
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

Single Train/Validation Split (Sanity Check)

Train image-only model with a single stratified 80/20 split:

```bash
python src/train.py \
  --mode image_only \
  --epochs 15 \
  --batch_size 8 \
  --lr 1e-4 \
  --seed 42 \
  --k_folds 1 \
  --csv_path data/sample_dataset_100.csv \
  --images_dir data/images \
  --output_dir outputs/image_only
```

Train image + clinical features model:

```bash
python src/train.py \
  --mode with_clinical \
  --epochs 15 \
  --batch_size 8 \
  --lr 1e-4 \
  --seed 42 \
  --k_folds 1 \
  --csv_path data/sample_dataset_100.csv \
  --images_dir data/images \
  --output_dir outputs/with_clinical
```

Single-split training is primarily used for sanity checking and fast iteration.

Stratified K-Fold Cross-Validation (Primary Evaluation)

Train using 5-fold stratified cross-validation:

```bash
python src/train.py \
  --mode image_only \
  --epochs 15 \
  --batch_size 8 \
  --lr 1e-4 \
  --seed 42 \
  --k_folds 5 \
  --csv_path data/sample_dataset_100.csv \
  --images_dir data/images \
  --output_dir outputs/image_only_cv
```

Results from k-fold CV are saved as:

- Per-fold metrics: outputs/image_only_cv/cv_fold_metrics.csv
- Aggregated summary: outputs/image_only_cv/cv_summary.csv (mean ± std)
- Per-fold artifacts: outputs/image_only_cv/fold_1/, fold_2/, …

Model Architecture

- Image encoder: ResNet18 pretrained on ImageNet
- Image features: 512-dimensional embedding
- Clinical features: 7 binary inputs (optional)
- Fusion: Concatenation of image + clinical embeddings
- Classifier: Fully connected layers with dropout
- Output: Two logits (Negative / Positive)

Training Details

- Train/validation split: 80/20 (single-split mode, stratified)
- Cross-validation: Stratified k-fold (k=5)
- Data augmentation: Horizontal flip, small rotations, color jitter
- Optimizer: Adam (lr = 1e-4)
- Loss: CrossEntropyLoss
- Epochs: 15
- Batch size: 8
- Hardware: CPU or GPU (auto-detected)

Hyperparameters were intentionally conservative to mitigate overfitting given the extremely small dataset size.

Evaluation Metrics

The following metrics are reported:

- Accuracy
- Precision
- Recall
- F1 score
- ROC-AUC (threshold-independent)

Metrics and plots are saved automatically to the specified output directory.

Key Results

Single-Split Results (Sanity Check, 15 epochs)

| Model            | Accuracy | F1     | Recall | ROC-AUC |
|-----------------|----------|--------|--------|---------|
| Image-only      | 60.0%    | 69.23% | 81.82% | 53.54%  |
| Image + clinical| 55.0%    | 60.87% | 63.64% | 51.52%  |

Stratified 5-Fold Cross-Validation (Primary Results)

Image-only model (mean ± std across folds):

- Accuracy: 63.0% ± 7.5%
- F1: 61.8% ± 16.8%
- Recall: 68.0% ± 26.4%
- ROC-AUC: 58.8% ± 10.0%

Cross-validation shows performance consistently above random chance, with high variance across folds, which is expected given the limited dataset size and small validation sets per fold.

Interpretation

- The image-only model consistently outperformed the multimodal variant.
- Adding clinical features did not improve performance, likely due to:
  - Extremely small dataset size
  - Coarse, binary symptom features
  - Increased variance from additional parameters
- Cross-validation provides a more reliable estimate than a single split and highlights uncertainty in performance.
- This demonstrates that additional features do not automatically improve performance, particularly under severe data constraints.

Limitations

- Extremely small dataset (100 samples)
- High variance across validation folds
- Binary clinical features lack granularity
- No external or temporal validation cohort

Future Work

With additional data or time:

- Larger, more diverse dataset
- Richer clinical features (severity, duration, demographics)
- Model interpretability (Grad-CAM)
- Calibration and robustness analysis
- External validation on independent datasets

Reproducibility Notes

- Random seed controlled via --seed
- All dependencies pinned in requirements.txt
- All metrics and artifacts saved for inspection
