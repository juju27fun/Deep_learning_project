# Diabetic Retinopathy Detection

Anomaly Detection and Severity Classification of Diabetic Retinopathy Using Deep Learning (EfficientNetB3).

## Project Structure

```
diabetic-retinopathy/
├── data/
│   ├── raw/                 # Raw data (train.csv, test.csv, images)
│   └── processed/           # Processed data
├── notebooks/
│   └── 01_eda.ipynb         # Exploratory Data Analysis
├── src/
│   ├── models.py            # Model architecture (EfficientNetB3)
│   ├── preprocessing.py     # Image loading and augmentation
│   ├── evaluation.py        # Metrics (QWK, Confusion Matrix)
│   ├── interpretability.py  # Grad-CAM visualization
│   ├── train.py             # Training and evaluation pipeline
│   └── config.py            # Configuration constants
├── outputs/                 # Training artifacts (models, plots)
├── pyproject.toml           # Dependencies (Poetry)
└── README.md
```

## Setup

1.  **Install Dependencies**:
    Ensure you have [Poetry](https://python-poetry.org/) installed.
    ```bash
    poetry install
    ```

2.  **Data**:
    Place your dataset in the `data/` directory.
    - `train.csv` should be in the root or `data/` as configured.
    - Images should be in `data/train_images/`.

## Usage

### 1. Exploratory Data Analysis
Run the EDA notebook to visualize class distribution and sample images.
```bash
poetry run jupyter lab notebooks/01_eda.ipynb
```

### 2. Training and Evaluation
Run the full pipeline to train the model, evaluate on the test set, and generate reports.
```bash
poetry run python -m src.train
```

This script will:
- Train the model (Transfer Learning + Fine-tuning).
- Save the best model to `outputs/best_model.h5`.
- Generate training history plots.
- Calculate the Quadratic Weighted Kappa (QWK) score.
- Save the Confusion Matrix and Grad-CAM visualizations to `outputs/`.

## Configuration
Modify `src/config.py` to adjust hyperparameters like `BATCH_SIZE`, `IMG_SIZE`, or `SPLIT_RATIO`.