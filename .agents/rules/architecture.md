# Recommanded architecture

diabetic-retinopathy/
├── data/
│   ├── raw/                 # Downloaded from Kaggle [cite: 12]
│   └── processed/           # Resized/normalized images 
├── notebooks/
│   ├── 01_eda.ipynb         # Class balance visualization [cite: 13, 33]
│   └── 02_training.ipynb    # Model training and loss curves [cite: 35]
├── src/
│   ├── models.py            # Transfer learning architecture definitions 
│   ├── preprocessing.py     # Augmentation and processing logic 
│   ├── evaluation.py        # QWK, Confusion Matrix, F1-scores [cite: 27, 29]
│   └── interpretability.py  # Grad-CAM implementation [cite: 41]
├── outputs/                 # Saved models and plots for the report
├── pyproject.toml           # Poetry Configuration
└── README.md