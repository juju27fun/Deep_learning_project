# Environment & Technical Standards

## 1. Overview
This document outlines the frozen technical stack, versioning rules, and configuration standards for the "Diabetic Retinopathy Anomaly Detection" project. Adherence to these standards is required to ensure reproducibility and consistency across training and evaluation.

## 2. Environment Management
* **Manager:** Poetry
* **Reasoning:** Ensures strict dependency locking (`poetry.lock`) to satisfy the project's reproducibility requirement.
* **Python Version:** `3.10.x` (Selected for stability with TensorFlow 2.x and CUDA/cuDNN).

## 3. Core Libraries & Version Constraints
The following libraries are mandated for the transfer learning workflow.

| Category | Library | Version Constraint | Purpose |
| :--- | :--- | :--- | :--- |
| **Deep Learning** | `tensorflow` | `^2.15.0` | CNN Backbones (Keras), Training Loop & Tensor Ops |
| **Data Proc.** | `numpy` | `^1.26` | Matrix operations |
| | `pandas` | `^2.1` | CSV handling & Class balance analysis |
| | `opencv-python` | `^4.8` | Image resizing/preprocessing |
| **Metrics** | `scikit-learn` | `^1.3` | QWK Score, Confusion Matrix, Split |
| **Vis.** | `matplotlib` | `^3.8` | Loss curves, Sample visualization |
| | `seaborn` | `^0.13` | Heatmap visualization |
| **XAI** | `tf-keras-vis` | `^0.8` | Interpretability/Saliency Maps/GradCAM |

## 4. Model Choice

EfficientNet-B3
- transfer learning
- QWK metric


## 5. Configuration Constants
Global constants to be used across all notebooks and scripts to maintain consistency.

```python
# config.py standards

# Data Specifications
IMG_SIZE = (224, 224)      # Standard input size for ResNet/EfficientNet (Keras Applications)
NUM_CLASSES = 5            # 0=No DR, 1=Mild, 2=Moderate, 3=Severe, 4=Proliferative
BATCH_SIZE = 32            # Adjustable based on GPU VRAM

# Reproducibility
SEED = 42

# Training
SPLIT_RATIO = [0.7, 0.1, 0.2] # Train / Val / Test
