# Alzheimer's Disease Prediction and Multistage Classification

An MRI image-based system for **detecting** Alzheimer's disease and **classifying its stage** using a Support Vector Machine (SVM) with an RBF kernel. This project reproduces the methodology of our published paper, *"Early Prediction of Alzheimer Disease and Multiclass Classifier System"* (Advances in Nonlinear Variational Inequalities, Vol 28 No. 2s, 2025).

## Overview

Brain MRI images are preprocessed and fed into two separate SVM classifiers:

1. **Detection model (binary)** — Is the patient demented at all? `Non-Demented` vs. `Demented` (Very Mild, Mild, or Moderate combined).
2. **Classification model (multiclass)** — Which of the 4 stages is it? `Non-Demented`, `Very Mild Demented`, `Mild Demented`, `Moderate Demented`.

Both models use the same preprocessing pipeline: images are resized to 256×256 and flattened into 1D feature vectors before training.

## Dataset

This project uses the **Alzheimer MRI Preprocessed Dataset** (6,400 MRI images across 4 classes). The dataset is **not included in this repo** due to its size — download it yourself before running the notebook:

- Kaggle: [alzheimer-mri-dataset](https://www.kaggle.com/datasets/sachinkumar413/alzheimer-mri-dataset)
- Mirror (Mendeley Data): [Alzheimer MRI Preprocessed Dataset](https://data.mendeley.com/datasets/3r8hw8wmmk/1)

| Class | No. of Images |
|---|---|
| Non-Demented | 3200 |
| Very Mild Demented | 2240 |
| Mild Demented | 896 |
| Moderate Demented | 64 |

**Setup:** after downloading, place the extracted folder in the repo root, named exactly `Alzheimer (Preprocessed Data)`, with these subfolders directly inside it:

```
Alzheimer (Preprocessed Data)/
├── Non_Demented/
├── Very_Mild_Demented/
├── Mild_Demented/
└── Moderate_Demented/
```

The notebook resolves this path automatically relative to its own location (`DATA_DIR = os.path.join(os.getcwd(), "Alzheimer (Preprocessed Data)")`), so no manual path editing is needed as long as the folder sits next to `Alzheimer.ipynb`.

## Setup

```bash
git clone https://github.com/PradunyaKale/Alzheimer-s-Disease-prediction-and-multistage-classification.git
cd Alzheimer-s-Disease-prediction-and-multistage-classification
pip install -r requirements.txt
```

Then place the dataset folder as described above, and open `Alzheimer.ipynb` in Jupyter.

## Usage

Run the notebook top to bottom. It's organized in two halves:

1. Cells before the **`ALZHEIMER CLASSIFIER`** markdown cell — the binary detection model.
2. Cells after it — the 4-class classification model.

Each half preprocesses the images, splits into train/test (80/20), trains an RBF-kernel SVM, and evaluates it.

> **Note on training time:** fitting the SVM on full-resolution flattened images (65,536 features per image) takes roughly **1.5 hours per model** on a typical CPU. Once trained, each model is cached to disk via `joblib` (`svm_model.pkl` for detection, `svm_model2.pkl` for classification) so it never needs to be retrained — just reload with `joblib.load(...)`. These `.pkl` files aren't committed to the repo (see `.gitignore`) since they're large; you'll regenerate them the first time you run the training cells.

## Results

### Detection Model (Binary: Demented vs. Non-Demented)

| Split | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Train | 87.88% | 85.84% | 98.08% | 91.55% |
| Test | 84.38% | 82.17% | 97.21% | 89.06% |

**Test confusion matrix:**

| | Predicted: Not Detected | Predicted: Detected |
|---|---|---|
| **Actual: Not Detected** | 399 | 265 |
| **Actual: Detected** | 35 | 1221 |

### Classification Model (4-Class Stage Prediction)

| Split | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Train | 81.99% | 83.43% | 81.99% | 81.04% |
| Test | 75.70% | 77.06% | 75.70% | 74.13% |

**Test confusion matrix** (rows = actual, columns = predicted; order: Non-Demented, Very Mild, Mild, Moderate):

| | Non-Demented | Very Mild | Mild | Moderate |
|---|---|---|---|---|
| **Non-Demented** | 592 | 50 | 0 | 0 |
| **Very Mild** | 141 | 307 | 0 | 0 |
| **Mild** | 36 | 74 | 70 | 0 |
| **Moderate** | 3 | 7 | 0 | 0 |

The Moderate Demented class has very few samples (64 total), which is why it's underrepresented in the confusion matrix and harder for the model to learn reliably — a known class-imbalance limitation also noted in the accompanying paper.

## Tech Stack

- **Model:** Support Vector Machine (RBF kernel), via `scikit-learn`
- **Image processing:** Pillow, scikit-image
- **Data handling:** NumPy, pandas
- **Visualization:** Matplotlib, Seaborn

## Reference

Upadhye, G. D., Jaybhaye, S. M., Phatangare, S. A., Harale, A., Anandhan, J., Kabade, M., Kale, P., & Kale, P. (2025). *Early Prediction of Alzheimer Disease and Multiclass Classifier System*. Advances in Nonlinear Variational Inequalities, 28(2s), 73–82.

## Future Work

- Integrate CNN-based feature extraction alongside/instead of raw flattened pixels for improved generalizability.
- Address class imbalance (particularly the Moderate Demented class) via SMOTE or similar resampling.
- Package into a clinical-support tool for use under a healthcare professional's guidance.
