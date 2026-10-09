# CodeAlpha Disease Prediction

## Project Overview
A machine learning project that predicts whether heart disease is indicated by clinical attributes using the UCI Heart Disease dataset.

This project is for educational purposes only and is not a medical diagnostic tool.

## Dataset
- Source: UCI Machine Learning Repository
- Dataset: Heart Disease (repository dataset ID 45)
- Records: 303
- Features: 13
- Target: Original target values are converted into two classes:
  - `0`: No heart disease indicated
  - `1`: Heart disease indicated

## Machine Learning Model
- Random Forest Classifier
- Missing-value imputation
- Numerical feature scaling
- Categorical feature encoding
- Stratified train-test split

## Evaluation
Initial test results:
- Accuracy: 88.52%
- ROC-AUC: 0.9464

Results may vary if the data, model, or training configuration changes.

## Technologies
- Python
- pandas
- NumPy
- scikit-learn
- Matplotlib
- Seaborn
- ucimlrepo
- joblib

## Project Structure
```text
CodeAlpha_Disease_Prediction/
├── data/
├── models/
│   └── heart_disease_model.joblib
├── outputs/
│   ├── test_predictions.csv
│   ├── evaluation_report.txt
│   └── plots/
├── src/
│   ├── inspect_data.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── predict.py
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation
Create and activate a virtual environment, then install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Run the Project
Run all steps:

```powershell
python main.py
```

Or run individual scripts:

```powershell
python src/inspect_data.py
python src/train_model.py
python src/evaluate_model.py
python src/predict.py
```

The dataset is downloaded through `ucimlrepo` when required, so an internet connection is needed.

## Disclaimer
This project is intended for machine learning education only. Its predictions are not medical advice and must not be used to diagnose, treat, or rule out heart disease.