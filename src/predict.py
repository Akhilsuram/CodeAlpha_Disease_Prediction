
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "heart_disease_model.joblib"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "Model not found. Run: python src/train_model.py"
    )

model = joblib.load(MODEL_PATH)

# Example values in the format expected by the UCI Heart Disease dataset.
sample = {
    "age": 55,
    "sex": 1,
    "cp": 0,
    "trestbps": 140,
    "chol": 250,
    "fbs": 0,
    "restecg": 1,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 1.0,
    "slope": 1,
    "ca": 0,
    "thal": 2
}

input_data = pd.DataFrame([sample])
probability = model.predict_proba(input_data)[0, 1]
prediction = int(probability >= 0.5)

print("\nHeart Disease Prediction Demo")
print("-----------------------------")
print("Predicted class:", prediction)
print(f"Model probability for heart disease: {probability:.2%}")

if prediction == 1:
    print("Model result: Heart disease risk indicated.")
else:
    print("Model result: Heart disease not indicated by this model.")

print("\nEducational demo only—not a medical diagnosis.")
