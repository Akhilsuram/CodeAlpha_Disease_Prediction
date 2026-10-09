
from pathlib import Path

import joblib
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"
DATA_DIR = ROOT / "data"

for folder in (MODEL_DIR, OUTPUT_DIR, DATA_DIR):
    folder.mkdir(parents=True, exist_ok=True)

print("Loading Heart Disease dataset...")
dataset = fetch_ucirepo(id=45)

X = dataset.data.features.copy()
y = dataset.data.targets.iloc[:, 0]

# Convert target values 0 to 4 into binary labels:
# 0 = no heart disease, 1–4 = heart disease present.
y = pd.to_numeric(y, errors="coerce")
valid_rows = y.notna()
X = X.loc[valid_rows].copy()
y = (y.loc[valid_rows] > 0).astype(int)

# Remove columns that contain no usable values.
X = X.dropna(axis=1, how="all")

categorical_columns = [
    column for column in
    ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
    if column in X.columns
]
numeric_columns = [
    column for column in X.columns
    if column not in categorical_columns
]

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_columns),
    ("categorical", categorical_pipeline, categorical_columns)
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42
    ))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training model...")
model.fit(X_train, y_train)

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, predictions)
auc = roc_auc_score(y_test, probabilities)

print("\nTraining completed.")
print(f"Test accuracy: {accuracy:.4f}")
print(f"ROC-AUC: {auc:.4f}")
print("\nClassification report:")
print(classification_report(
    y_test,
    predictions,
    target_names=["No heart disease", "Heart disease"],
    zero_division=0
))

joblib.dump(model, MODEL_DIR / "heart_disease_model.joblib")
print(f"\nModel saved to: {MODEL_DIR / 'heart_disease_model.joblib'}")

pd.DataFrame({
    "actual": y_test.to_numpy(),
    "predicted": predictions,
    "heart_disease_probability": probabilities
}).to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)

print("Test predictions saved to outputs/test_predictions.csv")
