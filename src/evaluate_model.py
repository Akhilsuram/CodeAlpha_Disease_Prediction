
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"
PLOT_DIR = OUTPUT_DIR / "plots"
PREDICTIONS_FILE = OUTPUT_DIR / "test_predictions.csv"

PLOT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(PREDICTIONS_FILE)
y_true = df["actual"]
y_pred = df["predicted"]
y_prob = df["heart_disease_probability"]

accuracy = accuracy_score(y_true, y_pred)
auc = roc_auc_score(y_true, y_prob)

report = classification_report(
    y_true,
    y_pred,
    target_names=["No heart disease", "Heart disease"],
    zero_division=0
)

print("\nMODEL EVALUATION")
print("----------------")
print(f"Accuracy: {accuracy:.4f}")
print(f"ROC-AUC:  {auc:.4f}")
print("\nClassification Report:")
print(report)

(OUTPUT_DIR / "evaluation_report.txt").write_text(
    f"Accuracy: {accuracy:.4f}\n"
    f"ROC-AUC: {auc:.4f}\n\n"
    f"Classification Report:\n{report}",
    encoding="utf-8"
)

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(7, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No disease", "Disease"],
    yticklabels=["No disease", "Disease"]
)
plt.title("Heart Disease Prediction - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(PLOT_DIR / "confusion_matrix.png", dpi=150)
plt.close()

fpr, tpr, _ = roc_curve(y_true, y_prob)
plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f"ROC-AUC = {auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--", label="Random baseline")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Heart Disease Prediction - ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig(PLOT_DIR / "roc_curve.png", dpi=150)
plt.close()

print("\nSaved:")
print("outputs/evaluation_report.txt")
print("outputs/plots/confusion_matrix.png")
print("outputs/plots/roc_curve.png")
