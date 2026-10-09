
from ucimlrepo import fetch_ucirepo

print("Downloading Heart Disease dataset...")

heart_disease = fetch_ucirepo(id=45)

X = heart_disease.data.features
y = heart_disease.data.targets

print("\nDataset loaded successfully.")
print("Feature shape:", X.shape)
print("Target shape:", y.shape)

print("\nFeature names:")
print(list(X.columns))

print("\nFirst five rows:")
print(X.head())

print("\nMissing values per column:")
print(X.isnull().sum())

print("\nTarget distribution:")
print(y.iloc[:, 0].value_counts(dropna=False).sort_index())
