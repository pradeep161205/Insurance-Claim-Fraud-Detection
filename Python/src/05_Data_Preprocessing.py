# ==========================================================
# Insurance Claim Fraud Detection
# Step 7 : Data Preprocessing & Train-Test Split
# ==========================================================

import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("=" * 60)
print("DATA PREPROCESSING")
print("=" * 60)

# ==========================================================
# Load Final Dataset
# ==========================================================

df = pd.read_csv("Python/Data/final/final_dataset.csv")

print("\nDataset Shape:")
print(df.shape)

# ==========================================================
# Define Features and Target
# ==========================================================

X = df.drop("Fraud_Label", axis=1)
y = df["Fraud_Label"]

print("\nFeatures Shape :", X.shape)
print("Target Shape   :", y.shape)

# ==========================================================
# Train-Test Split
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Samples :", len(X_train))
print("Testing Samples  :", len(X_test))

# ==========================================================
# Feature Scaling
# ==========================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nFeature Scaling Completed.")

# ==========================================================
# Save Scaler
# ==========================================================

os.makedirs("Python/models", exist_ok=True)

joblib.dump(
    scaler,
    "Python/models/scaler.pkl"
)

print("\nScaler Saved Successfully!")

# ==========================================================
# Save Processed Data
# ==========================================================

joblib.dump(X_train_scaled, "Python/models/X_train.pkl")
joblib.dump(X_test_scaled, "Python/models/X_test.pkl")
joblib.dump(y_train, "Python/models/y_train.pkl")
joblib.dump(y_test, "Python/models/y_test.pkl")

print("\nTrain-Test datasets saved successfully!")

print("=" * 60)
print("PREPROCESSING COMPLETED SUCCESSFULLY")
print("=" * 60)