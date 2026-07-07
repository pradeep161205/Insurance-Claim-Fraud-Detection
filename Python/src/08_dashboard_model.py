# ==========================================================
# Dashboard Model Training
# ==========================================================

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

print("=" * 60)
print("TRAINING DASHBOARD MODEL")
print("=" * 60)

# ----------------------------------------------------------
# Load Dataset
# ----------------------------------------------------------

df = pd.read_csv("Python\Data\processed\cleaned_insurance_claims.csv")

# ----------------------------------------------------------
# Keep only useful columns
# ----------------------------------------------------------

columns = [
    "Age",
    "Gender",
    "Occupation",
    "Annual_Income",
    "Policy_Type",
    "Premium_Amount",
    "Coverage_Amount",
    "Vehicle_Type",
    "Vehicle_Brand",
    "Vehicle_Value",
    "Days_to_Report",
    "Weather",
    "Police_Report",
    "Witness_Count",
    "Repair_Cost",
    "Medical_Cost",
    "Total_Claim_Amount",
    "Previous_Claims",
    "Previous_Fraud_History",
    "Fraud_Label"
]

df = df[columns]

# ----------------------------------------------------------
# Encode categorical columns
# ----------------------------------------------------------

encoders = {}

categorical = [
    "Gender",
    "Occupation",
    "Policy_Type",
    "Vehicle_Type",
    "Vehicle_Brand",
    "Weather",
    "Police_Report",
    "Previous_Fraud_History"
]

for col in categorical:

    le = LabelEncoder()

    df[col] = le.fit_transform(df[col])

    encoders[col] = le

# ----------------------------------------------------------
# Save Encoders
# ----------------------------------------------------------

joblib.dump(encoders, "Python/models/dashboard_encoders.pkl")

# ----------------------------------------------------------
# Split Data
# ----------------------------------------------------------

X = df.drop("Fraud_Label", axis=1)

y = df["Fraud_Label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ----------------------------------------------------------
# Scaling
# ----------------------------------------------------------

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

joblib.dump(scaler, "Python/models/dashboard_scaler.pkl")

# ----------------------------------------------------------
# Train Model
# ----------------------------------------------------------

model = RandomForestClassifier(
    random_state=42,
    n_estimators=200
)

model.fit(X_train, y_train)

# ----------------------------------------------------------
# Accuracy
# ----------------------------------------------------------

pred = model.predict(X_test)

print("Accuracy :", accuracy_score(y_test, pred))

# ----------------------------------------------------------
# Save Model
# ----------------------------------------------------------

joblib.dump(
    model,
    "Python/models/dashboard_model.pkl"
)

print("=" * 60)
print("Dashboard Model Saved Successfully")
print("=" * 60)