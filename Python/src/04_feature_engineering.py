# ==========================================================
# Insurance Claim Fraud Detection
# Step 6 : Feature Engineering
# ==========================================================

import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os

# ==========================================================
# Load Cleaned Dataset
# ==========================================================

df = pd.read_csv("Python/Data/processed/cleaned_insurance_claims.csv")

print("=" * 60)
print("FEATURE ENGINEERING")
print("=" * 60)

# ==========================================================
# Remove Unnecessary Columns
# ==========================================================

drop_columns = [
    "Claim_ID",
    "Policy_Number",
    "Customer_ID",
    "Customer_Name"
]

df.drop(columns=drop_columns, inplace=True)

print("\nRemoved ID columns.")

# ==========================================================
# Convert Date Columns
# ==========================================================

date_columns = [
    "Policy_Start_Date",
    "Policy_End_Date",
    "Accident_Date",
    "Claim_Date",
    "Fraud_Risk_Score" 
]

for col in date_columns:
    df[col] = pd.to_datetime(df[col])

# ==========================================================
# Extract Date Features
# ==========================================================

df["Policy_Start_Year"] = df["Policy_Start_Date"].dt.year
df["Policy_Start_Month"] = df["Policy_Start_Date"].dt.month

df["Accident_Year"] = df["Accident_Date"].dt.year
df["Accident_Month"] = df["Accident_Date"].dt.month

df["Claim_Year"] = df["Claim_Date"].dt.year
df["Claim_Month"] = df["Claim_Date"].dt.month

# ==========================================================
# Remove Original Date Columns
# ==========================================================

df.drop(columns=date_columns, inplace=True)

print("\nDate features created.")

# ==========================================================
# Encode Categorical Columns
# ==========================================================

label_encoder = LabelEncoder()

categorical_columns = df.select_dtypes(include="object").columns

for col in categorical_columns:
    df[col] = label_encoder.fit_transform(df[col])

print("\nCategorical columns encoded.")

# ==========================================================
# Display Dataset Info
# ==========================================================

print("\nFinal Dataset Shape:")
print(df.shape)

print("\nFinal Data Types:")
print(df.dtypes)

# ==========================================================
# Save Processed Dataset
# ==========================================================

os.makedirs("Python/Data/final", exist_ok=True)

output_file = "Python/Data/final/final_dataset.csv"

df.to_csv(output_file, index=False)

print("\nFinal dataset saved successfully.")
print(output_file)

print("\nFEATURE ENGINEERING COMPLETED SUCCESSFULLY!")