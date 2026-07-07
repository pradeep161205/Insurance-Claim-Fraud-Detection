# ============================================================
# Insurance Claim Fraud Detection
# Step 4 : Data Cleaning
# ============================================================

import pandas as pd

# Load Dataset
df = pd.read_csv("Python/Data/processed/insurance_claim_fraud_dataset.csv")

print("=" * 60)
print("DATA CLEANING")
print("=" * 60)

# ============================================================
# 1. Dataset Shape
# ============================================================

print("\nDataset Shape:")
print(df.shape)

# ============================================================
# 2. Missing Values
# ============================================================

print("\nMissing Values Before Cleaning:")
print(df.isnull().sum())

# ============================================================
# 3. Handle Missing FIR Number
# ============================================================

df["FIR_Number"] = df["FIR_Number"].fillna("Not Available")

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

# ============================================================
# 4. Remove Duplicate Records
# ============================================================

duplicates = df.duplicated().sum()
print("\nDuplicate Records:", duplicates)

df.drop_duplicates(inplace=True)

# ============================================================
# 5. Convert Date Columns
# ============================================================

date_columns = [
    "Policy_Start_Date",
    "Policy_End_Date",
    "Accident_Date",
    "Claim_Date"
]

for column in date_columns:
    df[column] = pd.to_datetime(df[column])

print("\nDate Columns Converted Successfully!")

# ============================================================
# 6. Verify Data Types
# ============================================================

print("\nUpdated Data Types:")
print(df.dtypes)

# ============================================================
# 7. Save Cleaned Dataset
# ============================================================

df.to_csv(
    "Python/Data/processed/cleaned_insurance_claims.csv",
    index=False
)

print("\nCleaned dataset saved successfully!")
print("\nData Cleaning Completed!")