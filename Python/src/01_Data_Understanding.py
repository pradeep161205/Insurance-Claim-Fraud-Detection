# ============================================================
# Insurance Claim Fraud Detection
# Step 1: Data Understanding
# ============================================================

# Import Libraries
import pandas as pd
import numpy as np

# Load Dataset
df = pd.read_csv("Python\Data\processed\cleaned_insurance_claims.csv")

# ============================================================
# Basic Information
# ============================================================

print("=" * 60)
print("Insurance Claim Fraud Detection Dataset")
print("=" * 60)

print("\nFirst 5 Rows:")
print(df.head())

print("\nLast 5 Rows:")
print(df.tail())

print("\nRandom Sample:")
print(df.sample(5))

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

# ============================================================
# Dataset Information
# ============================================================

print("\nDataset Information:")
df.info()

# ============================================================
# Statistical Summary
# ============================================================

print("\nNumerical Summary:")
print(df.describe())

print("\nCategorical Summary:")
print(df.describe(include="object"))

# ============================================================
# Missing Values
# ============================================================

print("\nMissing Values:")
print(df.isnull().sum())

# ============================================================
# Duplicate Records
# ============================================================

print("\nDuplicate Records:")
print(df.duplicated().sum())

# ============================================================
# Fraud Distribution
# ============================================================

print("\nFraud Distribution:")
print(df["Fraud_Label"].value_counts())

print("\nFraud Percentage:")
print((df["Fraud_Label"].value_counts(normalize=True) * 100).round(2))

print("\nData Understanding Completed Successfully!")