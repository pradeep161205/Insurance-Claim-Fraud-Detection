# ==========================================================
# Insurance Claim Fraud Detection
# Step 5 : Exploratory Data Analysis (EDA)
# ==========================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==========================================================
# Load Cleaned Dataset
# ==========================================================

df = pd.read_csv("Python/Data/processed/cleaned_insurance_claims.csv")

# ==========================================================
# Create Folder to Save Graphs
# ==========================================================

os.makedirs("Python/reports/figures", exist_ok=True)

# ==========================================================
# Graph Style
# ==========================================================

sns.set_style("whitegrid")

print("="*60)
print("EXPLORATORY DATA ANALYSIS")
print("="*60)

# ==========================================================
# 1. Fraud Distribution
# ==========================================================

plt.figure(figsize=(6,5))

sns.countplot(
    data=df,
    x="Fraud_Label",
    hue="Fraud_Label",
    palette="Set2",
    legend=False
)

plt.title("Fraud vs Non-Fraud Claims")
plt.xlabel("Fraud Label")
plt.ylabel("Number of Claims")

plt.savefig("Python/reports/figures/01_fraud_distribution.png")
plt.show()

# ==========================================================
# 2. Age Distribution
# ==========================================================

plt.figure(figsize=(8,5))

sns.histplot(
    data=df,
    x="Age",
    bins=20,
    kde=True,
    color="skyblue"
)

plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")

plt.savefig("Python/reports/figures/02_age_distribution.png")
plt.show()

# ==========================================================
# 3. Gender Distribution
# ==========================================================

plt.figure(figsize=(6,5))

sns.countplot(
    data=df,
    x="Gender",
    hue="Gender",
    palette="pastel",
    legend=False
)

plt.title("Gender Distribution")

plt.savefig("Python/reports/figures/03_gender_distribution.png")
plt.show()

# ==========================================================
# 4. Occupation Distribution
# ==========================================================

plt.figure(figsize=(10,6))

sns.countplot(
    data=df,
    y="Occupation",
    order=df["Occupation"].value_counts().index,
    palette="viridis"
)

plt.title("Occupation Distribution")

plt.savefig("Python/reports/figures/04_occupation_distribution.png")
plt.show()

# ==========================================================
# 5. Policy Type Distribution
# ==========================================================

plt.figure(figsize=(8,5))

sns.countplot(
    data=df,
    x="Policy_Type",
    order=df["Policy_Type"].value_counts().index,
    palette="Set3"
)

plt.xticks(rotation=20)

plt.title("Policy Type Distribution")

plt.savefig("Python/reports/figures/05_policy_type_distribution.png")
plt.show()

# ==========================================================
# 6. Vehicle Type Distribution
# ==========================================================

plt.figure(figsize=(8,5))

sns.countplot(
    data=df,
    x="Vehicle_Type",
    order=df["Vehicle_Type"].value_counts().index,
    palette="coolwarm"
)

plt.xticks(rotation=20)

plt.title("Vehicle Type Distribution")

plt.savefig("Python/reports/figures/06_vehicle_type_distribution.png")
plt.show()

# ==========================================================
# 7. Weather Distribution
# ==========================================================

plt.figure(figsize=(8,5))

sns.countplot(
    data=df,
    x="Weather",
    order=df["Weather"].value_counts().index,
    palette="Spectral"
)

plt.xticks(rotation=20)

plt.title("Weather Conditions")

plt.savefig("Python/reports/figures/07_weather_distribution.png")
plt.show()

# ==========================================================
# 8. Claim Amount Distribution
# ==========================================================

plt.figure(figsize=(8,5))

sns.histplot(
    data=df,
    x="Total_Claim_Amount",
    bins=30,
    kde=True,
    color="green"
)

plt.title("Total Claim Amount Distribution")

plt.savefig("Python/reports/figures/08_claim_amount_distribution.png")
plt.show()

# ==========================================================
# 9. Correlation Heatmap
# ==========================================================

plt.figure(figsize=(12,10))

numeric_df = df.select_dtypes(include=["int64","float64"])

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")

plt.savefig("Python/reports/figures/09_correlation_heatmap.png")
plt.show()

# ==========================================================
# 10. Boxplot - Claim Amount
# ==========================================================

plt.figure(figsize=(8,5))

sns.boxplot(
    data=df,
    x="Total_Claim_Amount",
    color="orange"
)

plt.title("Outlier Detection - Claim Amount")

plt.savefig("Python/reports/figures/10_boxplot_claim_amount.png")
plt.show()

# ==========================================================
# 11. Fraud vs Claim Amount
# ==========================================================

plt.figure(figsize=(8,5))

sns.boxplot(
    data=df,
    x="Fraud_Label",
    y="Total_Claim_Amount",
    palette="Set1"
)

plt.title("Fraud vs Claim Amount")

plt.savefig("Python/reports/figures/11_fraud_vs_claim_amount.png")
plt.show()

# ==========================================================
# 12. Previous Claims vs Fraud
# ==========================================================

plt.figure(figsize=(8,5))

sns.countplot(
    data=df,
    x="Previous_Claims",
    hue="Fraud_Label",
    palette="Set2"
)

plt.title("Previous Claims vs Fraud")

plt.savefig("Python/reports/figures/12_previous_claims_vs_fraud.png")
plt.show()

print("="*60)
print("EDA COMPLETED SUCCESSFULLY")
print("All graphs saved in reports/figures")
print("="*60)