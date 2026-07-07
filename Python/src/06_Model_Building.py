# ==========================================================
# Insurance Claim Fraud Detection
# Step 8 : Model Building
# ==========================================================

import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# ==========================================================
# Load Train-Test Data
# ==========================================================

X_train = joblib.load("Python/models/X_train.pkl")
X_test = joblib.load("Python/models/X_test.pkl")
y_train = joblib.load("Python/models/y_train.pkl")
y_test = joblib.load("Python/models/y_test.pkl")

print("=" * 60)
print("MODEL BUILDING")
print("=" * 60)

# ==========================================================
# Models
# ==========================================================

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "XGBoost": XGBClassifier(
        random_state=42,
        eval_metric="logloss"
    )
}

results = []

best_model = None
best_accuracy = 0

# ==========================================================
# Train & Evaluate
# ==========================================================

for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    results.append([
        name,
        accuracy,
        precision,
        recall,
        f1
    ])

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model

# ==========================================================
# Results Table
# ==========================================================

results_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ]
)

print("\n")
print("=" * 60)
print(results_df)
print("=" * 60)

# ==========================================================
# Save Best Model
# ==========================================================

joblib.dump(
    best_model,
    "Python/models/best_fraud_model.pkl"
)

print("\nBest model saved successfully!")

print("=" * 60)
print("MODEL BUILDING COMPLETED")
print("=" * 60)