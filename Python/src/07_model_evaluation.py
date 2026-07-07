# ==========================================================
# Insurance Claim Fraud Detection
# Step 9 : Model Evaluation
# ==========================================================

import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score
)

# Load Model
model = joblib.load("Python/models/best_fraud_model.pkl")

# Load Test Data
X_test = joblib.load("Python/models/X_test.pkl")
y_test = joblib.load("Python/models/y_test.pkl")

# Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

# Classification Report
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Not Fraud","Fraud"],
            yticklabels=["Not Fraud","Fraud"])

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
auc = roc_auc_score(y_test, y_prob)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
plt.plot([0,1],[0,1],"r--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()
plt.show()

print(f"\nROC-AUC Score : {auc:.4f}")

print("\nMODEL EVALUATION COMPLETED")