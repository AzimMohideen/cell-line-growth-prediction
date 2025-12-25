import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from xgboost import XGBClassifier

# Create results folder
os.makedirs("../results", exist_ok=True)

# Load processed dataset
df = pd.read_csv("../data/processed.csv")

# Separate features and target
X = df.iloc[:, :-1].values   # NumPy array (important for XGBoost)
y = df.iloc[:, -1]

# Encode target labels
le = LabelEncoder()
y = le.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train XGBoost model
model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42,
    eval_metric="logloss"
)

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)

# Classification report
report = classification_report(
    y_test,
    y_pred,
    target_names=le.classes_
)
print("\nClassification Report:\n", report)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Save metrics
with open("../results/metrics.txt", "w") as f:
    f.write(f"Accuracy: {acc}\n\n")
    f.write("Classification Report:\n")
    f.write(report)

# ------------------ GRAPH 1: CONFUSION MATRIX ------------------
plt.figure(figsize=(5,4))
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.colorbar()

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.xticks(range(len(le.classes_)), le.classes_)
plt.yticks(range(len(le.classes_)), le.classes_)
plt.tight_layout()
plt.savefig("../results/confusion_matrix.png")
plt.close()

# ------------------ GRAPH 2: FEATURE IMPORTANCE ------------------
importances = model.feature_importances_
top_idx = np.argsort(importances)[-20:]

plt.figure(figsize=(7,4))
plt.barh(range(len(top_idx)), importances[top_idx])
plt.title("Top 20 Feature Importances")
plt.xlabel("Importance")
plt.ylabel("Feature Index")
plt.tight_layout()
plt.savefig("../results/feature_importance.png")
plt.close()

# ------------------ GRAPH 3: CLASS DISTRIBUTION ------------------
unique, counts = np.unique(y, return_counts=True)

plt.figure(figsize=(4,4))
plt.bar(le.inverse_transform(unique), counts)
plt.title("Class Distribution")
plt.xlabel("Growth Property")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("../results/class_distribution.png")
plt.close()

print("\nGraphs saved in results/ folder")
