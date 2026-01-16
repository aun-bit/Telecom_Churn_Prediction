# evaluation.py
# ------------------------------
# Full Model Evaluation & Visualizations
# Metrics: Accuracy, Precision, Recall, F1, Confusion Matrix, ROC-AUC

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    roc_auc_score
)

# -------------------------------
# Load test data
X_test = np.load("data/processed/X_test.npy", allow_pickle=True)
y_test = np.load("data/processed/y_test.npy", allow_pickle=True)

# Ensure plots directory exists
os.makedirs("plots", exist_ok=True)

# Store metrics for summary chart
model_names = []
accuracy_scores = []
f1_scores = []

print("\nEvaluating models...\n")

# -------------------------------
# Evaluate all models
for model_file in sorted(os.listdir("models")):
    if not model_file.endswith(".pkl"):
        continue

    # Load model
    model_path = os.path.join("models", model_file)
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    model_name = model_file.replace(".pkl", "")
    model_names.append(model_name)

    # Predictions
    y_pred = model.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    accuracy_scores.append(acc)
    f1_scores.append(f1)

    # ---------------- Confusion Matrix ----------------
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"plots/{model_name}_confusion_matrix.png")
    plt.close()

    # ---------------- Precision, Recall, F1 ----------------
    plt.figure(figsize=(5,4))
    plt.bar(['Precision', 'Recall', 'F1 Score'], [prec, rec, f1], color=['skyblue','orange','green'])
    plt.ylim(0,1)
    plt.title(f"{model_name} - Precision/Recall/F1")
    plt.tight_layout()
    plt.savefig(f"plots/{model_name}_precision_recall_f1.png")
    plt.close()

    # ---------------- ROC-AUC ----------------
    try:
        y_proba = model.predict_proba(X_test)[:,1]
        auc_score = roc_auc_score(y_test, y_proba)
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.figure(figsize=(5,4))
        plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}", color='purple')
        plt.plot([0,1], [0,1], 'k--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{model_name} ROC Curve")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(f"plots/{model_name}_roc_auc.png")
        plt.close()
    except AttributeError:
        print(f"⚠️ {model_name} does not support predict_proba, skipping ROC-AUC")

    # ---------------- Console Output ----------------
    print(f"Model: {model_name}")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"✓ Plots saved for {model_name}\n")

# -------------------------------
# Summary Bar Chart: Accuracy & F1 (All Models)
plt.figure(figsize=(10,6))
bar_width = 0.4
x = np.arange(len(model_names))
plt.bar(x - bar_width/2, accuracy_scores, width=bar_width, color='skyblue', label='Accuracy')
plt.bar(x + bar_width/2, f1_scores, width=bar_width, color='orange', alpha=0.7, label='F1 Score')
plt.xticks(x, model_names, rotation=20)
plt.ylim(0,1)
plt.ylabel("Score")
plt.title("Model Performance Metrics (Accuracy & F1)")
plt.legend()
plt.tight_layout()
plt.savefig("plots/model_metrics_summary.png")
plt.show()

print("\n✅ Evaluation complete. All plots saved in 'plots/' folder.")
