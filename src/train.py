# train.py
# ------------------------------
# Model Training (Data Mining) with Concept Explanations

import numpy as np
import pickle
import os

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

# ------------------------------
# Load Data
# ------------------------------
X_train = np.load("data/processed/X_train.npy", allow_pickle=True)
y_train = np.load("data/processed/y_train.npy", allow_pickle=True)

X_test = np.load("data/processed/X_test.npy", allow_pickle=True)
y_test = np.load("data/processed/y_test.npy", allow_pickle=True)

# ------------------------------
# Create Models Directory
# ------------------------------
os.makedirs("models", exist_ok=True)

# ------------------------------
# Model Selection and Reasoning
# ------------------------------
# Logistic Regression: Simple linear model, good baseline for binary classification.
# Decision Tree: Non-linear model that splits data on feature thresholds, easy to interpret.
# Random Forest: Ensemble of Decision Trees, reduces overfitting, generally higher accuracy.
# K-Nearest Neighbors (KNN): Non-parametric, predicts based on closest neighbors.
# Support Vector Machine (SVM): Finds optimal hyperplane to separate classes; works well with scaled data.
# Scaling (StandardScaler) is used for Logistic Regression, KNN, and SVM to normalize feature ranges.

models = {
    "logistic_regression": Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(max_iter=2000))
    ]),
    "decision_tree": DecisionTreeClassifier(random_state=42),
    "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "k_nearest_neighbors": Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier())
    ]),
    "support_vector_machine": Pipeline([
        ('scaler', StandardScaler()),
        ('svc', SVC(probability=True, class_weight='balanced'))
    ])
}

print("Training models...\n")

# ------------------------------
# Train, Evaluate, and Save Models
# ------------------------------
for name, model in models.items():
    # Train the model
    model.fit(X_train, y_train)

    # ------------------------------
    # Cross-Validation
    # ------------------------------
    # Cross-validation (5-fold) is used to check how well the model generalizes to unseen data.
    # It splits the training data into 5 parts, trains on 4 parts, tests on 1, and averages results.
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    print(f"\n{name} 5-Fold CV Accuracy: {cv_scores.mean():.4f}")

    # ------------------------------
    # Evaluate on Test Set
    # ------------------------------
    # Accuracy: Overall percentage of correct predictions.
    # F1-Score: Harmonic mean of precision and recall, balances false positives and false negatives.
    # Confusion Matrix: Shows true positives, false positives, true negatives, false negatives.
    # Classification Report: Precision, recall, F1 for each class, and weighted averages.
    y_pred = model.predict(X_test)
    print(f"\n--- {name} Evaluation on Test Set ---")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred, average='weighted'))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=0))

    # ------------------------------
    # Save the model
    # ------------------------------
    with open(f"models/{name}.pkl", "wb") as f:
        pickle.dump(model, f)

    print(f"✓ {name} trained, evaluated, and saved\n")

print("\nTraining completed successfully.")
