# inference.py
# ------------------------------
# Inference using trained ML models with Concept Explanations

import numpy as np
import pickle

# ------------------------------
# Load Trained Model
# ------------------------------
# You can change this to any model you trained
model_name = "random_forest"  
with open(f"models/{model_name}.pkl", "rb") as f:
    model = pickle.load(f)

# ------------------------------
# Load Feature Names
# ------------------------------
# Feature names must match the order used in training
with open("data/processed/feature_names.txt", "r") as f:
    feature_names = [line.strip() for line in f.readlines()]

print(f"Model expects {len(feature_names)} features")

# ------------------------------
# Prepare New Sample
# ------------------------------
# Initialize empty feature vector
X_new = np.zeros((1, len(feature_names)))

# Example new customer feature values
feature_values = {
    "SeniorCitizen": 0,
    "tenure": 45,
    "MonthlyCharges": 75.5,
    "TotalCharges": 3400,
    "Contract_Two year": 1,
    "PaymentMethod_Credit card (automatic)": 1,
    "InternetService_Fiber optic": 1
}

# Fill feature vector based on feature names
for feature, value in feature_values.items():
    if feature in feature_names:
        index = feature_names.index(feature)
        X_new[0, index] = value

# ------------------------------
# Predict
# ------------------------------
# Demonstrates real-world application of the model
prediction = model.predict(X_new)
probability = model.predict_proba(X_new)[0][1]  # Probability of churn (class 1)

# ------------------------------
# Output Result
# ------------------------------
if prediction[0] == 1:
    print(f"⚠️ Customer is likely to CHURN (probability: {probability:.2f})")
else:
    print(f"✅ Customer is likely to STAY (probability: {1-probability:.2f})")
