# Telecom Customer Churn Prediction

**Authors:**  
Aun Mustansar Hussain  
Muneer Abbas  

## 📌 Project Overview
Customer churn is a major challenge in the telecom industry. This project applies machine learning techniques to predict whether a customer is likely to churn based on historical usage and billing data.

The goal is to help telecom companies take proactive retention actions using data-driven insights.

---

## 🎯 Objectives
- Predict customer churn using machine learning models
- Compare multiple classifiers to select the best-performing model
- Evaluate performance using Accuracy, Precision, Recall, F1-score, and ROC-AUC
- Provide visual analytics and an inference pipeline for business use
- Support sustainable business growth aligned with UN SDGs

---

## 🗂 Dataset
- Customers: 7,032
- Features: Tenure, charges, contract type, payment method, internet service, etc.
- Target Variable: Churn (0 = Stay, 1 = Churn)

---

## ⚙️ Models Used
- Logistic Regression
- Decision Tree
- Random Forest
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)

---

## 📊 Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- ROC-AUC
- 5-Fold Cross Validation

---

## 🏆 Best Model
**Random Forest** achieved the best overall performance in terms of accuracy and F1-score.

---

## 🔍 Inference
The trained Random Forest model is used to predict churn for new customers by loading the saved model and feature space.

---

## 🧪 How to Run
```bash
pip install -r requirements.txt
python src/train.py
python src/evaluation.py
python src/inference.py
