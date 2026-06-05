# Behavioral Fraud Detection on Imbalanced Financial Data

## Overview
This repository contains a machine learning pipeline engineered to identify fraudulent transactions within a highly imbalanced dataset (fraud rate < 0.2%). The project focuses on cost-sensitive learning, advanced evaluation metrics, and model interpretability.

## Key Features & Techniques
* **Handling Extreme Imbalance:** Applied **SMOTE** (Synthetic Minority Over-sampling Technique) to the training data to maximize minority-class recall without causing data leakage.
* **Cost-Sensitive Learning:** Utilized an **XGBoost** classifier, tuning the `scale_pos_weight` hyperparameter to heavily penalize misclassifications of the minority (fraudulent) class.
* **Robust Evaluation:** Standard accuracy is an unreliable metric for highly imbalanced data (the "accuracy paradox"). This model is evaluated strictly using **PR-AUC (Precision-Recall Area Under Curve)** and **Macro F1-score**.
* **Interpretability:** Integrated **SHAP (SHapley Additive exPlanations)** to determine feature importance, ensuring the model's decisions are transparent and mathematically explainable.

## Dataset
* **Source:** Kaggle Credit Card Fraud Detection Dataset
* **Size:** ~284,807 transactions (492 frauds, 284,315 valid).

## Tech Stack
* `Python`
* `XGBoost`
* `Scikit-Learn` / `Imbalanced-Learn`
* `SHAP`
* `Pandas` / `NumPy`
