# Automated Approval Agent
A Machine Learning + Streamlit Project for Health Insurance Applicant Processing

This project builds an AI agent that predicts expected medical costs for new applicants and autonomously decides whether to auto‑approve, flag for manual review, or reject an application based on risk.

## Dataset Justification
The selected dataset contains applicant demographic and health‑related attributes commonly used by insurance providers to estimate medical expenses. It is appropriate because:

Includes age, BMI, smoking status, region, number of children, etc.

These features have direct correlation with medical charges.

Widely used for health‑cost prediction benchmarks.

Provides enough samples for training, testing, and validation.

This makes it suitable for building a cost‑prediction model that supports automated decision‑making.

## Dataset Processing
After downloading the dataset, the following preprocessing steps were performed:

1. Data Cleaning
Handle missing values

Remove duplicates

Standardize column names

2. Feature Engineering
One‑hot encode categorical variables (e.g., smoker, region)

Create interaction features (e.g., BMI × smoker)

Analyze correlations to identify impactful features on medical charges

3. Scaling
Normalize or standardize numerical features

Ensure consistent scaling for model training and inference

These steps prepare the dataset for accurate and stable model performance.

## Algorithm Selection
Several algorithms were evaluated for predicting medical costs:

Chosen Algorithms
Linear Regression

Baseline model

Easy to interpret

Random Forest Regressor

Handles non‑linear relationships

Robust against overfitting

XGBoost (optional)

High performance for tabular data

Excellent handling of feature interactions

Why These Algorithms?
Perform well on structured numerical + categorical datasets

Allow comparison between simple and complex models

Provide flexibility for tuning and improving accuracy

## Workflow Explanation
A complete end‑to‑end workflow was followed:

1. Data Collection
Download dataset → load into pandas → inspect structure.

2. Data Processing
Clean → encode → scale → engineer features → analyze correlations.

3. Train/Test Split
Split dataset (e.g., 80/20) to evaluate generalization.

4. Model Training
Train selected algorithms → tune hyperparameters → compare performance.

5. Evaluation
Metrics used:

MAE

RMSE

R² score

Model behavior checks:

Overfitting → high train accuracy, low test accuracy

Underfitting → low accuracy on both

Models were tuned to achieve balanced performance.

6. Save Best Model
python
import joblib
joblib.dump(model, "best_model.joblib")
7. Deployment
Integrate the trained model into a Streamlit web application.

## Final Demo (Streamlit UI)
The Streamlit interface acts as the front‑end for the AI agent.

Input (Perception Layer)
Users enter applicant data via:

Text fields

Dropdowns

Sliders

Radio buttons

Output (Reasoning Layer)
The model predicts the expected medical cost.

Agent Action (Act Layer)
Based on the predicted cost, the agent displays decisions using Streamlit alert boxes:

Auto‑Approved

Flagged for Manual Actuary Review

Application Rejected – High Risk

This creates a complete automated approval workflow.
