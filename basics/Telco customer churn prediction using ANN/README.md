# Telco Customer Churn Prediction using ANN

A binary classification project that predicts customer churn for a telecom company using an Artificial Neural Network (ANN) built with TensorFlow/Keras.

## Overview

This project uses the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) to predict whether a customer will churn based on their account details, services subscribed, and billing information.

## Dataset

- **Source:** Telco Customer Churn (Kaggle)
- **Rows:** 7,043 (11 rows dropped due to invalid `TotalCharges` values, leaving 7,032)
- **Target variable:** `Churn` (binary: Yes/No)

## Workflow

1. **Data Cleaning**
   - Dropped the `customerID` column (not predictive)
   - Standardized inconsistent categorical values (e.g. "No internet service" / "No phone service" → "No")
   - Converted `TotalCharges` from object to numeric, coercing invalid entries and dropping the resulting nulls

2. **Feature Encoding**
   - Binary-encoded all Yes/No columns (0/1)
   - Mapped `gender` to binary (Male: 0, Female: 1)
   - One-hot encoded multi-category columns: `InternetService`, `Contract`, `PaymentMethod`

3. **Exploratory Data Analysis**
   - Distribution plots for `TotalCharges` and `MonthlyCharges` (histograms with KDE)

4. **Preprocessing**
   - Feature scaling with `StandardScaler`
   - Train/test split (80/20, `random_state=42`)

5. **Model — Artificial Neural Network (Keras Sequential)**
   - Input layer: Dense(20, ReLU)
   - Output layer: Dense(1, Sigmoid)
   - Optimizer: Adam
   - Loss: Binary Crossentropy
   - Trained for 100 epochs

6. **Evaluation**
   - Classification report (precision, recall, F1-score)
   - Confusion matrix (visualized with a seaborn heatmap)

## Results

- **Test Accuracy:** ~78–80%
- **Confusion Matrix:** 912 + 191 correct predictions, 183 + 121 misclassifications

## Tech Stack

- Python
- pandas, seaborn, matplotlib
- scikit-learn (`StandardScaler`, `train_test_split`, `classification_report`, `confusion_matrix`)
- TensorFlow / Keras

## Possible Next Steps

- Experiment with additional hidden layers (already scaffolded but commented out in the notebook)
- Address class imbalance (churn is typically a minority class) with techniques like SMOTE or class weighting
- Hyperparameter tuning (learning rate, batch size, number of neurons)
- Compare ANN performance against classical models (Logistic Regression, Random Forest, XGBoost)
- Add feature importance / SHAP analysis for interpretability

## Notebook

See [`telco-customer-churn-using-ann.ipynb`](./telco-customer-churn-using-ann.ipynb) for the full code, commentary, and visualizations.
