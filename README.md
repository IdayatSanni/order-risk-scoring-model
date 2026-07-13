# Order Risk Scoring Model

## Overview

This project develops a machine learning model to predict whether an online order is **high-risk** (potential fraud or payment default) using customer, transaction, behavioral, and historical purchasing information.

The objective is to assist e-commerce businesses in identifying risky transactions before order fulfillment, reducing financial loss while maintaining a positive customer experience.

---

## Project Objectives

- Explore and understand customer purchasing behavior
- Perform exploratory data analysis (EDA)
- Engineer meaningful predictive features
- Train classification models
- Evaluate model performance using appropriate metrics for an imbalanced dataset
- Identify the most important drivers of order risk

---

## Dataset

The dataset contains approximately **30,000 online purchase orders** with **44 attributes** describing:

- Customer information
- Payment information
- Order characteristics
- Historical purchasing behavior
- Fraud verification checks
- Previous payment history

### Target Variable

| Variable | Description |
|----------|-------------|
| `CLASS` | High-risk order (Yes / No) |

---

## Dataset Features

### Customer Information

- Email submitted
- Phone number submitted
- Birthdate
- New customer indicator

### Order Information

- Order value
- Number of items
- Order weekday
- Order time

### Payment Information

- Payment method
- Card type
- Card validity

### Fraud Detection Checks

- IP verification
- Cookie verification
- Card verification
- Bank account verification
- Address verification

### Historical Customer Behavior

- Previous order value
- Previous order quantity
- Previous payment reminders

---

## Exploratory Data Analysis

The project includes extensive exploratory data analysis to understand feature distributions and identify predictive variables.

Key analyses include:

- Missing value analysis
- Class imbalance assessment
- Categorical feature analysis
- Numerical feature analysis
- Risk rate comparisons
- Feature importance investigation
- Correlation analysis

Several new features were engineered during the analysis, including:

- Age groups
- Order value categories
- Purchase amount categories
- Time-of-day buckets

---

## Key Findings

Several variables demonstrated strong relationships with order risk.

| Feature | Observation |
|----------|-------------|
| `CHK_IP` | Highest fraud risk among verification checks |
| `CHK_COOKIE` | Strong indicator of risky transactions |
| `NEUKUNDE` | New customers showed higher default rates |
| `B_EMAIL` | Missing email associated with increased risk |
| `B_TELEFON` | Missing phone number increased risk |
| `DATE_LORDER` | Customers without previous orders showed higher risk |

These insights informed feature selection for model development.

---

## Machine Learning Workflow

1. Data Cleaning
2. Missing Value Handling
3. Feature Engineering
4. Encoding Categorical Variables
5. Train/Test Split
6. Model Training
7. Performance Evaluation
8. Feature Importance Analysis

---

## Evaluation Metrics

Because the dataset is imbalanced, multiple evaluation metrics are used instead of relying solely on accuracy.

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Jupyter Notebook

---

## Repository Structure

```text
order-risk-scoring-model/
│
├── risk-train.txt              # Training dataset
├── risk-attributes.txt         # Dataset documentation
├── churn_prediction.ipynb      # EDA and model development
├── train.py                    # Model training script
├── main.py                     # Entry point
└── README.md
```

---

## Future Improvements

- Hyperparameter optimization
- Cross-validation
- XGBoost implementation
- LightGBM implementation
- Explainable AI using SHAP values
- Model deployment using Flask or FastAPI
- Real-time risk scoring API

---

## Results

This project demonstrates how machine learning can support fraud detection by leveraging customer behavior, payment information, and transaction history to identify potentially high-risk online orders before fulfillment.

---

## Author

**Idayat Sanni**

Artificial Intelligence & Web Development Graduate

- LinkedIn
- GitHub
- Portfolio
