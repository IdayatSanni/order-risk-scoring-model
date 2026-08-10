# Order Risk Scoring Model

An end-to-end binary classification machine learning project that predicts whether an online order is **high-risk** or **low-risk** using customer behavior, transaction history, payment validation, and fraud-related indicators.

## Project Overview

The dataset contains **30,000 online purchase orders** with **44 features**. The target variable (`CLASS`) indicates whether an order is considered high-risk (`yes`) or low-risk (`no`). Because only **5.8% of orders are high-risk**, the project focuses on handling **class imbalance** and improving the model’s ability to detect minority-class cases.

## Workflow

* Exploratory Data Analysis (EDA)
* Numerical and categorical feature analysis
* Feature engineering (`AGE_GROUP`, `TIME_BUCKET`, `HAS_PREVIOUS_ORDER`)
* Missing value strategy and feature selection
* One-Hot Encoding and Standard Scaling
* Train/test split with stratification
* Model comparison
* Hyperparameter tuning
* SMOTE oversampling
* Performance evaluation using precision, recall, F1-score, and confusion matrices

## Models Evaluated

| Model                       | F1 (High-Risk Class) |
| --------------------------- | -------------------: |
| Logistic Regression         |                 0.02 |
| Decision Tree               |                 0.16 |
| Random Forest               |                 0.01 |
| Gradient Boosting           |                 0.02 |
| Tuned Decision Tree         |             **0.25** |
| SMOTE + Logistic Regression |                 0.19 |

## Key Findings

* **Class imbalance significantly affected model performance**, with baseline models predicting very few high-risk orders.
* **Hyperparameter tuning produced the best overall improvement**, increasing the minority-class F1-score from **0.16 to 0.25**.
* **SMOTE greatly improved recall (0.69)** but introduced many false positives, reducing overall precision and accuracy.
* Features related to **previous order history, payment reminders, IP validation, and customer status** showed stronger predictive value than demographic features such as age.

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* imbalanced-learn (SMOTE)

## What I Learned

This project strengthened my understanding of:

* feature engineering for tabular data,
* preprocessing pipelines with `ColumnTransformer`,
* one-hot encoding vs. label encoding,
* feature scaling,
* binary classification evaluation,
* handling imbalanced datasets,
* hyperparameter tuning,
* and the trade-off between precision and recall in fraud detection systems.
