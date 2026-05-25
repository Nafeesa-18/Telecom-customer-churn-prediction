# Telecom Customer Churn Prediction

## 📌 Project Overview

Telecom Customer Churn Prediction is a Machine Learning project that predicts whether a customer is likely to leave a telecom service provider based on customer demographics, subscription details, and usage patterns.

The project uses Logistic Regression and a real-world telecom customer dataset to identify customers at risk of churn. A Streamlit web application is also developed to provide an interactive prediction interface.

## 🎯 Objectives

- Analyze telecom customer data.
- Identify factors affecting customer churn.
- Build a Machine Learning model to predict churn.
- Help telecom companies improve customer retention strategies.

## 📊 Dataset Information

The dataset contains customer information such as:

- Gender
- Age
- Tenure in Months
- Offer
- Internet Type
- Contract Type
- Payment Method
- Monthly Charge
- Satisfaction Score
- Customer Lifetime Value (CLTV)

### Target Variable

Customer Status

- 0 → Customer Stayed / Joined
- 1 → Customer Churned

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- Joblib
- Streamlit

## Machine Learning Workflow

 1. Data Collection
   Collected telecom customer churn dataset.

 2. Data Preprocessing
- Removed unnecessary columns
- Handled missing values
- Encoded categorical variables using Label Encoding
- Scaled numerical features using StandardScaler

 3. Feature Selection

The final model uses the following features:

1. Gender
2. Age
3. Tenure in Months
4. Offer
5. Internet Type
6. Contract
7. Payment Method
8. Monthly Charge
9. Satisfaction Score
10. CLTV

### 4. Model Training

Algorithm Used:**Logistic Regression**

### 5. Model Evaluation

Performance Metrics:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

### Accuracy

**94.11%**

### Confusion Matrix

```text
[[996  13]
 [ 70 330]]
```

### Classification Report

```text
Precision: 0.96
Recall:    0.82
F1-Score:  0.89
```

## Streamlit Application Features

- User-friendly interface
- Customer churn prediction
- Churn probability estimation
- Real-time prediction
- Interactive input forms

## Future Enhancements

- Random Forest Classifier
- XGBoost Model
- Feature Importance Visualization
- Advanced Analytics Dashboard
- Cloud Deployment

## Learning Outcomes

Through this project, the following concepts were implemented:

- Data Cleaning
- Data Preprocessing
- Feature Engineering
- Label Encoding
- Feature Scaling
- Logistic Regression
- Model Evaluation
- Model Serialization
- Streamlit Deployment

---

## Author

**Nafeesa Hajira**

B.Tech – Computer Science Engineering (Artificial Intelligence & Machine Learning)

## Project Result

The developed Telecom Customer Churn Prediction model achieved an accuracy of **94.11%**, demonstrating strong performance in identifying customers likely to churn and supporting data-driven customer retention strategies.

##Intern ID
CITS1245

