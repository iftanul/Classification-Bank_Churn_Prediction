# 🏦 Bank Customer Churn Prediction  
An End-to-End Machine Learning Project with Business-Oriented Deployment  

---

## 🚀 Live Interactive Application

Experience the full churn prediction system in action:

👉 https://classification-bankchurnprediction-projects.streamlit.app/

This deployed application demonstrates:

- End-to-end ML pipeline integration
- Real-time churn probability scoring
- Business-focused risk interpretation
- Decision-support recommendation logic

---

## 📌 Executive Summary

Customer churn is one of the most critical challenges in the banking industry.  
When customers stop using banking services, institutions lose revenue, increase acquisition costs, and reduce long-term profitability.

This project develops a machine learning solution to:

- Detect customers at risk of churn  
- Identify early behavioral warning signals  
- Support targeted retention strategies  
- Improve marketing efficiency and cost control  

The solution is deployed as an interactive Streamlit web application to simulate real-world decision-making scenarios.

---

## 🎯 Business Objective

The objective of this project is to build a churn prediction system that:

- Maximizes **Recall (Churn)** to reduce undetected customer loss  
- Maintains **Precision** to avoid inefficient retention spending  
- Provides interpretable insights for business stakeholders  

In churn prediction:

- False Negative = Customer churn not detected  
- Undetected churn = Direct revenue loss  

Therefore, recall becomes a primary evaluation metric.

---

## 📊 Dataset Overview

- Total customers: 10,127  
- Target variable: `attrition_flag`  
- Class distribution:
  - 84% Non-Churn  
  - 16% Churn  
- 20+ features including:
  - Demographic attributes  
  - Product relationship data  
  - Credit utilization behavior  
  - Transaction frequency and trend indicators  

The dataset reflects real-world imbalanced classification conditions.

---

## 🔎 Exploratory Data Analysis – Key Insights

Analysis reveals that churn is driven primarily by **behavioral patterns**, not demographic characteristics.

Key findings:

- Declining transaction frequency increases churn risk  
- Higher inactivity ratio strongly correlates with churn  
- Lower product relationship count increases vulnerability  
- Credit utilization patterns affect retention probability  
- Transaction trend decline (Q4–Q1) acts as an early warning signal  

Demographic features show significantly weaker influence compared to behavioral indicators.

---

## 🧠 Feature Engineering Highlights

To enhance predictive performance, several business-driven features were created:

- Transactions per Month (engagement intensity)  
- Inactive Ratio (normalized disengagement indicator)  
- Tenure Segmentation  
- Age Group Segmentation  
- Transaction Trend Change Metrics  

Feature selection methods applied:

- ANOVA F-Test  
- Mutual Information  
- Tree-Based Feature Importance  
- Correlation & Multicollinearity Analysis (VIF)  

Final selected features emphasize behavioral engagement signals.

---

## 🤖 Model Development

The following classification models were evaluated:

- Decision Tree  
- Random Forest  
- Logistic Regression  
- Gaussian Naive Bayes  
- K-Nearest Neighbors  
- MLP Classifier  
- AdaBoost  
- Gradient Boosting  
- Support Vector Machine  
- LightGBM  
- XGBoost  

After evaluation and hyperparameter tuning:

### 🥇 Final Model: XGBoost Classifier

Selected based on:

- Highest Recall on churn class  
- Strong F1-score balance  
- Acceptable generalization gap  
- High ROC AUC  

---

## 📈 Model Performance (Test Set)

| Metric | Score |
|--------|--------|
| Accuracy | 0.94 |
| Precision (Churn) | 0.85 |
| Recall (Churn) | 0.77 |
| F1 Score | 0.81 |
| ROC AUC | 0.96 |

### Confusion Matrix (Test Data)

- True Positive: 251  
- False Positive: 46  
- True Negative: 1655  
- False Negative: 74  

The model successfully detects 77% of churn customers while maintaining strong precision.

---

## 🧬 Model Interpretation

Top Predictive Drivers:

1. Total Revolving Balance  
2. Transactions per Month  
3. Total Relationship Count  
4. Inactive Ratio  
5. Transaction Change (Q4–Q1)  

SHAP analysis confirms:

- Declining activity significantly increases churn probability  
- Behavioral disengagement is a stronger signal than demographics  
- Higher product engagement reduces churn likelihood  

---

## 🖥 Streamlit Deployment Structure

The application is organized into several sections:

1. 📊 Business Case  
2. 📈 EDA & Insights  
3. 🤖 Model Performance  
4. 🔮 Live Churn Prediction Simulator  

Users can input customer data and receive:

- Churn Probability  
- Risk Classification  
- Business Recommendation  

---

## 🏗 Project Structure

