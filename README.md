# 🕵️‍♂️ Insurance Fraud Detection using Machine Learning

A robust **machine learning–based system** designed to detect fraudulent insurance claims with high accuracy.  
This project leverages multiple supervised and ensemble learning algorithms to identify and classify fraudulent patterns from real insurance data.

---

## 🚀 Features

- Comprehensive data preprocessing and feature engineering  
- Implementation of **12 ML models** for fraud classification  
- Hyperparameter tuning with `GridSearchCV` for optimal performance  
- Comparative analysis of accuracy, precision, recall, and F1-score  
- Visualization of results using graphs and heatmaps  

---

## 🧩 Machine Learning Models Used

| Category | Model | Description |
|-----------|--------|-------------|
| **Baseline Models** | **Decision Tree** | A tree-based model splitting data by features to classify fraud/non-fraud claims. |
|  | **K-Nearest Neighbors (KNN)** | Classifies claims based on similarity with nearest neighbors using Euclidean distance. |
|  | **Support Vector Classifier (SVC)** | Finds an optimal hyperplane to separate fraudulent and legitimate claims. |
| **Ensemble Methods** | **Random Forest** | Uses multiple decision trees to reduce overfitting and improve generalization. |
|  | **Extra Trees Classifier** | Similar to Random Forest but introduces extra randomness for efficiency and variance reduction. |
|  | **Voting Classifier** | Combines predictions from multiple models using majority voting for stronger performance. |
| **Boosting Algorithms** | **Gradient Boosting** | Builds models sequentially to minimize residual errors of prior models. |
|  | **Stochastic Gradient Boosting (SGB)** | Adds randomness to training subsets to improve generalization and reduce overfitting. |
|  | **AdaBoost (Adaptive Boosting)** | Adjusts sample weights dynamically to focus on harder-to-classify instances. |
|  | **XGBoost (Extreme Gradient Boosting)** | Advanced gradient boosting framework optimized for speed and accuracy. |
|  | **LightGBM (LGBM)** | Uses histogram-based learning for faster, memory-efficient training. |
|  | **CatBoost** | Handles categorical variables natively using ordered boosting; robust and efficient for mixed data. |

---

## 📊 Model Performance Summary

- **Top Performers:**  
  - **Voting Classifier**, **XGBoost**, **LightGBM**, **Extra Trees**, and **CatBoost** achieved the highest accuracy (~0.75).  
- **Moderate Performance:**  
  - **Gradient Boosting** and **Random Forest** (~0.65 accuracy).  
- **Lower Performance:**  
  - **AdaBoost** and **Decision Tree** (~0.6 accuracy).  
  - **KNN** and **SVC** slightly below 0.6.  
  - **Stochastic Gradient Boosting (SGB)** performed lowest (<0.5).  
- **Best Overall Model:**  
  - After hyperparameter tuning, **Decision Tree** delivered the best trade-off between accuracy and interpretability for this dataset.

---

## 📈 Evaluation Metrics

The models were evaluated using:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**
- **Cross-Validation**

These metrics provided a balanced assessment of model performance on an imbalanced dataset.

---

## 🧠 Workflow Overview

1. **Data Collection** – Insurance claims dataset sourced from Kaggle.  
2. **Preprocessing** – Handling missing values, removing duplicates, encoding categorical features, and outlier detection.  
3. **Data Splitting** – 75% training and 25% testing to prevent overfitting.  
4. **Model Training** – 12 models trained and tuned using GridSearchCV.  
5. **Evaluation** – Comparison of all models using key metrics and visualizations.  


