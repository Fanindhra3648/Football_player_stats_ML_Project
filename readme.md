# ⚽ End-to-End European Football Player Goal Prediction Pipeline

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![IDE](https://img.shields.io/badge/IDE-VS%20Code-blueviolet.svg)](https://code.visualstudio.com/)

An end-to-end Machine Learning regression project designed to predict individual European football players' seasonal goal outputs (`Goals`) using multi-dimensional performance, passing, and match-exposure statistics.

---

## 📌 Project Overview
Evaluating offensive performance in professional football requires processing highly correlated positional and physical metrics. This project implements a full Machine Learning workflow—from exploratory data analysis and feature engineering to regularized linear modeling, ensemble boosting, hyperparameter tuning, and unseen sample prediction.

* **Target Variable:** `Goals` (Continuous numerical metric representing seasonal goal contributions)
* **Dataset Scope:** 2,689 players across 124 performance metrics from top European leagues (2021–2022 season)
* **Primary Objective:** Build and evaluate multiple ML algorithms to accurately predict goal contributions while addressing multicollinearity and non-linear feature interactions.

---

## 🏗️ Project Architecture & Workflow

```text
├── Data Loading (Latin1 Encoding, Delimiter Handling)
├── Preprocessing & Cleaning (Missing Value Imputation, ID Removal)
├── Categorical Encoding (pd.get_dummies)
├── Feature Scaling (StandardScaler for Linear Regressors)
├── Multi-Model Development (Linear, Ridge, Random Forest, Gradient Boosting)
├── Comparative Performance Analysis (MAE, RMSE, R² Score)
├── Hyperparameter Tuning (GridSearchCV with 3-Fold Cross-Validation)
└── Unseen Data Validation (2022-2023 Season Sample Test)