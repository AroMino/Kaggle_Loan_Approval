# Kaggle Loan Approval Prediction

## Overview
This repository contains an exploratory data analysis and modeling workflow for the [Kaggle Playground Series - Season 4, Episode 10](https://www.kaggle.com/competitions/playground-series-s4e10) focusing on Loan Approval Prediction.

The primary objective is to predict whether a client will default on a loan at the time of application, supporting credit risk decision-making. Reducing financial risk and improving loan approval decisions through machine learning models can significantly increase portfolio profitability.

## Problem Formulation
* **Task Type**: Supervised Binary Classification
* **Target Variable**: `loan_status` 
  * `0` → No default (loan repaid)
  * `1` → Default

The goal is to accurately identify defaulters while maintaining a strong ROC-AUC score and balanced F1-score for the default class.

## Repository Contents
- **`exploration.ipynb`**: Comprehensive Jupyter Notebook detailing the data wrangling, exploratory data analysis (EDA), preprocessing pipelines, and model evaluation steps.
- **`train.csv`**: The training dataset consisting of 58,645 borrower records.
- **`test.csv`**: The test dataset containing 39,098 borrower records used for evaluating predictions.
- **`sample_submission.csv`**: Template file illustrating the expected format for Kaggle submissions.
- **`playground-series-s4e10.zip`**: Compressed archive of the raw competition data.

## Features
The dataset contains a variety of applicant and loan-specific attributes, measuring credit risk factors such as:
- **Personal Info**: `person_age`, `person_income`, `person_home_ownership`, `person_emp_length`
- **Loan Details**: `loan_amnt`, `loan_intent`, `loan_grade`, `loan_int_rate`, `loan_percent_income`
- **Credit History**: `cb_person_cred_hist_length`, `cb_person_default_on_file`

## Modeling Approach
1. **Data Collection & Splits**: The training data is split into 80% training and 20% validation sets.
2. **Preprocessing**: Handling missing values, one-hot encoding categorical features, and standard scaling numerical variables using `ColumnTransformer` and pipeline tools.
3. **Model Training**: Investigating models capable of handling class imbalances and improving predictive performance compared to rule-based baselines.
