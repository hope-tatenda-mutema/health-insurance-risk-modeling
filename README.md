# Health Insurance Risk Modeling

This project analyzes a medical insurance dataset to understand key cost drivers and build predictive models for annual insurance charges using Python, pandas, and scikit‑learn.

## Project overview

The dataset contains 2,772 records with variables such as age, gender, BMI, number of children, smoking status, region, and annual insurance charges in USD. The goal is to perform end‑to‑end data cleaning, exploratory data analysis (EDA), and regression modeling to support data‑driven insurance pricing.

## Objectives

- Run EDA to identify which attributes most affect insurance charges.  
- Build single‑variable and multivariable linear regression models to predict charges.  
- Apply ridge regression and polynomial features to refine model performance.

## Data preparation

Key data preparation steps:

- Loaded the medical insurance dataset and added descriptive column headers (age, gender, BMI, number of children, smoke, region, charges).  
- Replaced missing values: continuous variables (such as age) with the mean, categorical variables (such as smoking status) with the mode.  
- Converted data types appropriately, created an integer smoker indicator, and rounded charges to two decimal places.

These steps produced a consistent, analysis‑ready dataset with no missing values in the features used for modeling.

## Exploratory data analysis

The EDA focused on understanding how customer characteristics relate to charges.

- Visualizations:  
  - A regression plot of BMI vs charges shows a positive relationship, with higher BMI generally associated with higher costs.  
  - A boxplot of charges by smoking status shows smokers have substantially higher charges than non‑smokers.

- Correlation analysis (selected correlations):  
  - Smoking status and charges: approximately 0.79 (very strong positive correlation).  
  - Age and charges: approximately 0.30 (moderate positive correlation).  
  - BMI and charges: approximately 0.20 (weak‑to‑moderate positive correlation).

These results highlight smoking as the most influential single factor on insurance charges, followed by age and BMI.

## Modeling

The project develops several regression models to predict annual charges.

### 1. Single‑variable linear regression

- Feature: smoker indicator only.  
- Model: `LinearRegression`.  
- Performance: \(R^2 \approx 0.62\), meaning smoking alone explains about 62% of the variance in charges.

### 2. Multivariable linear regression

- Features: age, BMI, number of children, smoke.  
- Model: `LinearRegression`.  
- Performance: \(R^2 \approx 0.75\), showing improved fit by combining multiple risk factors.

### 3. Polynomial regression pipeline

- Pipeline: `StandardScaler` → `PolynomialFeatures(include_bias=False)` → `LinearRegression`.  
- Features: same core predictors as the multivariable model.  
- Performance: in‑sample \(R^2 \approx 0.84\), capturing non‑linear relationships between features and charges.

### 4. Ridge regression with and without polynomial features

- Train/test split: 80% train, 20% test.  
- Model: `Ridge(alpha=0.1)`.

Results:

| Model                                    | Features                     | Test R²  |
|------------------------------------------|------------------------------|----------|
| Ridge regression                          | Original features            | ≈ 0.67   |
| Ridge regression + degree‑2 polynomials   | Polynomial‑transformed input | ≈ 0.78   |

Using ridge regression with polynomial features provides a stronger, more generalizable model than basic linear regression on the same feature set.

## Key insights

- Smoking status is the dominant cost driver and should be a primary factor in pricing and risk segmentation.  
- Age and BMI also contribute meaningfully to charges and improve prediction when combined with smoking status.  
- Regularized polynomial models (ridge with degree‑2 features) deliver better test performance, suggesting that non‑linear effects are important but require regularization to avoid overfitting.

From a business perspective, these models can support more accurate premium setting, identify high‑risk groups, and justify targeted wellness or smoking‑cessation programs.

## Technologies used

- Python, Jupyter Notebook  
- pandas, NumPy  
- seaborn, Matplotlib  
- scikit‑learn: `LinearRegression`, `Ridge`, `train_test_split`, `Pipeline`, `StandardScaler`, `PolynomialFeatures`, `r2_score`

