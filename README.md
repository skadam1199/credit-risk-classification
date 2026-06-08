# Credit Risk Classification Model — Vehicle Loan Default Prediction

## Project Overview
Built a credit risk classification model to predict vehicle loan defaults using the L&T Vehicle Loan Default dataset (233,154 records). This project mirrors real-world auto lending credit risk workflows including data cleaning, feature engineering, model training, validation, and governance reporting.

## Business Problem
Financial institutions lose significant revenue due to loan defaults. This model predicts the probability that a borrower will default on their first EMI payment, enabling lenders to make smarter underwriting decisions.

## Dataset
- **Source:** L&T Vehicle Loan Default Prediction (Kaggle)
- **Records:** 233,154 loan applications
- **Features:** 41 variables including demographic, loan, and bureau data
- **Target:** `loan_default` (1 = defaulted, 0 = no default)
- **Default Rate:** 21.71%

## Technical Approach

### 1. Data Cleaning & Feature Engineering
- Handled missing values in Employment Type
- Calculated borrower Age from Date of Birth and Disbursal Date
- Engineered features: overdue ratio, total current balance
- Encoded categorical variables

### 2. Model Training — H2O AutoML
- Trained 10 models automatically including GBM, XGBoost, DRF, GLM
- Best model: Gradient Boosting Machine (GBM)
- 80/20 train/validation split with stratification

### 3. Model Evaluation
| Metric | Value |
|--------|-------|
| AUC | 0.6569 |
| KS Statistic | 0.2274 |
| KS Interpretation | Acceptable separation |

### 4. Backtesting — 6 Simulated Credit Cycles
Tested model stability across 6 time-ordered data splits:

| Cycle | Period | AUC | Default Rate |
|-------|--------|-----|--------------|
| Cycle 1 | Aug 2018 | 0.6997 | 21.62% |
| Cycle 2 | Aug–Sep 2018 | 0.6987 | 21.18% |
| Cycle 3 | Sep 2018 | 0.6951 | 18.16% |
| Cycle 4 | Sep–Oct 2018 | 0.6872 | 21.15% |
| Cycle 5 | Oct 2018 | 0.6816 | 21.95% |
| Cycle 6 | Oct 2018 | 0.6835 | 26.19% |

**Mean AUC: 0.691 | Std Dev: 0.007 — Model is stable across time periods**

Notable finding: Cycle 6 showed a default rate spike to 26.19%, flagged as potential population shift requiring investigation before production deployment.

## Key Visualizations
- `eda_plots.png` — Exploratory data analysis across key features
- `roc_curve.png` — ROC curve with AUC score
- `confusion_matrix.png` — Prediction accuracy breakdown
- `ks_plot.png` — KS statistic plot showing model separation
- `backtesting.png` — AUC and default rate stability across 6 cycles

## Tech Stack
- Python, H2O AutoML, Scikit-learn
- Pandas, NumPy, Matplotlib, Seaborn
- SciPy (KS statistic)
- Google Colab

## Files
- `Car_loan_default.ipynb` — Full notebook with all code
- `*.png` — Model evaluation visualizations

## Governance Report Summary
Model provides meaningful lift over random baseline. Recommend additional feature engineering and hyperparameter tuning to improve AUC above 0.75 before production deployment. Backtesting confirms temporal stability with low variance across credit cycles.
