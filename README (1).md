# Predicting Customer Term Deposit Subscription Using Supervised Machine Learning

## Project Overview

This project develops a supervised machine learning workflow to predict whether a bank customer will subscribe to a term deposit.

The analysis uses the **UCI Bank Marketing dataset** and compares multiple classification algorithms under different approaches to handling the strong imbalance in the target variable.

The project focuses on predictive performance, minority-class detection, threshold selection, feature interpretation, and practical implications for targeted bank marketing.

---

## Business Problem

Bank marketing campaigns involve contacting customers who may or may not subscribe to a term deposit. Treating every customer equally can lead to inefficient use of marketing resources.

The objective of this project is to build a classification model that can help identify customers with a higher likelihood of subscribing, allowing marketing teams to prioritise their outreach.

Because only a relatively small proportion of customers subscribe, the problem is treated as an **imbalanced binary classification task**.

---

## Project Objectives

The project aims to:

1. Inspect and understand the dataset structure and data quality.
2. Identify the distribution of the target variable.
3. Prepare numerical and categorical variables for modelling.
4. Compare different supervised classification algorithms.
5. Investigate approaches for handling class imbalance.
6. Evaluate models using precision, recall, F1 score and ROC-AUC in addition to accuracy.
7. Select a suitable final model based on predictive performance and business considerations.
8. Investigate probability thresholds for the selected model.
9. Examine the most important predictive features.
10. Translate the modelling results into practical stakeholder insights.

---

## Dataset

The project uses the **UCI Bank Marketing dataset**.

The modelling data contains:

- **45,211 observations**
- **17 original columns**
- **16 original predictors**
- **1 binary target variable**

The target variable is:

- `no` — customer did not subscribe
- `yes` — customer subscribed

### Modelling Features

The following 15 predictors were used:

| Type | Features |
|---|---|
| Numerical | `age`, `balance`, `day`, `campaign`, `pdays`, `previous` |
| Categorical | `job`, `marital`, `education`, `default`, `housing`, `loan`, `contact`, `month`, `poutcome` |

### Excluded Variable

`duration` was excluded from modelling.

This variable records the duration of the current contact. Since the duration is only known during or after the interaction, including it would make the model less suitable for a realistic pre-contact targeting scenario and could introduce target leakage.

---

## Target Distribution

The target is strongly imbalanced:

| Class | Count | Percentage |
|---|---:|---:|
| `no` | 39,922 | 88.3% |
| `yes` | 5,289 | 11.7% |

This imbalance means that accuracy alone is not sufficient for evaluating the models.

A classifier that simply predicted `no` for every customer would already achieve approximately 88.3% accuracy while failing to identify any potential subscribers.

For this reason, the project gives particular attention to:

- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion matrices

---

## Data Quality

The dataset was inspected for:

- Standard missing values
- Duplicate records
- Categorical placeholder values
- Numerical distributions
- Categorical distributions
- Target imbalance

### Findings

- No standard `NaN` missing values were detected.
- No duplicate rows were found.
- Several categorical variables contain the value `unknown`.
- `pdays = -1` represents customers who had not previously been contacted.

The `unknown` category was retained rather than removing affected records. This avoids unnecessary loss of observations and preserves the information represented by the dataset.

---

## Preprocessing

The preprocessing workflow separates numerical and categorical features.

### Numerical Features

Numerical variables are:

1. Median-imputed
2. Standardised using `StandardScaler`

### Categorical Features

Categorical variables are:

1. Imputed using the most frequent category
2. One-hot encoded using `OneHotEncoder`
3. Configured to ignore previously unseen categories

The preprocessing is implemented through scikit-learn pipelines to ensure that transformations are consistently applied during model training and evaluation.

---

## Train-Test Split

The dataset was divided using an **80/20 stratified train-test split**.

```text
Training set: 36,168 observations
Test set:       9,043 observations
```

Stratification was used to preserve the minority-class proportion in both partitions.

---

## Models

Three supervised classification algorithms were evaluated:

### 1. Logistic Regression

Used as a linear classification baseline and as an interpretable reference model.

### 2. Decision Tree

Used to capture non-linear relationships and provide a tree-based comparison.

### 3. Random Forest

Used as an ensemble tree-based model capable of modelling more complex relationships between customer and campaign variables.

---

## Class Imbalance Experiments

Each algorithm was evaluated under three approaches.

### Baseline

Models were trained using the original class distribution.

### Class Weighting

The `class_weight="balanced"` option was used to give greater importance to the minority class during model training.

### SMOTENC

SMOTENC was used to oversample the minority class while accounting for the dataset's mixture of numerical and categorical variables.

This produced **nine model configurations** in total.

---

## Model Results

The final test-set comparison was:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.893 | 0.663 | 0.179 | 0.281 | 0.772 |
| Decision Tree | 0.831 | 0.298 | 0.329 | 0.313 | 0.613 |
| Random Forest | 0.894 | 0.622 | 0.233 | 0.340 | 0.793 |
| Logistic Regression (Balanced) | 0.755 | 0.266 | 0.624 | 0.373 | 0.772 |
| Decision Tree (Balanced) | 0.840 | 0.308 | 0.295 | 0.301 | 0.604 |
| **Random Forest (Balanced)** | **0.881** | **0.488** | **0.413** | **0.448** | **0.791** |
| Logistic Regression (SMOTENC) | 0.726 | 0.232 | 0.582 | 0.332 | 0.738 |
| Decision Tree (SMOTENC) | 0.776 | 0.255 | 0.474 | 0.331 | 0.645 |
| Random Forest (SMOTENC) | 0.836 | 0.355 | 0.491 | 0.412 | 0.779 |

### Model Selection

The **Random Forest with class weighting** was selected as the strongest overall configuration based on F1 score.

Its default-threshold performance was:

- Accuracy: **0.881**
- Precision: **0.488**
- Recall: **0.413**
- F1: **0.448**
- ROC-AUC: **0.791**

The ordinary Random Forest achieved the highest ROC-AUC at 0.793, but the balanced Random Forest provided the strongest F1 score and substantially better minority-class recall than the ordinary Random Forest.

---

## Threshold Analysis

The selected Random Forest produces probability estimates, allowing the classification threshold to be adjusted.

Several thresholds were tested:

| Threshold | Precision | Recall | F1 |
|---:|---:|---:|---:|
| 0.30 | 0.337 | 0.619 | 0.437 |
| 0.35 | 0.377 | 0.567 | 0.453 |
| **0.40** | **0.419** | **0.528** | **0.468** |
| 0.45 | 0.457 | 0.467 | 0.462 |
| 0.50 | 0.486 | 0.417 | 0.449 |
| 0.55 | 0.504 | 0.357 | 0.418 |
| 0.60 | 0.538 | 0.305 | 0.390 |

Among the tested thresholds, **0.40 produced the highest F1 score**.

At this threshold:

- Precision = **41.9%**
- Recall = **52.8%**
- F1 = **46.8%**

The 0.40 threshold is therefore the preferred operating point for this project when F1 is the selection criterion.

In a production setting, the threshold should instead be validated against business costs, campaign capacity and the relative consequences of false positives and false negatives.

---

## Feature Importance

The balanced Random Forest identified the following as the strongest individual features in the fitted model:

1. `balance`
2. `age`
3. `day`
4. `campaign`
5. `pdays`
6. `poutcome_success`
7. `contact_unknown`
8. `previous`
9. `housing_no`
10. `housing_yes`

These values describe predictive contribution within the fitted Random Forest. They should not be interpreted as evidence that a feature directly causes a customer to subscribe.

---

## Key Findings

### 1. Accuracy can be misleading

The high accuracy of some baseline models does not necessarily indicate strong subscriber detection. The minority class is only 11.7% of the dataset.

### 2. Class weighting improved minority detection

Applying class weighting increased recall for the positive class and produced a better F1 balance for the Random Forest.

### 3. Random Forest provided the strongest overall balance

The balanced Random Forest achieved the highest F1 score among the nine tested configurations.

### 4. Threshold selection matters

Changing the probability threshold from the conventional 0.50 to 0.40 increased recall while maintaining a useful precision level, producing the highest tested F1 score.

### 5. Customer and campaign history variables are important

Variables such as balance, age, campaign count, previous-contact history and campaign outcome contribute strongly to the model's predictions.

---

## Business Implications

The model can be used as a decision-support tool for targeted marketing.

Potential applications include:

- Prioritising customers for campaign outreach.
- Ranking customers by predicted subscription probability.
- Adjusting the number of customers contacted according to campaign capacity.
- Reducing reliance on blanket marketing campaigns.
- Using probability thresholds to control the precision-recall trade-off.

The model should support rather than replace business judgement. Marketing teams should also consider campaign costs, customer experience, contact limits and current business rules.

---

## Limitations

Several limitations should be considered:

1. Evaluation uses a single stratified train-test split.
2. The dataset represents historical marketing behaviour and may not fully represent future customers.
3. Threshold selection in this project was performed on test-set predictions; a production workflow should use a separate validation process.
4. Random Forest feature importance describes predictive contribution rather than causality.
5. Changes in customer behaviour or campaign strategy could reduce future model performance.

---

## Future Improvements

Possible extensions include:

- Cross-validation
- Systematic hyperparameter tuning
- Probability calibration
- Time-based validation
- Cost-sensitive threshold optimisation
- More detailed model interpretability
- Monitoring for changes in customer and campaign behaviour
- Evaluation on newer campaign data

---

## Project Structure

```text
Predicting Customer Term Deposit/
│
├── data/
│   ├── raw/
│   │   ├── bank.csv
│   │   └── bank-full.csv
│   └── processed/
│
├── notebooks/
│   └── bank_marketing_classification_revised_executed.ipynb
│
├── outputs/
│   ├── figures/
│   │   ├── model_f1_comparison.png
│   │   ├── precision_recall_comparison.png
│   │   ├── random_forest_feature_importance.png
│   │   └── roc_curve_comparison.png
│   │
│   ├── models/
│   │   └── random_forest_balanced.joblib
│   │
│   └── tables/
│       ├── model_comparison.csv
│       └── random_forest_threshold_analysis.csv
│
├── report/
│   └── Predicting_Customer_Term_Deposit_Final_Report.pdf
│
├── src/
│   ├── data_inspection.py
│   ├── evaluate_models.py
│   └── preprocessing.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Imbalanced-learn
- Matplotlib
- Seaborn
- Joblib
- Jupyter Notebook
- Git / GitHub

---

## Reproducibility

The main analysis is contained in:

```text
notebooks/bank_marketing_classification_revised_executed.ipynb
```

The notebook contains the data inspection, preprocessing, model training, evaluation, threshold analysis and interpretation used for this project.

The trained Random Forest model is saved under:

```text
outputs/models/random_forest_balanced.joblib
```

The model comparison and threshold-analysis results are stored as CSV files under:

```text
outputs/tables/
```

---

## References

- Moro, S., Cortez, P., & Rita, P. (2014). A data-driven approach to predict the success of bank telemarketing. *Decision Support Systems, 62*, 22–31.
- UCI Machine Learning Repository — Bank Marketing dataset.
- Scikit-learn documentation.
- Imbalanced-learn documentation.

---

## Project Status

**Completed**

The project includes:

- Data inspection
- Exploratory analysis
- Leakage assessment
- Preprocessing
- Three classification algorithms
- Three imbalance-handling strategies
- Nine model configurations
- Model comparison
- Threshold analysis
- Feature importance
- Final model selection
- Business interpretation
- Final report
