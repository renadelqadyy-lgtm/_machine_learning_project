 Cancer Stage Prediction

 Project Overview

This project presents a **Machine Learning approach to classifying cancer disease stages** using the **Global Cancer Patients 2015–2024** dataset.

The main goal was to investigate whether demographic, environmental, lifestyle, and clinical patient information could be used to accurately predict cancer stage (**Stage 0, I, II, III, or IV**).

**Course:** CBIO313
**Project:** Final Project
**Dataset:** Global Cancer Patients 2015–2024
**Author:** Renad Tawfik
**Student ID:** 231001113
**Date:** August 2026

---

## 🎯 Objectives

The project aimed to:

* Build a machine learning model capable of predicting cancer stage.
* Explore the dataset and identify potentially useful predictive features.
* Compare multiple machine learning algorithms.
* Apply feature engineering and feature selection.
* Perform hyperparameter tuning and cross-validation.
* Deploy the final model as an interactive Streamlit application.

---

## 📊 Dataset

The project uses the **Global Cancer Patients 2015–2024** dataset from Kaggle.

The original dataset contains:

* **50,000 patient records**
* **15 columns**
* Data covering **2015–2024**
* Five cancer-stage classes: **Stage 0, I, II, III, IV**

### Main Feature Categories

**Demographic**

* Age
* Gender
* Country/Region
* Year

**Lifestyle & Environmental**

* Genetic Risk
* Air Pollution
* Alcohol Use
* Smoking
* Obesity Level

**Clinical**

* Cancer Type
* Cancer Stage
* Survival Years
* Target Severity Score

**Treatment**

* Treatment Cost (USD)

Dataset source:
https://www.kaggle.com/datasets/zahidmughal2343/global-cancer-patients-2015-2024

---

## 🧹 Data Preprocessing

A six-step preprocessing pipeline was applied:

1. **Drop Identifier Columns**

   * Removed `Patient_ID`.

2. **Remove Duplicate Rows**

   * No duplicate rows were found.

3. **Handle Missing Values**

   * Numeric missing values were replaced using the median.
   * Categorical missing values were replaced using the mode.

4. **Outlier Removal**

   * Applied the **1.5 × IQR** method to:

     * Age
     * Survival Years
     * Treatment Cost

5. **Encode Target Variable**

   * Stage 0 → 0
   * Stage I → 1
   * Stage II → 2
   * Stage III → 3
   * Stage IV → 4

6. **Encode Categorical Features**

   * Gender
   * Country/Region
   * Cancer Type

After preprocessing, the dataset retained **50,000 rows** with no missing values or duplicate rows.

---

## 🔍 Exploratory Data Analysis

Several visualization techniques were used, including:

* Count plots
* Pie charts
* Histograms
* KDE plots
* Box plots
* Violin plots
* Bar charts
* Correlation heatmaps
* Pair plots

The analysis showed that the five cancer-stage classes were almost perfectly balanced, with each representing approximately **20%** of the dataset.

However, the available demographic, lifestyle, environmental, and clinical features showed very weak relationships with `Cancer_Stage`.

---

## ⚙️ Feature Engineering

Three additional features were created:

### Treatment Efficiency

```text
Treatment_Efficiency =
Survival_Years / (Treatment_Cost_USD + 1)
```

### Is Senior

```text
Is_Senior = 1 if Age >= 60, otherwise 0
```

### Age Group

Age was divided into four groups:

* Young
* Middle
* Senior
* Elderly

---

## 🧠 Feature Selection

Two feature-selection approaches were used:

### 1. Filter Method

**SelectKBest** with the ANOVA F-statistic (`f_classif`) was used to identify the top features.

### 2. Embedded Method

A preliminary **Random Forest** model was used to obtain feature importance scores.

The final feature set combined the features selected by both approaches.

---

## 🤖 Machine Learning Models

Three classification algorithms were trained:

### Logistic Regression

Used as the baseline linear classifier.

### Random Forest

An ensemble of decision trees capable of capturing nonlinear relationships and feature interactions.

### XGBoost

A gradient boosting algorithm used for multiclass classification.

The data was divided using an **80/20 stratified train-test split**:

* Training: **40,000 records**
* Testing: **10,000 records**

---

## 🔧 Hyperparameter Tuning

Random Forest was selected for tuning because it achieved the strongest baseline performance.

`GridSearchCV` with **5-fold cross-validation** was used.

The best configuration was:

```text
n_estimators = 50
max_depth = None
min_samples_split = 2
class_weight = balanced
```

The best mean cross-validated F1 score was approximately **0.2015**.

---

## 📈 Results

The final tuned Random Forest achieved:

| Metric    |  Score |
| --------- | -----: |
| Accuracy  | 0.2022 |
| Precision | 0.2020 |
| Recall    | 0.2022 |
| F1 Score  | 0.2017 |

The 5-fold cross-validation results were approximately:

| Metric    |   Mean |
| --------- | -----: |
| Accuracy  | 0.2006 |
| Precision | 0.2005 |
| Recall    | 0.2006 |
| F1 Score  | 0.2001 |

Because the dataset contains five approximately balanced classes, random guessing would produce about **20% accuracy**.

Therefore, the final model performed very close to random chance.

---

## 💡 Main Finding

The central finding of the project is that **Cancer Stage could not be reliably predicted from the available features in this particular dataset**.

This conclusion was supported by:

* Near-zero correlations between `Cancer_Stage` and the numerical features.
* Strong overlap between stage distributions in the visual analysis.
* Similar performance across Logistic Regression, Random Forest, and XGBoost.
* Hyperparameter tuning producing no meaningful improvement.
* Cross-validation confirming that the approximately 20% performance was stable.

This does **not necessarily indicate a problem with the machine learning pipeline**. Instead, the dataset appears to contain very limited learnable information connecting the available patient attributes to cancer stage.

---

## 🚀 Deployment

The final tuned Random Forest model was deployed as an interactive **Streamlit web application**.

### Live Application

https://cancer-project-machine-learning.streamlit.app

The application allows users to enter patient information and receive a predicted cancer stage.

### Deployment Files

The application uses:

* `model.pkl`
* `scaler.pkl`
* `label_encoder.pkl`
* `feature_names.pkl`

---

## 🎨 Project Presentation

The project presentation is available on Canva:

https://canva.link/5nsgh9y915q96kf

---

## 🔮 Future Work

Possible improvements include:

1. Using `Target_Severity_Score` as a prediction target instead of `Cancer_Stage`.
2. Using a real-world clinical dataset where cancer stage is genuinely related to patient characteristics.
3. Developing additional features based on variables that show stronger relationships with disease severity.
4. Exploring more clinically relevant patient information for future cancer-stage prediction models.

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* XGBoost
* GridSearchCV
* Random Forest
* Logistic Regression
* Streamlit
* Jupyter Notebook
 References
Mughal, Z. (2024). *Global Cancer Patients 2015–2024*. Kaggle.
Pedregosa, F. et al. (2011). *Scikit-learn: Machine Learning in Python*. Journal of Machine Learning Research.
Chen, T. & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*. ACM SIGKDD.
Streamlit Documentation.

---
 Author

**Renad Tawfik**
Nile University — Biotechnology
CBIO313 Final Project
Student ID: 231001113
