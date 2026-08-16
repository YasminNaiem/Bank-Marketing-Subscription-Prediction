# 🏦 Bank Marketing Subscription Prediction using PCA & SVM

## 📌 Project Overview

This project focuses on predicting whether a client will subscribe to a term deposit as part of a bank marketing campaign.

The project applies a complete machine learning workflow, including:

- Dataset verification
- Exploratory Data Analysis (EDA)
- Data cleaning
- Data preprocessing
- Dimensionality reduction using PCA
- Support Vector Machine (SVM) classification
- Hyperparameter tuning using GridSearchCV
- Cross-validation
- Model evaluation

---

## 🎯 Objective

The main objective is to build a machine learning classification model that predicts whether a customer will subscribe to a term deposit.

### Target Variable

- `yes` → `1` : Customer subscribes
- `no` → `0` : Customer does not subscribe

---

## 📊 Dataset

The project uses the **Bank Marketing Dataset**.

### Dataset Characteristics

- **Records:** 41,188
- **Features:** 20+
- **Target:** `y`

The dataset contains both numerical and categorical customer and campaign-related features.

### Feature Examples

**Numerical Features:**
- Age
- Call Duration
- Campaign
- Previous Contacts
- Employment Variation Rate
- Consumer Price Index
- Consumer Confidence Index
- Euribor 3 Month Rate
- Number of Employees

**Categorical Features:**
- Job
- Marital Status
- Education
- Default
- Housing Loan
- Personal Loan
- Contact Type
- Month
- Day of Week
- Previous Campaign Outcome

---

## 🔎 Exploratory Data Analysis

Several visualizations were performed to understand the data and relationships between features and the target variable.

### EDA Performed

1. Target Variable Distribution
2. Call Duration vs Subscription
3. Previous Campaign Outcome vs Subscription Rate
4. Monthly Subscription Rate
5. Subscription Rate by Job Category
6. Correlation Heatmap

### Key Observations

- The target variable is imbalanced, with most clients not subscribing.
- Longer call durations are associated with higher subscription probability.
- Customers with successful outcomes from previous campaigns have higher subscription rates.
- Subscription rates vary across different months.
- Subscription behavior differs across job categories.
- Several economic indicators show strong correlations, particularly `euribor3m` and `emp.var.rate`.

---

## 🧹 Data Cleaning

### Duplicate Handling

Duplicate rows were identified and removed before modeling.

### Unknown Values

The dataset contains `"unknown"` values in categorical features. These values were handled using the mode calculated from the **training data only** to avoid data leakage.

### Outlier Handling

Numerical outliers were handled using **IQR capping** instead of removing records. This approach preserves the dataset size while reducing the impact of extreme values.

---

## ⚙️ Data Preprocessing

The preprocessing pipeline treats numerical and categorical variables separately.

### Numerical Features

- Median Imputation
- Standard Scaling

### Categorical Features

- Most Frequent Imputation
- One-Hot Encoding (`drop='first'`, `handle_unknown='ignore'`)

A `ColumnTransformer` was used to combine both preprocessing pipelines.

---

## ✂️ Train-Test Split

The dataset was split into:

- **80% Training Data**
- **20% Testing Data**

A stratified split was used to preserve the target class distribution. The train-test split was performed **before preprocessing** to prevent data leakage.

---

## 📉 Principal Component Analysis (PCA)

PCA was applied to reduce the dimensionality of the processed dataset. The number of components was selected using the **95% cumulative explained variance rule**.

### PCA Workflow

1. Fit PCA on the training data
2. Calculate cumulative explained variance
3. Select the number of components required to preserve at least 95% of the variance
4. Transform both training and testing data

This reduces dimensionality while preserving most of the information in the original features.

---

## 🤖 Support Vector Machine (SVM)

A Support Vector Machine classifier with an **RBF kernel** was used for classification.

The base model uses:

- RBF Kernel
- Balanced Class Weights
- Random State = 42

The model was trained using the PCA-transformed features.

### Hyperparameter Tuning

`GridSearchCV` was used to find the best SVM parameters.

**Parameters tested:**

| Parameter | Values |
|---|---|
| `C` | 0.1, 1, 10 |
| `gamma` | 0.001, 0.01, 0.1 |
| `kernel` | RBF |

**Cross-validation:**

- 5-fold cross-validation
- Optimization metric: **F1 Score**

Using F1 as the optimization metric is particularly useful because the target classes are imbalanced.

---

## 📈 Model Evaluation

The final model was evaluated using multiple classification metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC AUC
- Confusion Matrix
- Classification Report

A ROC curve was also generated to visualize the classification performance.

---

## 🛠️ Tools & Method

- **Language/Libraries:** Python, pandas, NumPy, scikit-learn, matplotlib/seaborn
- **Modeling:** PCA for dimensionality reduction + SVM (RBF kernel) for classification
- **Tuning:** GridSearchCV with 5-fold cross-validation, optimized for F1 score

---

## 📁 Repository Structure

```
├── data/                    # Raw and/or processed Bank Marketing dataset
├── notebooks/                # EDA, preprocessing, PCA, and modeling notebooks
├── src/                       # Reusable preprocessing/training scripts (if applicable)
└── README.md                   # This file
```

> Update this structure to match your actual repo layout before pushing.

---

## 📌 Definitions

- **PCA (Principal Component Analysis)** — a dimensionality-reduction technique that projects features onto orthogonal components ranked by explained variance
- **SVM (Support Vector Machine)** — a classifier that finds the optimal boundary (hyperplane) separating classes, here using an RBF kernel for non-linear separation
- **F1 Score** — the harmonic mean of precision and recall, useful for imbalanced classification tasks
- **ROC AUC** — Area under the Receiver Operating Characteristic curve, measuring separability between classes across thresholds
