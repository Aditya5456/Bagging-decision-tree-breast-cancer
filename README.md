# bagging-decision-tree-breast-cancer

# Bagging with Decision Trees: Reducing Variance

Ensemble learning project showing how **bagging (bootstrap aggregating)** reduces the variance of decision trees on a binary classification task: diagnosing breast tumours as malignant or benign.

## Problem
A single, fully grown decision tree overfits. It reaches ~100% training accuracy but its test accuracy is lower and unstable across different train/test splits (high variance).

## Approach
1. Explore and visualise the data (violin plots, correlation heatmaps)
2. Train a baseline single decision tree
3. Train a `BaggingClassifier` with 100 decision trees
4. Compare both models using repeated stratified cross-validation (5-fold × 5 repeats)
5. Study the effect of the number of trees
6. Evaluate with confusion matrices, classification reports and ROC-AUC
7. Inspect feature importance averaged over the bagged trees
8. (Optional) Repeat on a correlation-reduced feature set

## Dataset
[Breast Cancer Wisconsin (Diagnostic)](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)
- 569 samples, 30 numeric features (mean, standard error and worst value of 10 cell-nucleus measurements)
- Target: `diagnosis` (M = malignant, B = benign), 357 benign / 212 malignant

The data is not included in this repo. Download `data.csv` from Kaggle and place it in the project folder or in `data/`.

## Results
| Model | Test Accuracy | CV Mean Acc | CV Std Dev |
|---|---|---|---|
| Single Decision Tree | 0.9005847953216374 | 0.930747 | 0.025258|
| Bagging (100 trees) | 0.9590643274853801 | 0.957476	 | 0.022555|

- The single tree overfits: train accuracy ___ vs test accuracy ___.
- Bagging improves test accuracy and **reduces the standard deviation across 25 CV folds**, which confirms lower variance.
- Accuracy plateaus after roughly 50 to 100 trees.

## How bagging works
Each tree is trained on a bootstrap sample (random sample with replacement) of the training set. The trees make partly independent errors, and averaging their votes cancels much of that noise. This lowers variance while leaving bias about the same.

## How to run
```bash
git clone https://github.com/<your-username>/bagging-decision-tree-breast-cancer.git
cd bagging-decision-tree-breast-cancer
pip install -r requirements.txt
# place data.csv in this folder, then:
jupyter notebook bagging_decision_tree_breast_cancer.ipynb
```
Requires scikit-learn 1.2 or newer (uses the `estimator=` argument of `BaggingClassifier`).

## Tech stack
Python, pandas, NumPy, scikit-learn, seaborn, matplotlib
