# AutoOpt - Demo Runs

This document explains the demo scripts in the `demo_run` folder. These scripts test different features of AutoOpt, such as model tuning, feature pruning, and preprocessing.

## 1. `demo_run-randomized-search.py`
- Uses **RandomizedSearchCV** to optimize a **Logistic Regression model**.
- Tests the model before and after optimization using accuracy.
- Shows if RandomizedSearchCV improves the model.

## 2. `demo_run-rfe.py`
- Uses **Recursive Feature Elimination (RFE)** to reduce features for an **SVM model**.
- Tests the model before and after feature selection.
- Shows if RFE improves the model's accuracy.

## 3. `demo_run-pruning.py`
- Uses **feature pruning** to select important features for an **SVM model**.
- Compares accuracy before and after pruning.
- Helps determine if pruning improves performance.

## 4. `demo_run-gridsearch.py`
- Uses **GridSearchCV** to optimize an **SVM model**.
- Trains and tests the model before and after tuning.
- Displays the best parameters and checks if tuning improves accuracy.

### Summary
These demo scripts test different ML techniques:
- **RandomizedSearchCV** and **GridSearchCV** for tuning models.
- **RFE** and **feature pruning** for selecting important features.
- All scripts compare model accuracy before and after applying these techniques.


