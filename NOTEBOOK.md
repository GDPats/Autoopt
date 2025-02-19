# AutoOpt - Jupyter Notebook Workflow with Iris Dataset

This README explains the steps followed in the Jupyter Notebook using the **AutoOpt library** and the **Iris dataset**. It also clarifies what each step aims to achieve and the expected results.

## Objective
Demonstrate the functionality of AutoOpt for:
- **Preprocessing** (Scaling, Normalization)
- **Hyperparameter Tuning** (Grid Search, Randomized Search)
- **Feature Selection (Pruning)**

## Dataset Used
The **Iris dataset** is a small, well-known dataset containing:
- **4 numerical features** representing flower measurements.
- **3 classes** representing different species of Iris flowers.

## Workflow Steps Explained

### Step 1: Load the Iris Dataset
- **What happens:**
  - Load the dataset from scikit-learn.
  - Display the first few rows as a DataFrame for easy viewing.
- **Expected result:**
  - A table with 4 feature columns (sepal length, sepal width, petal length, petal width) and a target column representing the flower species (0, 1, or 2).

### Step 2: Split Data into Training & Test Sets
- **What happens:**
  - Split the dataset into **80% training** and **20% test data**.
- **Why:**
  - Separate training and testing data to evaluate model performance on unseen data.
- **Expected result:**
  - Outputs the shapes of the training and test sets (e.g., (120, 4) for training, (30, 4) for test).

### Step 3: Preprocessing Techniques
#### Step 3a: Standardization (Scaling)
- **What happens:**
  - Apply different scalers to adjust the range and distribution of feature values:
    - **StandardScaler:** Mean = 0, Standard Deviation = 1
    - **MinMaxScaler:** Scale features to range [0, 1]
    - **RobustScaler:** Robust to outliers; scales based on median and IQR
- **Why:**
  - Scaling can improve the performance of machine learning models, especially for distance-based algorithms.
- **Expected result:**
  - Transformed feature values printed (first 5 rows) showing the effect of each scaler.

#### Step 3b: Normalization
- **What happens:**
  - Apply **L2 Normalization**: Scale rows so that each has a unit norm.
- **Why:**
  - Useful when we want to compare samples based on their direction rather than magnitude.
- **Expected result:**
  - Transformed feature values printed (first 5 rows) with each row having a unit norm.

**Note:**
- **OneHotEncoding is NOT used** because the Iris dataset is entirely numerical (no categorical data).

### Step 4: Hyperparameter Tuning - Grid Search CV
- **What happens:**
  - Perform **GridSearchCV** on **RandomForestClassifier**:
    - Test different values for hyperparameters like **n_estimators**, **max_depth**, and **min_samples_split**.
- **Why:**
  - To **find the best hyperparameters** that give the best performance.
- **Expected result:**
  - Best Random Forest model and its best hyperparameters printed.

### Step 5: Hyperparameter Tuning - Randomized Search CV
- **What happens:**
  - Perform **RandomizedSearchCV** on **SVM**:
    - Randomly sample hyperparameters like **C**, **kernel**, and **gamma**.
- **Why:**
  - Faster than grid search when the hyperparameter space is large.
- **Expected result:**
  - Best SVM model and its best hyperparameters printed.

### Step 6: Feature Selection (Pruning)
#### Step 6a: L1 Pruning (Logistic Regression with L1 Regularization)
- **What happens:**
  - Use **L1 regularization** to **remove less important features**.
- **Why:**
  - Reduces model complexity and avoids overfitting.
- **Expected result:**
  - Shape of the dataset after pruning (some columns removed).

#### Step 6b: Feature Importance Pruning (Random Forest)
- **What happens:**
  - Use **feature importance from Random Forest** to **select the top 3 features**.
- **Why:**
  - Reduce dimensionality while keeping the most important features.
- **Expected result:**
  - Shape of the dataset after selecting the top 3 features.

#### Step 6c: Recursive Feature Elimination (RFE)
- **What happens:**
  - Iteratively remove the least important features **until only 3 features remain**.
- **Why:**
  - Find the optimal subset of features.
- **Expected result:**
  - Shape of the dataset after keeping the top 3 features.

## Summary
This notebook replicates the functionality provided in AutoOpt's **interactive console menu (`__main__.py`)** but in a **clear step-by-step Jupyter Notebook workflow**.

Each step demonstrates:
- Data preparation
- Model tuning
- Feature selection

The **Iris dataset** is an ideal choice as it is small and well-balanced, making it easy to visualize the impact of each technique.

## How to Run
1. Ensure AutoOpt is installed and the notebook is in the same directory as the AutoOpt package.
2. Open the Jupyter Notebook:
   ```bash
   jupyter notebook new.ipynb
   ```
3. Execute each cell in sequence to observe the process and results.

