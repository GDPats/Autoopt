# AutoOpt Interactive Library

AutoOpt is an interactive Python library that simplifies machine learning workflows by enabling users to perform model tuning, data preprocessing, and feature pruning through a guided console interface. It is designed for both beginners and advanced users to streamline key steps in ML workflows.

---

## **Features**

### 1. **Model Tuning**
Optimize machine learning models using Scikit-learn's GridSearchCV and RandomizedSearchCV:

- **GridSearchCV**:
  - Exhaustively searches through a parameter grid to find the best hyperparameters for a model.
  - Example: Tuning the number of estimators (`n_estimators`) and maximum depth (`max_depth`) of a Random Forest.

- **RandomizedSearchCV**:
  - Searches randomly across the parameter grid, offering faster results for large grids.
  - Example: Tuning the regularization parameter (`C`) and kernel type for an SVM.

When selected, the library:
- Splits the Iris dataset into training and testing sets.
- Applies the chosen search method.
- Displays the best model and its hyperparameters.

---

### 2. **Preprocessing**
Prepare datasets for machine learning models using these preprocessing methods:

- **Standardization**:
  - Scales features to standardize data values.
  - Options include:
    - `StandardScaler`: Scales data to have a mean of 0 and standard deviation of 1.
    - `MinMaxScaler`: Scales data to a specific range (e.g., 0 to 1).
    - `RobustScaler`: Handles outliers by scaling data using the interquartile range.

- **Normalization**:
  - Adjusts feature values so they have a norm of 1 (e.g., L2 normalization).

- **Encoding**:
  - Converts categorical features into numerical representations.
  - Example: One-Hot Encoding creates binary columns for each category.

When selected:
- Users pick a preprocessing method.
- The transformation is applied to the Iris dataset, and the processed dataset is displayed.

---

### 3. **Pruning**
Simplify datasets by reducing feature dimensions using:

- **L1 Pruning (Logistic Regression)**:
  - Uses L1 regularization to eliminate less important features by setting some coefficients to zero.
  - Ideal for sparse datasets.

- **Feature Importance Pruning (Random Forest)**:
  - Keeps the most important features based on Random Forest feature importance scores.

- **Recursive Feature Elimination (RFE)**:
  - Iteratively removes the least important features, retaining only the specified top features.

When selected:
- The chosen pruning method is applied to the Iris dataset.
- The pruned dataset and details of the pruning process are displayed.

---

## **How It Works**

1. **Interactive Console**:
   - The library provides a guided menu with the following options:
     1. Run GridSearchCV.
     2. Run RandomizedSearchCV.
     3. Apply Preprocessing.
     4. Apply Pruning.
     5. Exit the application.

2. **Dataset Handling**:
   - The library uses the Iris dataset, a well-known ML dataset with four features and three target classes.

3. **Dynamic Outputs**:
   - Results are shown for:
     - Best models and hyperparameters (in model tuning).
     - Processed data (in preprocessing).
     - Pruned datasets and details of applied methods (in pruning).

---

## **Why Use AutoOpt?**

- **Beginner-Friendly**: Offers an intuitive, guided interface to learn and perform ML tasks.
- **Hands-On Learning**: Demonstrates key ML concepts like hyperparameter tuning, preprocessing, and feature selection.
- **Extensible**: Easily add more models, preprocessing methods, or pruning techniques to suit advanced needs.

---

## **Getting Started**

### **Installation**

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/autoopt.git
   cd autoopt
   ```

2. Activate a virtual environment (recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Linux/MacOS
   venv\Scripts\activate  # Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### **Running the Application**

1. Launch the interactive interface:
   ```bash
   python3 -m autoopt
   ```

2. Follow the console instructions to:
   - Optimize models.
   - Preprocess datasets.
   - Apply feature pruning techniques.

---

## **Example Output**

```plaintext
What would you like to do?
1. Run GridSearchCV
2. Run RandomizedSearchCV
3. Apply Preprocessing
4. Apply Pruning
5. Exit

Enter your choice (1-5): 1
Best Model (GridSearchCV): RandomForestClassifier(max_depth=10, n_estimators=100)
Best Parameters (GridSearchCV): {'max_depth': 10, 'n_estimators': 100}

...

What would you like to do?
1. Run GridSearchCV
2. Run RandomizedSearchCV
3. Apply Preprocessing
4. Apply Pruning
5. Exit

Enter your choice (1-5): 4
Select pruning method:
1. L1 Pruning (Logistic Regression)
2. Feature Importance Pruning (Random Forest)
3. Recursive Feature Elimination (RFE)

Enter your choice (1-3): 2
Pruned Dataset (Feature Importance): [[5.1, 3.5], [4.9, 3.0], ...]
```

---

## **Next Steps**

- Add support for additional datasets.
- Extend preprocessing methods to include scaling for specific features.
- Enhance the pruning module with advanced techniques like PCA or t-SNE.
- Publish the library on PyPI for easy distribution.

---

