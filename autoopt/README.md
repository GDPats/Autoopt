# AutoOpt

AutoOpt is a Python library designed to simplify the process of model fine-tuning and data preprocessing for machine learning practitioners. With a focus on automation and ease of use, AutoOpt provides a simple interface for users to apply advanced techniques without requiring deep expertise in machine learning or programming.

## Features
- **Automated Model Tuning**: Provides out-of-the-box support for `GridSearchCV` and `RandomizedSearchCV` from Scikit-learn.
- **Predefined Parameter Grids**: Includes default parameter grids for common baseline models such as Random Forest, SVM, and Logistic Regression.
- **Customizable Parameters**: Allows users to modify parameter grids or add new models.
- **Flexible Execution**: Run fine-tuning with minimal setup or customize for specific needs.

---

## Getting Started
### Prerequisites
Ensure you have the following installed:
- Python >= 3.6
- Scikit-learn >= 0.24.0

### Installation
Clone the repository and install the required dependencies.

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/autoopt.git
   cd autoopt
   ```

2. Activate your virtual environment (optional but recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Linux/MacOS
   venv\Scripts\activate  # Windows
   ```

3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Project Structure
Here is an overview of the project structure:

```
autoopt_project/          # Main project directory
├── autoopt/              # Library folder
│   ├── __init__.py       # Initializes the package
│   ├── optimizer.py      # Core functions for model fine-tuning
│   ├── model_configs.py  # Contains default parameter grids
│   └── __main__.py       # Entry point for testing the library
├── setup.py              # Packaging file for PyPI
└── README.md             # Project documentation (this file)
```

---

## Usage

### 1. Import the Library
To use AutoOpt, you can import it into your Python scripts:

```python
from autoopt.optimizer import run_grid_search, run_randomized_search
```

### 2. Run GridSearchCV
Here’s an example of running `GridSearchCV` with a Random Forest model:

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from autoopt.optimizer import run_grid_search

# Load dataset
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Run GridSearchCV
best_model, best_params = run_grid_search("random_forest", X_train, y_train)

print("Best Model:", best_model)
print("Best Parameters:", best_params)
```

### 3. Run RandomizedSearchCV
Here’s an example of running `RandomizedSearchCV` with an SVM model:

```python
from autoopt.optimizer import run_randomized_search

# Run RandomizedSearchCV
best_model, best_params = run_randomized_search("svm", X_train, y_train, n_iter=10)

print("Best Model:", best_model)
print("Best Parameters:", best_params)
```

---

## Adding Custom Models
To add new models or update parameter grids, edit `model_configs.py`:

```python
MODEL_PARAM_GRID = {
    "random_forest": {
        "model": RandomForestClassifier,
        "params": {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5, 10],
        }
    },
    "svm": {
        "model": SVC,
        "params": {
            "C": [0.1, 1, 10],
            "kernel": ["linear", "rbf"],
            "gamma": ["scale", "auto"],
        }
    },
    "logistic_regression": {
        "model": LogisticRegression,
        "params": {
            "penalty": ["l2"],
            "C": [0.1, 1, 10],
            "solver": ["liblinear", "lbfgs"],
        }
    }
}
```

---

## Testing
To test the library, navigate to the project root directory and run:

```bash
python3 -m autoopt
```

This will execute the test script (`__main__.py`) with predefined examples to verify the functionality.

---

## Next Steps
- Add more baseline models.
- Improve parameter grids for existing models.
- Package and publish the library on PyPI.

---

## Contributing
Contributions are welcome! If you’d like to contribute, please fork the repository and submit a pull request.

---

## License
This project is licensed under the MIT License. See the LICENSE file for details.

