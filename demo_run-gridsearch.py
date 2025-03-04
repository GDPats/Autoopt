import json
import numpy as np
from sklearn.datasets import load_iris , load_wine
from sklearn.model_selection import train_test_split
from autoopt.optimizer import run_grid_search
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load dataset
X, y = load_wine(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Using SVM model for demonstration the accuracy before and after GridSearchCV
svm_model = SVC() 
svm_model.fit(X_train, y_train)
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("\n=== SVM Performance (Before GridSearchCV) ===")
print(f"Test Accuracy: {accuracy:.4f}")

print("\n=== Running GridSearchCV for SVM ===")
grid_svm = run_grid_search("svm", X_train, y_train)
grid_svm_dict = json.loads(grid_svm)

# Convert best model string to actual model instance
svm_best_model = eval(grid_svm_dict["best_model"])

# Train and evaluate best-tuned SVM model
svm_best_model.fit(X_train, y_train)  # Train model
y_pred_tuned = svm_best_model.predict(X_test)  # Make predictions
tuned_accuracy = accuracy_score(y_test, y_pred_tuned)  # Compute accuracy

print("\n=== SVM Performance (After GridSearchCV) ===")
print(f"Best Parameters: {grid_svm_dict['best_params']}")
print(f"Test Accuracy: {tuned_accuracy:.4f}")

print("\n=== SVM Model Comparison ===")
print(f"SVM Without GridSearchCV -> Test Accuracy: {accuracy:.4f}")
print(f"SVM With GridSearchCV    -> Test Accuracy: {tuned_accuracy:.4f}")

if tuned_accuracy > accuracy:
    print("\n GridSearchCV Improved SVM Performance! \u2713")
else:
    print("\n GridSearchCV Did Not Improve Performance.")
