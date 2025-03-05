import json
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from autoopt.optimizer import run_randomized_search
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load dataset
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling to improve convergence
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Using Logistic Regression model for demonstration of accuracy before and after RandomizedSearchCV
logreg_model = LogisticRegression(max_iter=5000)
logreg_model.fit(X_train, y_train)
y_pred = logreg_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("\n=== Logistic Regression Performance (Before RandomizedSearchCV) ===")
print(f"Test Accuracy: {accuracy:.4f}")

print("\n=== Running RandomizedSearchCV for Logistic Regression ===")
random_logreg = run_randomized_search("logistic_regression", X_train, y_train, n_iter=18)
random_logreg_dict = json.loads(random_logreg)

# Convert best model string to actual model instance
logreg_best_model = eval(random_logreg_dict["best_model"])

# Train and evaluate best-tuned Logistic Regression model
logreg_best_model.fit(X_train, y_train)  # Train model
y_pred_tuned = logreg_best_model.predict(X_test)  # Make predictions
tuned_accuracy = accuracy_score(y_test, y_pred_tuned)  # Compute accuracy

print("\n=== Logistic Regression Performance (After RandomizedSearchCV) ===")
print(f"Best Parameters: {random_logreg_dict['best_params']}")
print(f"Test Accuracy: {tuned_accuracy:.4f}")

print("\n=== Logistic Regression Model Comparison ===")
print(f"Logistic Regression Without RandomizedSearchCV -> Test Accuracy: {accuracy:.4f}")
print(f"Logistic Regression With RandomizedSearchCV    -> Test Accuracy: {tuned_accuracy:.4f}")

if tuned_accuracy > accuracy:
    print("\n RandomizedSearchCV Improved Logistic Regression Performance! \u2713")
else:
    print("\n RandomizedSearchCV Did Not Improve Performance.")
