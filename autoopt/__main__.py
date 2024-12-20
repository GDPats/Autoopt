from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from autoopt.optimizer import run_grid_search, run_randomized_search

# Load a sample dataset (Iris)
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

# Test run for GridSearchCV on Random Forest
best_model, best_params = run_grid_search("random_forest", X_train, y_train)
print("Best model (GridSearchCV):", best_model)
print("Best parameters (GridSearchCV):", best_params)

# Test run for RandomizedSearchCV on SVM
best_model, best_params = run_randomized_search("svm", X_train, y_train, n_iter=5)
print("Best model (RandomizedSearchCV):", best_model)
print("Best parameters (RandomizedSearchCV):", best_params)
