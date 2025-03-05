from sklearn.model_selection import RandomizedSearchCV
from autoopt.model_configs import MODEL_PARAM_GRID
import json
def run_randomized_search(model_name, X_train, y_train, param_grid=None, n_iter=10):
    """
    Runs RandomizedSearchCV on a specified model.

    Parameters:
        model_name (str): The name of the model to tune (e.g., "random_forest").
        X_train (array-like): Training features.
        y_train (array-like): Training labels.
        param_grid (dict): Optional. Parameter grid for RandomizedSearchCV.
        n_iter (int): Number of parameter settings sampled. Defaults to 10.

    Returns:
        best_model: Best model after tuning.
        best_params: Parameters for the best model.
    """
    model_info = MODEL_PARAM_GRID.get(model_name)
    if not model_info:
        raise ValueError(f"Model '{model_name}' not found in MODEL_PARAM_GRID.")

    model = model_info["model"]()
    random_params = param_grid or model_info["params"]

    random_search = RandomizedSearchCV(model, random_params, n_iter=n_iter, cv=5, n_jobs=-1, verbose=1)
    random_search.fit(X_train, y_train)
    return json.dumps({
        "best_model": str(random_search.best_estimator_),
        "best_params": random_search.best_params_
    }, indent=4)