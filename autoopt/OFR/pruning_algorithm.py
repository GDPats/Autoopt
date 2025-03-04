import json
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier

def run_pruning_algorithms(model_name, X_train, y_train, alpha=0.1, n_features_to_select=None):
    """
    Applies pruning algorithms based on the model selected.

    Parameters:
        model_name (str): The name of the model to prune (e.g., "logistic_regression", "lasso", "random_forest").
        X_train (array-like): Training features.
        y_train (array-like): Training labels.
        alpha (float): Regularization strength for L1 pruning (default: 0.1).
        n_features_to_select (int): Number of features to keep for feature importance pruning (optional).

    Returns:
        JSON string: A JSON object containing the pruned model and dataset information.
    """
    if model_name == "logistic_regression":
        model = LogisticRegression(penalty="l1", solver="liblinear", C=1/alpha)
    elif model_name == "lasso":
        model = Lasso(alpha=alpha)
    elif model_name == "random_forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        return json.dumps({"error": f"Pruning for model '{model_name}' is not implemented."})

    model.fit(X_train, y_train)

    if model_name in ["logistic_regression", "lasso"]:
        selector = SelectFromModel(model, prefit=True)
        X_pruned = selector.transform(X_train)
    elif model_name == "random_forest":
        importances = model.feature_importances_
        top_indices = importances.argsort()[-n_features_to_select:] if n_features_to_select else range(len(importances))
        X_pruned = X_train[:, top_indices]

    return json.dumps({
        "pruned_model": str(model),
        "n_features_selected": X_pruned.shape[1],
        "feature_indices": top_indices if model_name == "random_forest" else "N/A",
        "pruning_method": model_name
    }, indent=4)
