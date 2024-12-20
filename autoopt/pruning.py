from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.ensemble import RandomForestClassifier

def run_pruning_algorithms(model_name, X_train, y_train, alpha=0.1, n_features_to_select=None):
    """
    Applies pruning algorithms based on the model selected.

    Parameters:
        model_name (str): The name of the model to prune (e.g., "logistic_regression", "lasso", "random_forest").
        X_train (array-like): Training features.
        y_train (array-like): Training labels.
        alpha (float): Regularization strength for L1 pruning (default: 0.1).
        n_features_to_select (int): Number of features to keep for RFE (optional).

    Returns:
        pruned_model: Model after pruning.
        X_pruned: Dataset with pruned features.
    """
    if model_name == "logistic_regression":
        # L1-based pruning using Logistic Regression
        model = LogisticRegression(penalty="l1", solver="liblinear", C=1/alpha)
    elif model_name == "lasso":
        # L1-based pruning using Lasso
        model = Lasso(alpha=alpha)
    elif model_name == "random_forest":
        # Feature importance-based pruning using Random Forest
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        raise ValueError(f"Pruning for model '{model_name}' is not implemented.")

    model.fit(X_train, y_train)

    if model_name in ["logistic_regression", "lasso"]:
        selector = SelectFromModel(model, prefit=True)
        X_pruned = selector.transform(X_train)
    elif model_name == "random_forest":
        # Keep top features based on feature importance
        importances = model.feature_importances_
        top_indices = importances.argsort()[-n_features_to_select:] if n_features_to_select else range(len(importances))
        X_pruned = X_train[:, top_indices]
    else:
        raise ValueError(f"Unsupported pruning method for model '{model_name}'.")

    return model, X_pruned

def recursive_feature_elimination(model, X_train, y_train, n_features_to_select=5):
    """
    Recursive Feature Elimination (RFE) for feature pruning.

    Parameters:
        model: The model to use for RFE (e.g., LogisticRegression, RandomForestClassifier).
        X_train (array-like): Training features.
        y_train (array-like): Training labels.
        n_features_to_select (int): Number of features to keep.

    Returns:
        pruned_model: Model after RFE.
        X_pruned: Dataset with pruned features.
    """
    rfe = RFE(estimator=model, n_features_to_select=n_features_to_select)
    rfe.fit(X_train, y_train)
    X_pruned = rfe.transform(X_train)
    return model, X_pruned
