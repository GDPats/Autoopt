import json
from sklearn.feature_selection import RFE

def recursive_feature_elimination(model, X_train, y_train, n_features_to_select=5):
    """
    Recursive Feature Elimination (RFE) for feature pruning.

    Parameters:
        model: The model to use for RFE (e.g., LogisticRegression, RandomForestClassifier).
        X_train (array-like): Training features.
        y_train (array-like): Training labels.
        n_features_to_select (int): Number of features to keep.

    Returns:
        JSON string: A JSON object containing the pruned model and dataset information.
    """
    rfe = RFE(estimator=model, n_features_to_select=n_features_to_select)
    rfe.fit(X_train, y_train)
    X_pruned = rfe.transform(X_train)

    return json.dumps({
        "pruned_model": str(model),
        "n_features_selected": X_pruned.shape[1],
        "pruning_method": "Recursive Feature Elimination (RFE)"
    }, indent=4)
