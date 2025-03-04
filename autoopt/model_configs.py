from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

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
            "max_iter": [200, 500, 1000]  
        }
    }
}
