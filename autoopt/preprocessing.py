from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, normalize, OneHotEncoder
import numpy as np
import pandas as pd

# New module for preprocessing functionalities
def standardize_data(X, method="standard"):
    """
    Standardizes the dataset based on the specified method.

    Parameters:
        X (array-like): Input dataset.
        method (str): Standardization method ("standard", "minmax", "robust").

    Returns:
        X_scaled: Scaled dataset.
    """
    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    elif method == "robust":
        scaler = RobustScaler()
    else:
        raise ValueError(f"Unknown standardization method: {method}")

    return scaler.fit_transform(X)

def normalize_data(X, norm="l2"):
    """
    Normalizes the dataset.

    Parameters:
        X (array-like): Input dataset.
        norm (str): Norm to use ("l1", "l2", "max").

    Returns:
        X_normalized: Normalized dataset.
    """
    return normalize(X, norm=norm)

def encode_data(X, categorical_columns=None):
    """
    One-hot encodes the specified categorical columns in the dataset.

    Parameters:
        X (array-like or DataFrame): Input dataset.
        categorical_columns (list or None): List of column indices or names to encode. If None, encodes all columns.

    Returns:
        X_encoded: Dataset with one-hot encoding applied.
    """
    encoder = OneHotEncoder(sparse_output=False)

    if isinstance(X, pd.DataFrame):
        categorical_data = X[categorical_columns] if categorical_columns else X
        X_encoded = encoder.fit_transform(categorical_data)
        return pd.DataFrame(X_encoded, columns=encoder.get_feature_names_out(categorical_data.columns))
    else:
        X_encoded = encoder.fit_transform(X)
        return X_encoded

def preprocessing_pipeline(X, methods):
    """
    Applies a sequence of preprocessing methods to the dataset.

    Parameters:
        X (array-like): Input dataset.
        methods (list): List of preprocessing steps (e.g., ["standard", "normalize"]).

    Returns:
        X_processed: Preprocessed dataset.
    """
    for method in methods:
        if method in ["standard", "minmax", "robust"]:
            X = standardize_data(X, method=method)
        elif method == "normalize":
            X = normalize_data(X)
        else:
            raise ValueError(f"Unknown preprocessing method: {method}")

    return X
