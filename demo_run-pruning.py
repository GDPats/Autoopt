import json
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load dataset
X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling to improve model performance
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Using SVM model before pruning
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train, y_train)
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("\n=== SVM Performance (Before Pruning) ===")
print(f"Test Accuracy: {accuracy:.4f}")

print("\n=== Running Feature Pruning for SVM ===")
selector = SelectFromModel(svm_model, prefit=True, max_features=20)
X_train_pruned = selector.transform(X_train)
X_test_pruned = selector.transform(X_test)

# Train and evaluate pruned SVM model
pruned_svm_model = SVC(kernel='linear', random_state=42)
pruned_svm_model.fit(X_train_pruned, y_train)
y_pred_tuned = pruned_svm_model.predict(X_test_pruned)
tuned_accuracy = accuracy_score(y_test, y_pred_tuned)

print("\n=== SVM Performance (After Pruning) ===")
print(f"Test Accuracy: {tuned_accuracy:.4f}")

print("\n=== SVM Model Comparison ===")
print(f"SVM Without Pruning -> Test Accuracy: {accuracy:.4f}")
print(f"SVM With Pruning    -> Test Accuracy: {tuned_accuracy:.4f}")

if tuned_accuracy > accuracy:
    print("\n Feature Pruning Improved SVM Performance! \u2713")
else:
    print("\n Feature Pruning Did Not Improve Performance.\u2717")
