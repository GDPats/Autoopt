from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from autoopt.OFR.optimizer import run_grid_search, run_randomized_search
from autoopt.pruning import run_pruning_algorithms, recursive_feature_elimination
from autoopt.preprocessing import standardize_data, normalize_data, encode_data


def main():
    # Load dataset
    data = load_iris()
    X, y = data.data, data.target

    while True:
        print("\nWhat would you like to do?")
        print("1. Run GridSearchCV")
        print("2. Run RandomizedSearchCV")
        print("3. Apply Preprocessing")
        print("4. Apply Pruning")
        print("5. Exit")

        choice = input("Enter your choice (1-5): ")

        if choice == "1":
            # GridSearchCV
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            best_model, best_params = run_grid_search("random_forest", X_train, y_train)
            print("Best Model (GridSearchCV):", best_model)
            print("Best Parameters (GridSearchCV):", best_params)

        elif choice == "2":
            # RandomizedSearchCV
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            best_model, best_params = run_randomized_search("svm", X_train, y_train, n_iter=5)
            print("Best Model (RandomizedSearchCV):", best_model)
            print("Best Parameters (RandomizedSearchCV):", best_params)

        elif choice == "3":
            # Preprocessing
            print("\nSelect preprocessing method:")
            print("1. Standardization (StandardScaler)")
            print("2. Standardization (MinMaxScaler)")
            print("3. Standardization (RobustScaler)")
            print("4. Normalization")
            print("5. Encoding")

            method_choice = input("Enter your choice (1-5): ")

            if method_choice == "1":
                X_processed = standardize_data(X, method="standard")
                print("Data after StandardScaler:", X_processed)

            elif method_choice == "2":
                X_processed = standardize_data(X, method="minmax")
                print("Data after MinMaxScaler:", X_processed)

            elif method_choice == "3":
                X_processed = standardize_data(X, method="robust")
                print("Data after RobustScaler:", X_processed)

            elif method_choice == "4":
                X_processed = normalize_data(X)
                print("Data after Normalization:", X_processed)

            elif method_choice == "5":
                X_processed = encode_data(X)
                print("Data after Encoding:", X_processed)

            else:
                print("Invalid preprocessing choice.")

        elif choice == "4":
            # Pruning
            print("\nSelect pruning method:")
            print("1. L1 Pruning (Logistic Regression)")
            print("2. Feature Importance Pruning (Random Forest)")
            print("3. Recursive Feature Elimination (RFE)")

            pruning_choice = input("Enter your choice (1-3): ")

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            if pruning_choice == "1":
                pruned_model, X_pruned = run_pruning_algorithms("logistic_regression", X_train, y_train, alpha=0.1)
                print("Pruned Dataset (L1):", X_pruned)

            elif pruning_choice == "2":
                pruned_model, X_pruned = run_pruning_algorithms("random_forest", X_train, y_train, n_features_to_select=3)
                print("Pruned Dataset (Feature Importance):", X_pruned)

            elif pruning_choice == "3":
                from sklearn.ensemble import RandomForestClassifier
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                pruned_model, X_pruned = recursive_feature_elimination(model, X_train, y_train, n_features_to_select=3)
                print("Pruned Dataset (RFE):", X_pruned)

            else:
                print("Invalid pruning choice.")

        elif choice == "5":
            print("Exiting...")
            break

        else:
            print("Invalid choice. Please select again.")

if __name__ == "__main__":
    main()
