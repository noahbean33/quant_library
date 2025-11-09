import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import accuracy_score

def find_best_svm_parameters(X_train, y_train, X_test, y_test, parameter_grid=None):
    """
    Performs a grid search to find the best hyperparameters for an SVC model.

    Args:
        X_train (pd.DataFrame): Training feature data.
        y_train (pd.Series): Training target data.
        X_test (pd.DataFrame): Testing feature data.
        y_test (pd.Series): Testing target data.
        parameter_grid (dict, optional): A dictionary defining the grid of parameters
            to search. If None, a default grid will be used.

    Returns:
        tuple: A tuple containing the best parameter dictionary and the best accuracy score.
    """
    if parameter_grid is None:
        # Default grid if none is provided
        parameter_grid = {
            'gamma': [0.01, 0.001, 0.0001],
            'C': [1, 10, 100, 1000]
        }

    grid = list(ParameterGrid(parameter_grid))

    best_accuracy = 0
    best_parameter = None

    print(f"Starting grid search with {len(grid)} parameter combinations...")

    for p in grid:
        model = SVC(C=p['C'], gamma=p['gamma'])
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_parameter = p
    
    print("Grid search complete.")
    print(f"Best Accuracy: {best_accuracy:.4f}")
    print(f"Best Parameters: {best_parameter}")

    return best_parameter, best_accuracy
