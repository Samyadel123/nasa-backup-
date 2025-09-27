import pandas as pd


def train_model(model, X_train: pd.DataFrame, y_train: pd.Series) -> None:
    """Trains the given model using the provided training data.

    Args:
        model: The machine learning model to be trained.
        X_train (pd.DataFrame): The features for training.
        y_train (pd.Series): The target variable for training.

    Returns:
        None
    """
    model.fit(X_train, y_train)

def predict_exoplanet(model, X: pd.DataFrame) -> pd.Series:
    
    """Predicts the class labels for the given input data using the trained model.

    Args:
        model: The trained machine learning model.
        X (pd.DataFrame): The input features for prediction.

    Returns:
        pd.Series: The predicted class labels.
    """
    return pd.Series(model.predict(X))

def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Evaluates the given model using the provided test data.

    Args:
        model: The machine learning model to be evaluated.
        X_test (pd.DataFrame): The features for testing.
        y_test (pd.Series): The target variable for testing.

    Returns:
        None
    """
    y_pred = model.predict(X_test)
    from sklearn.metrics import classification_report

    report = classification_report(y_test, y_pred, output_dict=True)
    return report


def save_model(model, file_path: str) -> None:
    """Saves the trained model to a file.

    Args:
        model: The machine learning model to be saved.
        file_path (str): The path where the model should be saved.

    Returns:
        None
    """
    import joblib

    joblib.dump(model, file_path)
