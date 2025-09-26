from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
import pandas as pd


def process(X_train: pd.DataFrame, X_test: pd.DataFrame):
    scaler = StandardScaler()
    imputer = SimpleImputer(strategy="median")
    transform = make_pipeline(imputer, scaler)
    X_train = pd.DataFrame(transform.fit_transform(X_train), columns=X_train.columns)
    X_test = pd.DataFrame(transform.transform(X_test), columns=X_test.columns)
    return X_train, X_test
