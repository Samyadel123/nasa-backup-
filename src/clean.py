import pandas as pd
import numpy as np
from .util import drop_unwanted_features, drop_based_variance, null_elimination

first_drop = [
    "kepid",
    "kepoi_name",
    "kepler_name",
    "koi_disposition",
    "koi_tce_delivname",
]


def clean_data(data: pd.DataFrame, target_name: str) -> tuple[pd.DataFrame, pd.Series]:
    """_summary_

    Args:
        data (pd.DataFrame): the original data
        target_name (str): the name of the target column

    Returns:
        tuple[pd.DataFrame, pd.Series]: cleaned data and target series
    """
    # drop unwanted features
    data = drop_unwanted_features(features=first_drop, data=data)

    # drop based on null values
    null_values = data.isna().mean() * 100
    names = data.columns
    to_drop_null = null_elimination(null_values=null_values, threshold=30, names=names)
    data = drop_unwanted_features(features=to_drop_null, data=data)
    # separate target
    target = data[target_name]
    data.drop(labels=[target_name], axis=1, inplace=True)

    # drop based on variance
    numerical_cols = data.select_dtypes(include=np.number).columns
    to_drop_variance = drop_based_variance(
        threshold=0.1, data=data, numerical_cols=numerical_cols
    )
    data = drop_unwanted_features(features=to_drop_variance, data=data)
    return data, target
