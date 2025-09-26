import pandas as pd
import numpy as np
from .util import drop_unwanted_features, drop_based_variance, null_elimination


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """_summary_

    Args:
        data (pd.DataFrame): the original data
        target_name (str): the name of the target column

    Returns:
        tuple[pd.DataFrame, pd.Series]: cleaned data and target series
    """
    # drop unwanted features
    categorical_cols = data.select_dtypes(include="object").columns
    data = drop_unwanted_features(features=categorical_cols, data=data)

    # drop based on null values
    null_values = data.isna().mean() * 100
    names = data.columns
    to_drop_null = null_elimination(null_values=null_values, threshold=30, names=names)
    data = drop_unwanted_features(features=to_drop_null, data=data)
    # separate target

    # drop based on variance
    numerical_cols = data.select_dtypes(include=np.number).columns
    to_drop_variance = drop_based_variance(
        threshold=0.1, data=data, numerical_cols=numerical_cols
    )

    data = drop_unwanted_features(features=to_drop_variance, data=data)
    return data
