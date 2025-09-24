import pandas as pd


def null_elimination(
    null_values: pd.Series, threshold: int, names: list[str]
) -> list[str]:
    """_summary_

    Args:
        null_values (pd.Series): null pandas series (data.isna().mean())
        threshold (int): our allowed null percent in each feature

    Returns:
        list[str]: features to eliminant
    """
    selected_features = []
    for i in range(len(null_values)):
        if null_values[i] > threshold:
            selected_features.append(names[i])

    return selected_features


def drop_unwanted_features(features: list[str], data: pd.DataFrame) -> pd.DataFrame:
    """_summary_

    Args:
        features (list[str]): features i don't want
        data (pd.DataFrame): the original data

    Returns:
        pd.DataFrame: copy of the original data without the non-wanted features
    """
    return_data = data.copy()
    return_data.drop(labels=features, axis=1, inplace=True)
    return return_data


def replace_values_in_target(
    replace_names: list[str], target: pd.Series, new_value: str
) -> pd.Series:
    """_summary_

    Args:
        replace_names (list[str]): values to replace
        target (pd.Series): the target series
        new_value (str): the new value

    Returns:
        pd.Series: the modified target series
    """
    target_copy = target.copy()
    for name in replace_names:
        target_copy.replace(to_replace=name, value=new_value, inplace=True)
    return target_copy


def drop_based_variance(
    threshold: float, data: pd.DataFrame, numerical_cols: list[str]
) -> list[str]:
    """_summary_

    Args:
        threshold (float): the variance threshold
        data (pd.DataFrame): the data
        numerical_cols (list[str]): the numerical columns

    Returns:
        list[str]: features to drop
    """
    to_drop = []
    for col in numerical_cols:
        if data[col].var() < threshold:
            to_drop.append(col)
    return to_drop
