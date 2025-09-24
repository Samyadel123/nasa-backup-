from util import (
    drop_based_variance,
    drop_unwanted_features,
    null_elimination,
    replace_values_in_target,
)
import pandas as pd
from abc import ABC, abstractmethod


class Preprocessor(ABC):
    def __init__(self, data: pd.DataFrame, target: pd.Series):
        self.data = data
        self.target = target

    @abstractmethod
    def fit(
        self, data: pd.DataFrame, target: pd.Series
    ) -> tuple[pd.DataFrame, pd.Series]:
        """_summary_
            this method will preprocess the data before passing it to the model
            1. pass the data after splitting to train and test
            2. drop unwanted features and keep a list of dropped features
            3.
        Returns:
            tuple[pd.DataFrame, pd.Series]: data and target after preprocessing
        """
        pass

    @abstractmethod
    def fit_transform(
        self,
        data: pd.DataFrame,
        target: pd.Series,
    ) -> tuple[pd.DataFrame, pd.Series]:
        pass

    @abstractmethod
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        pass
