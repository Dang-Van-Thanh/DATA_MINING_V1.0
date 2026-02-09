import pandas as pd
import numpy as np

class DataCleaner:
    def __init__(self, drop_columns=None):
        self.drop_columns = drop_columns or []

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # drop leakage columns
        df = df.drop(columns=self.drop_columns, errors="ignore")

        # replace unknown with NaN
        df.replace("unknown", np.nan, inplace=True)

        return df
