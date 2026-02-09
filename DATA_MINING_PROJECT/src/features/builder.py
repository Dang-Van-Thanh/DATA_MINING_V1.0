import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class FeatureBuilder:
    def __init__(self, num_cols, cat_cols, scaling="standard"):
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.scaling = scaling
        self.pipeline = None

    def build_pipeline(self):
        num_pipe = Pipeline([
            ("scaler", StandardScaler())
        ])

        cat_pipe = Pipeline([
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])

        self.pipeline = ColumnTransformer([
            ("num", num_pipe, self.num_cols),
            ("cat", cat_pipe, self.cat_cols)
        ])

        return self.pipeline

    def fit_transform(self, X):
        return self.pipeline.fit_transform(X)

    def transform(self, X):
        return self.pipeline.transform(X)
