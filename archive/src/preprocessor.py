import pandas as pd
from sklearn.base import TransformerMixin


class Preprocessor(TransformerMixin):
    ids = ["ID", "base_date", "Local Code"]
    
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        cols = X.columns
        raw_cols = cols[cols.str.startswith("feat_") & ~cols.str.startswith("feat_cat_")]
        cat_cols = cols[cols.str.startswith("feat_cat_")]
        X = X.set_index(self.ids)
        raw_X = X[raw_cols]
        cat_X = X[cat_cols].astype("category")

        out_X = pd.concat([raw_X, cat_X], axis=1)
        return out_X
