import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


class RankerWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y, eval_set=None, eval_metric=None, **kwargs):
        import pandas as pd
        dt = X.index.get_level_values("base_date")
        X, y, g = self._to_rank_fmt(
            X, y,
            #dt.year.astype("str") + "_" + ((dt.month - 1) // 6).astype("str")
            dt.year.astype("str") + "_" + dt.quarter.astype("str")
            )

        eval_group = None
        if eval_set is not None:
            eval_X, eval_y, eval_group = self._to_rank_fmt(
                eval_set[0][0], eval_set[0][1], 1)
            eval_set = [(eval_X, eval_y)]
        self.estimator.fit(
            X, y, group=g, eval_set=eval_set, eval_group=[eval_group],
            eval_metric=eval_metric,
            **kwargs
            )
        return self

    def predict(self, X):
        #print(X.columns)
        return self.estimator.predict(X)

    def _to_rank_fmt(self, X, y, g):
        #X["g"] = X["feat_quarter"] + X["Result_FinancialStatement FiscalYear"]
        X = X.copy()
        X["g"] = g
        X["y"] = y
        X = X.sort_values(["g", "y"], ascending=True)
        gb = X.groupby("g")
        X["y"] = gb.cumcount()
        y = X.y.values
        X = X.drop(["y", "g"], axis=1)
        g = gb.size()
        #print(y, y.max())
        return X, y, g


class MultitaskRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, estimator):
        self.estimator = estimator
    
    def fit(self, X, y=None, eval_set=None, **kwargs):
        from sklearn.base import clone

        if eval_set is not None:
            eval_set_high = [(ds[0], ds[1][:, 0]) for ds in eval_set]
            eval_set_low = [(ds[0], ds[1][:, 1]) for ds in eval_set]
        else:
            eval_set_high = None
            eval_set_low = None
        self.estimator_high_ = clone(self.estimator).fit(X, y[:, 0], eval_set=eval_set_high, **kwargs)
        self.estimator_low_ = clone(self.estimator).fit(X, y[:, 1], eval_set=eval_set_low, **kwargs)
        return self
    
    def predict(self, X):
        p_high = self.estimator_high_.predict(X)
        p_low = self.estimator_low_.predict(X)
        
        return np.vstack([p_high, p_low]).T