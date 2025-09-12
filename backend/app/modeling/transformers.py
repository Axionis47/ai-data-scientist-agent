from typing import List, Optional, Dict, Set, Any
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class TopKCategorical(BaseEstimator, TransformerMixin):
    def __init__(self, cols: Optional[List[str]] = None, k: int = 50):
        self.cols = cols
        self.k = k
        self.keep_: Dict[str, Set[str]] = {}

    def fit(self, X: Any, y: Any = None):
        df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        cols = self.cols or list(df.columns)
        for c in cols:
            vc = df[c].astype(str).value_counts()
            self.keep_[c] = set(vc.head(self.k).index.tolist())
        return self

    def transform(self, X):
        df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        for c in self.cols or list(df.columns):
            df[c] = df[c].astype(str)
            df[c] = df[c].where(df[c].isin(self.keep_.get(c, set())), other="__OTHER__")
        return df
