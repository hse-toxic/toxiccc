import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


class VectorizeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        if isinstance(X, list):
            dct = {'comment': X}
            X = pd.DataFrame(dct)
        self.vectorizer = CountVectorizer(ngram_range=(1, 2))
        self.vectorizer.fit(X['comment'])
        return self

    def transform(self, X):
        if isinstance(X, list):
            dct = {'comment': X}
            X = pd.DataFrame(dct)
        X_train = self.vectorizer.transform(X['comment'])
        return X_train


class TfidfVectorizeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        if isinstance(X, list):
            dct = {'comment': X}
            X = pd.DataFrame(dct)
        self.vectorizer = TfidfVectorizer()
        self.vectorizer.fit(X['comment'])
        return self

    def transform(self, X):
        if isinstance(X, list):
            dct = {'comment': X}
            X = pd.DataFrame(dct)
        X_train = self.vectorizer.transform(X['comment'])
        return X_train
