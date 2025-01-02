from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer


class VectorizeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        if type(X) == list:
            dct = {'comment': X}
            X = pd.DataFrame(dct)
        self.vectorizer = CountVectorizer(ngram_range=(1, 2))
        self.vectorizer.fit(X['comment'])
        return self

    def transform(self, X):
        if type(X) == list:
            dct = {'comment': X}
            X = pd.DataFrame(dct)
        X_train = self.vectorizer.transform(X['comment'])
        return X_train
