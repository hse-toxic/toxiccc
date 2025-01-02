import os
import uvicorn
from api.v1.api_route import router
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict
from fastapi.openapi.utils import get_openapi
from dotenv import load_dotenv
import logging


import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd


from services.services import load_models

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


load_models()

# loaded_models = {}
# models = {}
# for model_id in ['Ngrams_NaiveByas']:
#     with open(f"models/{model_id}.joblib", 'rb') as file:
#         model = joblib.load(file)
#         loaded_models[model_id] = {}
#         loaded_models[model_id]['model'] = model
#         loaded_models[model_id]['type'] = model_id
#         models[model_id] = model_id


app = FastAPI(
    title="model_trainer",
    docs_url="/api/openapi",
    openapi_url="/api/openapi.json",
)


logging.basicConfig(level=logging.INFO)


class StatusResponse(BaseModel):
    status: str

    model_config = ConfigDict(
        json_schema_extra={"examples": [{"status": "App healthy"}]}
    )


app.include_router(router, prefix="/api/v1/models")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
