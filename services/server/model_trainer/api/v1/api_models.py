from typing import List, Union, Dict, Any, Literal
from pydantic import BaseModel


class ModelConfig(BaseModel):
    id: str
    ml_model_type: Literal["glove_logreg", "Ngrams_NaiveByas", "word2vec_logreg"]
    hyperparameters: Dict[str, Any]


class FitRequest(BaseModel):
    X: List[str]
    y: List[float]
    config: ModelConfig


class FitResponse(BaseModel):
    message: str


class ModelListResponse(BaseModel):
    models: List[Dict]


class SetRequest(BaseModel):
    id: str


class SetResponse(BaseModel):
    message: str


class PredictRequest(BaseModel):
    X: List[str]


class PredictionResponse(BaseModel):
    predictions: List[float]


class UnloadRequest(BaseModel):
    id: str


class UnloadResponse(BaseModel):
    message: str


class RemoveResponse(BaseModel):
    message: str


class StatusResponse(BaseModel):
    status: str
