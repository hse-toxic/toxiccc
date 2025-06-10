from app.app import fit_impl, set_impl, get_status_impl, predict_impl, transform_impl
from app.app import list_models_impl
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from http import HTTPStatus
from typing import List

from .api_models import ModelConfig, ModelListResponse, StatusResponse
from .api_models import FitRequest, FitResponse
from .api_models import SetRequest, SetResponse
from .api_models import PredictRequest, PredictionResponse
from .api_models import TransformRequest, TransformResponse

import sys
sys.path.append('../../')


models = {}

router = APIRouter()


# API endpoints
@router.post("/fit", status_code=HTTPStatus.CREATED,
             response_model=List[FitResponse])
async def fit(request: List[FitRequest]):
    # Реализуйте логику обучения и сохранения модели.
    # Обратите внимание на формат входных данных.
    # Обучать Нужно логистическую и линейную регрессии.
    result = await fit_impl(request[0])
    return JSONResponse(content=[result[1]], status_code=result[0])


@router.post("/set", response_model=List[SetResponse])
async def set_api(request: SetRequest):
    result = await set_impl(request)
    return JSONResponse(content=[result[1]], status_code=result[0])


@router.get("/get_status", response_model=List[StatusResponse])
async def get_status():
    # Получение списка  загруженных для инференса моделей
    result = await get_status_impl()
    return JSONResponse(content=[result[1]], status_code=result[0])


@router.post("/predict", response_model=PredictionResponse)
async def predict(request: List[PredictRequest]):
    # Реализуйте инференс одной из загруженных моделей
    result = await predict_impl(request[0])
    return JSONResponse(content=result[1], status_code=result[0])


@router.get("/models", response_model=List[ModelListResponse])
async def list_models():
    # Реализуйте получения списка обученных моделей
    result = await list_models_impl()
    return JSONResponse(content=[result[1]], status_code=result[0])


@router.post("/transform", response_model=TransformResponse)
async def transform(request: List[TransformRequest]):
    # Реализуйте инференс одной из загруженных моделей
    result = await transform_impl(request[0])
    return JSONResponse(content=result[1], status_code=result[0])
