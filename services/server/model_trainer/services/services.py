import os
import asyncio
import logging
from joblib import dump, load
from fastapi import HTTPException
import time
from concurrent.futures import ProcessPoolExecutor
from dotenv import load_dotenv
import dill
import re
import nltk
import pymorphy3
from nltk import word_tokenize
from nltk.corpus import stopwords
from joblib import load
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression


import sys
sys.path.append('../')
from api.v1.api_models import ModelConfig, FitRequest, FitResponse, ModelListResponse, SetRequest, PredictRequest, PredictionResponse


nltk.download('stopwords')
nltk.download('punkt_tab')
morph = pymorphy3.MorphAnalyzer()
ru_stopwords = stopwords.words("russian") + ['это', 'её']
vectorizer = load('models/vectorizer.joblib')

models_lock = asyncio.Lock()
loaded_models = {}
models = {}
workers = 0

# load models
# for model_id in ['glove_logreg', 'Ngrams_NaiveByas', 'word2vec_logreg']:
for model_id in ['Ngrams_NaiveByas']:
    with open(f"models/{model_id}.pkl", 'rb') as file:
        model = dill.load(file)
        loaded_models[model_id] = {}
        loaded_models[model_id]['model'] = model
        loaded_models[model_id]['type'] = model_id
        models[model_id] = model_id

# load base model
# model = load('models/base.joblib')
# loaded_models['base'] = {}
# loaded_models['base']['model'] = model
# loaded_models['base']['type'] = 'logreg'
# models['base'] = model_id
current_model_id = 'Ngrams_NaiveByas'

load_dotenv()
MODELS_DIR = os.getenv('MODELS_DIR', 'models')
MAX_WORKERS = int(os.getenv('MAX_WORKERS', 4))
MAX_MODELS = int(os.getenv('MAX_MODELS', 8))


logger = logging.getLogger(__name__)
executor = ProcessPoolExecutor(max_workers=MAX_WORKERS)


def transform(text: str):
    def clean_text(text):
        text = text.lower()
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'www\S+', '', text)
        text = re.sub(r"\'s", " ", text)
        text = re.sub('\W', ' ', text)
        text = re.sub('\s+', ' ', text)
        text = text.strip(' ')
        return text

    text = clean_text(text)
    text = " ".join(
        filter(
            lambda word: word not in ru_stopwords,
            map(lambda word: morph.parse(word)[0].normal_form, word_tokenize(text))
        )
    )

    logger.info(text)

    return text


# fix: other models should not requre any preprocessing, implement in pipeline
def predict(model, X):
    logger.info(f'{current_model_id}')
    logger.info(X)
    # X = vectorizer.transform(list(map(lambda comment: transform(comment), X)))
    logger.info(X)
    return model.predict(X)


def fit_model(model, X, y):
    print(f'[{os.getpid()}] fitting')

    # X = vectorizer.transform(list(map(lambda comment: transform(comment), X)))
    model.fit(X, y)

    print(f'[{os.getpid()}] fitted')
    return model


async def fit_impl(request: FitRequest):
    global workers

    print(f'[{os.getpid()}] fit_request')
    model_id = request.config.id # specify new id for user model
    model_type = request.config.ml_model_type # specify model type to for training
    async with models_lock:
        if model_type not in models:
            raise HTTPException(status_code=400, detail=f'Type {model_id} does not exist')

        if model_id in models:
            raise HTTPException(status_code=400, detail=f'Model with id #{model_id} already exists')

        if workers == MAX_WORKERS:
            raise HTTPException(status_code=400, detail=f'Can not fit more than {MAX_WORKERS} models at once')

        try:
            with open(f"models/{model_type}.pkl", 'rb') as file:
                model = pickle.load(file)
                loaded_models[model_id] = {}
                loaded_models[model_id]['model'] = model
                loaded_models[model_id]['type'] = model_type
                models[model_id] = model_id

        except Exception as e:
            raise HTTPException(status_code=400, detail=f'Fit failed with model #{model_id}: {e}')

        workers += 1


    try:
        start = time.time()

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(executor, fit_model, model, request.X, request.y)
        model = result

        result = time.time() - start
        print(f'[{os.getpid()}] {result:.2f}s')
    except Exception as e:
        raise HTTPException(status_code=400, detail=f'Fit failed with model #{model_id}: {e}')

    async with models_lock:
        workers -= 1
        loaded_models[model_id] = {}
        loaded_models[model_id]['model'] = model
        loaded_models[model_id]['type'] = request.config.ml_model_type
        models[model_id] = request.config.ml_model_type

    return 200, {'message': f'Model #{model_id} was trained and saved'}


async def set_impl(request: SetRequest):
    global current_model_id

    model_id = request.id
    async with models_lock:
        if model_id not in loaded_models:
            raise HTTPException(status_code=400, detail=f'Model #{model_id} does not exist')

        current_model_id = model_id
    return 200, {'message': f'Current model is #{model_id}'}


async def get_status_impl():
    async with models_lock:
        models_list = []
        for model_id, model_data in loaded_models.items():
            models_list.append(f'{model_id} (type: {model_data["type"]})')
        if models_list:
            return 200, {'status': f'Loaded models: {", ".join(models_list)}'}
        else:
            return 200, {'status': 'Models are not loaded'}


async def predict_impl(request: PredictRequest):
    async with models_lock:
        try:
            model = loaded_models[current_model_id]['model']
            y_pred = predict(model, request.X)
            return 200, {'predictions': y_pred.tolist()}
        except Exception as e:
            raise HTTPException(status_code=400, detail=f'Predict failed with model #{current_model_id}: {e}')


async def list_models_impl():
    async with models_lock:
        models_list = []
        for model_id, model_type in models.items():
            models_list.append({'id': model_id, 'type': model_type})
        return 200, {'models': models_list}
