from aiohttp import ClientSession, web
import logging
import os


logger = logging.getLogger(__name__)


addr = os.getenv('SERVER_ADDR')
url = f'http://{addr}/api/v1/models/'
predict_api = 'predict'
set_api = 'set'


async def async_predict_impl(payload):
    async with ClientSession() as session:
        async with session.post(url + predict_api, json=payload) as response:
            return await response.json()


async def async_set_impl(payload):
    async with ClientSession() as session:
        async with session.post(url + set_api, json=payload) as response:
            return response.status, await response.json()


async def predict(text: str):
    payload = [
        {
            "X": [
                text
            ]
        }
    ]
    response = await async_predict_impl(payload)
    if 'predictions' not in response or len(response['predictions']) != 1:
        logger.info(f'prediction failed: {response}')
        return 0.0
    return response['predictions'][0]


async def set_model(model_id: str):
    payload = {
        "id": model_id
    }
    return await async_set_impl(payload)
