from aiohttp import ClientSession, web
import logging


logger = logging.getLogger(__name__)


url = 'http://0.0.0.0:8000/api/v1/model/'
predict_api = 'predict'


async def async_predict_impl(payload):
    async with aiohttp.ClientSession() as session:
        async with session.post(url + fit_api, json=payload) as response:
            return await response.json()


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
