import requests
import numpy as np
import time
import asyncio
import aiohttp


url = 'http://localhost:8000/api/v1/models/'
fit_api = 'fit'
remove_by_id_api = 'remove/'
remove_api = 'remove_all'
predict_api = 'predict'
load_api = 'load'
unload_api = 'unload'
get_status_api = 'get_status'
list_models_api = 'list_models'

F = 10
N = 100
K = 5

X = np.random.rand(N, F).tolist()
y = [np.random.randint(K) for i in range(N)]
y[0] = 42
payload1 = [{
    "X": X,
    "y": y,
    "config": {
        "id": "logistic1",
        "ml_model_type": "logistic",
        "hyperparameters": {'max_iter': 1000, 'tol': 1e-6},
    },
}]
payload2 = [{
    "X": X,
    "y": y,
    "config": {
        "id": "logistic2",
        "ml_model_type": "logistic",
        "hyperparameters": {'max_iter': 1000, 'tol': 1e-6},
    },
}]


async def async_fit_impl(payload):
    async with aiohttp.ClientSession() as session:
        async with session.post(url + fit_api, json=payload) as response:
            print(await response.json())


def async_fit():
    async def foo():
        tasks = [
            async_fit_impl(payload1),
            async_fit_impl(payload2),
        ]
        results = await asyncio.gather(*tasks)

        response = requests.delete(url + remove_api)

    asyncio.run(foo())


def default_fit_impl(payload):
    response = requests.post(url + fit_api, json=payload)
    print(response.json())


def default_fit():
    default_fit_impl(payload1)
    default_fit_impl(payload2)

    response = requests.delete(url + remove_api)


def bench(func, k=1):
    s = 0
    for i in range(k):
        start = time.time()
        func()
        end = time.time()
        s += end - start
    return s / k


def async_predict():
    async def foo():
        tasks = [
            async_fit_impl(payload1),
            async_fit_impl(payload2),
        ]
        results = await asyncio.gather(*tasks)

        response = requests.delete(url + remove_api)

    asyncio.run(foo())


def test_fit():
    print('[default_fit]\tbench start')
    result = bench(default_fit, 1)
    print(f'[default_fit]\t{result:.2f}s')

    print('[async_fit]\tbench start')
    result = bench(async_fit, 1)
    print(f'[async_fit]\t{result:.2f}s')


async def async_predict_impl(payload):
    async with aiohttp.ClientSession() as session:
        async with session.post(url + predict_api, json=payload) as response:
            print(await response.json())


def test_predict():
    payload = [{
        "X": [[1, 2], [3, 4]],
        "y": [1, 2],
        "config": {
            "id": "linear1",
            "ml_model_type": "logistic",
            "hyperparameters": {'max_iter': 1000, 'tol': 1e-6},
        },
    }]

    response = requests.post(url + fit_api, json=payload)
    print(response.json())


    async def foo():
        payload1 = [{
            'id': 'linear1',
            'X': [[1, 2]]
        }]
        payload2 = [{
            'id': 'linear1',
            'X': [[3, 4]]
        }]

        tasks = [
            async_predict_impl(payload1),
            async_predict_impl(payload2)
        ]
        results = await asyncio.gather(*tasks)

        for response in results:
            if response is None:
                continue
            print(response.json())

    asyncio.run(foo())

    response = requests.delete(url + remove_api)


def test_base():
    # basic fit
    payload = [{
        "X": [[1, 2], [3, 4]],
        "y": [1, 2],
        "config": {
            "id": "linear1",
            "ml_model_type": "logistic",
            "hyperparameters": {'max_iter': 1000, 'tol': 1e-6},
        },
    }]
    response = requests.post(url + fit_api, json=payload)
    print(response.json())
    assert(response.status_code == 200)

    # fit with same id
    response = requests.post(url + fit_api, json=payload)
    print(response.json())
    assert(response.status_code == 400)

    # predict with non existing id
    payload = [{
        'id': 'linear2',
        'X': [[1, 2]]
    }]
    response = requests.post(url + predict_api, json=payload)
    print(response.json())
    assert(response.status_code == 400)

    # predict with existing
    payload = [{
        'id': 'linear1',
        'X': [[1, 2]]
    }]
    response = requests.post(url + predict_api, json=payload)
    print(response.json())
    assert(response.status_code == 200)

    # load non existing
    payload = {
        'id': 'linear0'
    }
    response = requests.post(url + load_api, json=payload)
    print(response.json())
    assert(response.status_code == 400)

    # unload non existing
    payload = {
        'id': 'linear0'
    }
    response = requests.post(url + unload_api, json=payload)
    print(response.json())
    assert(response.status_code == 400)

    # unload existing
    payload = {
        'id': 'linear1'
    }
    response = requests.post(url + unload_api, json=payload)
    print(response.json())
    assert(response.status_code == 200)

    # unload already unloaded
    payload = {
        'id': 'linear1'
    }
    response = requests.post(url + unload_api, json=payload)
    print(response.json())
    assert(response.status_code == 200)

    # unload already unloaded
    payload = {
        'id': 'linear1'
    }
    response = requests.post(url + unload_api, json=payload)
    print(response.json())
    assert(response.status_code == 200)

    # get status
    response = requests.get(url + get_status_api)
    print(response.json())
    assert(response.status_code == 200)

    # list models
    response = requests.get(url + list_models_api)
    print(response.json())
    assert(response.status_code == 200)

    # load existing
    payload = {
        'id': 'linear1'
    }
    response = requests.post(url + load_api, json=payload)
    print(response.json())
    assert(response.status_code == 200)

    # get status
    response = requests.get(url + get_status_api)
    print(response.json())
    assert(response.status_code == 200)

    # list models
    response = requests.get(url + list_models_api)
    print(response.json())
    assert(response.status_code == 200)

    # predict with reloaded
    payload = [{
        'id': 'linear1',
        'X': [[1, 2]]
    }]
    response = requests.post(url + predict_api, json=payload)
    print(response.json())
    assert(response.status_code == 200)

    # remove all
    response = requests.delete(url + remove_api)
    print(response.json())
    assert(response.status_code == 200)


test_fit()
test_predict()
test_base()
