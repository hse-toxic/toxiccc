[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/pBuX0a9_)
# Домашняя работа 5-6 (PRO)

В этом задании необходимо создать API для обучения, загрузки,
получения предсказания и управления моделью машинного обучения.

API основан на FastAPI. В качестве модели машинного обучения предлагается использовать простую линейную регрессию из библиотеки scikit-learn.
В качестве хранилища для моделей можно использовать словарь.


# Требования
- Python 3.10+
- Зависимости описаны в requirements.txt
- FastAPI и Uvicorn для создания и запуска API
- Описание требуемого API находится в **openapi_pro.yaml**. Для просмотра можно пользоваться Swagger Editor: https://editor.swagger.io

# Снижение балла
- Разрешено использование дополнительных зависимостей. В этом случае, при отсутствии этих зависимостей в requirements.txt, работа оценивается в 0 баллов
- За любое несоответвие сигнатур эндпоинтов (методов API) описанным в **openapi_pro.yaml** оценка снижается

# Установка зависимостей
```pip install -r requirements.txt```

# Запуск API
```python model_trainer/main.py```

# Серверная часть

Сервер имеет счетчик активных процессов, обучение моделей запускается в отдельном процессе при помощи `asyncio` и `concurrent.futures.ProcessPoolExecutor`.

Конфиг подгружается из `.env` файла при помощи вызова библиотеки `dotenv`.

Сервер корректно обрабатывает запуски обучения без свободных ядер, некорректные запросы и учитывает имеющиеся ограничения по количеству загруженных моделей.

# Клиентская часть

Клиент реалиозован в `client.py`.


### test_fit

Обучение двух моделей в `default_fit` занимает +- 120 секунд (по 60 на каждую), с использованием асинхронных вызовов в `async_fit` работа выполняется в 2 раза быстрее.

Логи клиента:
```
[default_fit]   bench start
[{'message': 'Model #logistic1 was trained and saved'}]
[{'message': 'Model #logistic2 was trained and saved'}]
[default_fit]   121.08s
[async_fit]     bench start
[{'message': 'Model #logistic1 was trained and saved'}]
[{'message': 'Model #logistic2 was trained and saved'}]
[async_fit]     61.09s
```

### test_predict

Логи клиента:
```
[{'message': 'Model #linear1 was trained and saved'}]
[{'predictions': [1.0]}]
[{'predictions': [2.0]}]
```

### test_base

Логи клиента:
```
[{'message': 'Model #linear1 was trained and saved'}]
{'detail': 'Model #linear1 already exists'}
{'detail': 'Model #linear2 does not exist'}
[{'predictions': [1.0]}]
{'detail': 'Model #linear0 does not exist'}
{'detail': 'Model #linear0 does not exist'}
[{'message': 'Model #linear1 was unloaded'}]
[{'message': 'Model #linear1 is already unloaded'}]
[{'message': 'Model #linear1 is already unloaded'}]
[{'status': 'Models are not loaded'}]
[{'models': [{'id': 'linear1', 'type': 'logistic'}]}]
[{'message': 'Model #linear1 was loaded'}]
[{'status': 'Loaded models: linear1 (type: logistic)'}]
[{'models': [{'id': 'linear1', 'type': 'logistic'}]}]
[{'predictions': [1.0]}]
[{'message': 'All models were removed'}]
```
