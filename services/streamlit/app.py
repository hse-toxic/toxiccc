import requests
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from PIL import Image
import json
import os


addr = os.getenv('SERVER_ADDRESS')
url = f'http://{addr}/api/v1/models/'
fit_api = 'fit'
set_api = 'set'
get_status_api = 'get_status'
predict_api = 'predict'
transform_api = 'transform'
models_api = 'models'

main_model_id = 'tfidf_logreg'

api_list = [predict_api, set_api, get_status_api, models_api]


@st.cache_data
def load_data(file):
    return pd.read_csv(file)


st.title("Анализ токсичности комментариев с использованием Streamlit")
st.subheader("API")

# st.write('set')
response = requests.get(url + models_api)
if response.status_code != 200:
    st.write(f'Ошибка получения статуса доступных моделей: {response.json()}')
else:
    models_list = list(map(lambda x: x['id'], response.json()[0]['models']))
    model_id = st.selectbox('Тип модели', models_list)

    if st.button("установить модель"):
        payload = {
            "id": model_id
        }
        response = requests.post(url + set_api, json=payload)
        if response.status_code != 200:
            st.write(f'Ошибка при установке модели: {response.json()}')
        else:
            st.write(f'{response.json()[0]["message"]}')


# st.write('predict')
text = st.text_input("Текст", "")
if text:
    if st.button("определить токсичность"):
        payload = [
            {
                "X": [
                    text
                ]
            }
        ]
        response = requests.post(url + predict_api, json=payload)
        if response.status_code != 200:
            st.write(f"Error: {repsonse.json()}")
        else:
            flag = response.json()['predictions'][0]
            if flag:
                st.write(f"Сообщение '{text}' токсичное")
            else:
                st.write(f"Сообщение '{text}' нетоксичное")


# st.write('get_models')
if st.button("получить список моделей"):
    response = requests.get(url + models_api)
    if response.status_code != 200:
        st.write(
            f'Ошибка получения статуса доступных моделей: {response.json()}'
        )
    else:
        st.write(f'Result: {response.json()}')


st.subheader("Основная модель")
st.write(
    'В качестве основной модели вы решили выбрать TF-IDF + Logistic Regression'
)
if st.button(f"установить модель {main_model_id}"):
    payload = {
        "id": main_model_id
    }
    response = requests.post(url + set_api, json=payload)
    if response.status_code != 200:
        st.write(f'Ошибка при установке модели: {response.json()}')
text = st.text_input("Ваш текст", "")
if text:
    if st.button("токсичность"):
        payload = [
            {
                "X": [
                    text
                ]
            }
        ]
        response = requests.post(url + predict_api, json=payload)
        if response.status_code != 200:
            st.write(f"Error: {repsonse.json()}")
        else:
            flag = response.json()['predictions'][0]
            if flag:
                st.write(f"Сообщение '{text}' токсичное")
            else:
                st.write(f"Сообщение '{text}' нетоксичное")

image = Image.open('img/info.png')
st.image(image, caption='Model review')

st.subheader("Обучение модели на новых данных")
uploaded_file = st.file_uploader("Выберите CSV-файл", type=["csv"])
if uploaded_file is not None:
    st.text(
        'Прикрепите csv файл с данными, формат - две колонки: comment и toxic'
    )
    data = load_data(uploaded_file)
    data.head(5)

    if 'comment' not in list(data.columns.values):
        st.text('Датасет должен содержать колонку comment')
    elif 'toxic' not in data.columns.values:
        st.text('Датасет должен содержать колонку toxic')
    else:
        model_type = st.selectbox(
            'Выберите модель', ['Ngrams_NaiveByas', 'tfidf_logreg']
        )
        model_id = st.text_input("Укажите название для вашей модели")
        params = st.text_input(
            'Гиперпараметры в python-формате {"<параметр>": значение}', '{}'
        )
        params = json.loads(params)
        params = dict(map(lambda x: (f'model__{x[0]}', x[1]), params.items()))
        if st.button("обучить модель"):
            X = data['comment']
            y = data['toxic']
            payload = [
                {
                    "X": X.tolist(),
                    "y": y.tolist(),
                    "config": {
                        "id": model_id,
                        "ml_model_type": model_type,
                        "hyperparameters": params
                    }
                }
            ]
            # st.write(payload)
            response = requests.post(url + fit_api, json=payload)
            if response.status_code != 200:
                st.write(f'Ошибка при обучении модели: {response.json()}')
            else:
                st.write(f'{response.json()[0]["message"]}')
else:
    st.write("Пожалуйста, загрузите CSV-файл.")

st.subheader("Преобразование токсичного комментария")
toxic_text = st.text_input("Ваш токсичный текст", "")
if toxic_text:
    if st.button("иправить"):
        payload = [
            {
                "X": [
                    toxic_text
                ]
            }
        ]
        response = requests.post(url + transform_api, json=payload)
        if response.status_code != 200:
            st.write(f"Error: {repsonse.json()}")
        else:
            text = response.json()['X'][0]
            st.write(f"После обработки: '{text}'")
