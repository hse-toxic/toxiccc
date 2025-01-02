FROM python:3.10.12
ENV PIP_ROOT_USER_ACTION=ignore

WORKDIR app
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY model_trainer ./model_trainer
COPY models ./models

CMD ["python", "-u", "model_trainer/main.py"]