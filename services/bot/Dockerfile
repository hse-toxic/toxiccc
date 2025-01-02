FROM python:3.11
ENV PIP_ROOT_USER_ACTION=ignore

WORKDIR app

COPY . .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "-u", "main.py"]