FROM python:3.11
ENV PIP_ROOT_USER_ACTION=ignore

WORKDIR app
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["streamlit", "run", "app.py", "--server.port", "8081"]