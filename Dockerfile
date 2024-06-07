FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY ./mle-intv-main/model.pkl /app/mle-intv-main/model.pkl
COPY app.py /app/app.py

# Establecer la variable de entorno MODEL_PATH
ENV MODEL_PATH=/app/mle-intv-main/model.pkl

CMD ["python", "app.py"]
