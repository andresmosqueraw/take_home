# Usar la imagen base de Python
FROM python:3.9-slim

# Establecer el directorio de trabajo
WORKDIR /app

# Copiar los archivos necesarios
COPY requirements.txt requirements.txt
COPY app.py app.py

# Argumento para la ubicación del modelo
ARG MODEL_PATH_ARG
ENV MODEL_PATH=/app/model.pkl

# Copiar el modelo al contenedor
COPY ${MODEL_PATH_ARG} /app/model.pkl

# Instalar las dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Exponer el puerto de la API
EXPOSE 8888

# Comando para ejecutar la API
CMD ["python", "app.py"]
