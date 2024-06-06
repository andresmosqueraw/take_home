# Usar la imagen base de Python
FROM python:3.9-slim

# Establecer el directorio de trabajo
WORKDIR /app

# Copiar los archivos necesarios
COPY requirements.txt requirements.txt
COPY app.py app.py
COPY mle-intv-main/model.pkl mle-intv-main/model.pkl

# Instalar las dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Exponer el puerto de la API
EXPOSE 5000

# Comando para ejecutar la API
CMD ["python", "app.py"]
