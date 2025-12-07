# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Instalar dependencias del sistema necesarias para compilar ciertas libs
RUN apt-get update && apt-get install -y \
  build-essential \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código fuente
COPY src/ ./src/
COPY data/ ./data/

# Variables de entorno por defecto
ENV PYTHONPATH=/app

# Comando por defecto: Ejecutar el script principal
CMD ["python", "src/main.py"]