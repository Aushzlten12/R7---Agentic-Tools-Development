# Makefile Universal (Windows/Linux/Mac)

IMAGE_NAME = agentic-r7
VENV_NAME = venv

# --- Detección de Sistema Operativo ---
ifeq ($(OS),Windows_NT)
    # Windows
    PYTHON = $(VENV_NAME)/Scripts/python.exe
else
    # Linux / Mac
    PYTHON = $(VENV_NAME)/bin/python3
endif

# --- Definición de Comandos ---
PIP = $(PYTHON) -m pip
PYTEST = $(PYTHON) -m pytest

.PHONY: all install run test clean docker-build docker-run setup

all: install run

# 1. Crear entorno virtual
$(VENV_NAME):
	python -m venv $(VENV_NAME)

# 2. Instalar dependencias
install: $(VENV_NAME)
	$(PIP) install -r requirements.txt

setup: install

# 3. Ejecutar el Agente
# NOTA: Usamos -m src.main para que Python reconozca los imports correctamente
run:
	@echo "=== Ejecutando Agente ==="
	$(PYTHON) -m src.main

# 4. Ejecutar Tests
test:
	@echo "=== Ejecutando Tests ==="
	$(PYTEST) test/ -v

# 5. Limpieza
clean:
	@echo "=== Limpiando archivos temporales ==="
	rm -rf __pycache__
	rm -rf src/__pycache__
	rm -rf src/agent/__pycache__
	rm -rf src/tools/__pycache__
	rm -rf src/llm/__pycache__
	rm -rf src/utils/__pycache__
	rm -rf test/__pycache__
	rm -rf test/unit/__pycache__
	rm -rf test/integration/__pycache__
	rm -rf logs/

# --- Docker ---
docker-build:
	docker build -t $(IMAGE_NAME) .

docker-run:
	docker run -v $(PWD)/logs:/app/logs $(IMAGE_NAME)