IMAGE_NAME = agentic-r7
VENV_NAME = venv

ifeq ($(OS),Windows_NT)
    PYTHON = $(VENV_NAME)/Scripts/python.exe
else
    PYTHON = $(VENV_NAME)/bin/python3
endif

PIP = $(PYTHON) -m pip
PYTEST = $(PYTHON) -m pytest

# Scripts de evaluación
EVAL_RAG_SCRIPT = test/experiments/evaluate.py
EVAL_AGENT_SCRIPT = test/experiments/evaluate_agent.py

.PHONY: all install run test eval eval-agent eval-agent-real clean docker-build docker-run setup

all: install run

$(VENV_NAME):
	python -m venv $(VENV_NAME)

install: $(VENV_NAME)
	$(PIP) install -r requirements.txt

setup: install

run:
	@echo "=== Ejecutando Agente ==="
	$(PYTHON) -m src.main

test:
	@echo "=== Ejecutando Tests ==="
	$(PYTEST) test/ -v

# Eval: solo RAG (Recall@k / MRR)
eval: install
	@echo "=== Ejecutando Evaluación RAG (Recall@k / MRR) ==="
	$(PYTHON) $(EVAL_RAG_SCRIPT)

# Eval: agente completo (router + tools + latencia) usando MockLLM 
eval-agent: install
	@echo "=== Ejecutando Evaluación del Agente (MockLLM) ==="
	$(PYTHON) $(EVAL_AGENT_SCRIPT)

# Eval: agente completo usando el LLM real 
eval-agent-real: install
	@echo "=== Ejecutando Evaluación del Agente (LLM real) ==="
ifeq ($(OS),Windows_NT)
	powershell -NoProfile -ExecutionPolicy Bypass -Command "$$env:USE_REAL_LLM='1'; & '$(PYTHON)' '$(EVAL_AGENT_SCRIPT)'"
else
	USE_REAL_LLM=1 $(PYTHON) $(EVAL_AGENT_SCRIPT)
endif

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

docker-build:
	docker build -t $(IMAGE_NAME) .

docker-run:
	docker run -v $(PWD)/logs:/app/logs $(IMAGE_NAME)
