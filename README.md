# Project R7: Agentic Tools Workflow

Este repositorio contiene la implementación del Entregable 1 (E1) para el proyecto de investigación Agentic Tools. El sistema consiste en un agente conversacional basado en `flan-t5-small` capaz de orquestar herramientas externas mediante un flujo de control explícito.

## Arquitectura del Sistema

Para esta entrega (E1), el enrutamiento es determinista para establecer un baseline de rendimiento.

- **Core Engine**: Lógica de enrutamiento basada en reglas (Explicit Router) que decide qué herramienta invocar según el análisis léxico del prompt.
- **LLM Backend**: `google/flan-t5-small` encargado de la síntesis de respuestas finales.
- **Tools**:
  - `CalculatorTool`: Motor aritmético seguro para cálculos precisos.
  - `RAGTool`: Sistema de recuperación aumentada (Retrieval-Augmented Generation) utilizando ChromaDB y embeddings `all-MiniLM-L6-v2`.
  - `VerificationTool`: Mock de API para consulta de requisitos académicos.
- **Observability**: Logging estructurado (JSONL) para auditoría de trazas (Thought -> Action -> Observation).

## Instalación y Ejecución

Usando el archivo `Makefile` incluido para gestionar el ciclo de vida de la aplicación.

1. Entorno Local

Crear entorno virtual e instalar dependencias:

```Bash
make install
```

Ejecutar el agente en modo interactivo (CLI):

```Bash
make run
```

2. Ejecución con Docker

Para garantizar que el entorno es idéntico al de desarrollo, construir la imagen:

```Bash
make docker-build
```

Ejecutar el contenedor:

```Bash
make docker-run
```

## Uso de Agentes

(Pendiente aún)
