import os


class Config:
    # Modelos ligeros para CPU
    LLM_MODEL_ID = "google/flan-t5-small"
    EMBEDDING_MODEL_ID = "all-MiniLM-L6-v2"

    # Rutas
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.path.join(BASE_DIR, "data")
    LOG_DIR = os.path.join(BASE_DIR, "logs")

    # Parámetros RAG
    CHUNK_SIZE = 500
    K_RETRIEVAL = 2
