import os


class Config:
    # VOLVEMOS A FLAN-T5 (Arquitectura Seq2Seq)
    LLM_MODEL_ID = "google/flan-t5-large"

    # Modelo de Embeddings
    EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

    # Rutas
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.path.join(BASE_DIR, "data")
    LOG_DIR = os.path.join(BASE_DIR, "logs")

    # Parámetros RAG
    # Flan-T5 tiene una ventana pequeña (512 tokens), así que los chunks
    # no pueden ser muy grandes. 500 caracteres está bien.
    CHUNK_SIZE = 500
    K_RETRIEVAL = 2
