import os
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from src.tools.base import BaseTool
from src.config import Config


class RAGTool(BaseTool):
    def __init__(self):
        super().__init__(name="rag")
        self.data_path = Config.DATA_PATH

        print(f"[RAG] Inicializando RAG Híbrido (Dense + Sparse)...")
        # Cargar y procesar documentos
        self.documents = self._load_data()

        if not self.documents:
            print("[RAG] WARNING: No documents found.")
            return

        # Inicializar Componente Denso
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = self.embedder.encode(self.documents, convert_to_numpy=True)

        # Normalizar embeddings para usar Inner Product como similitud coseno
        faiss.normalize_L2(embeddings)
        self.dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings)

        # Inicializar Componente Sparse (BM25)
        # Tokenización simple: minúsculas y split por espacios
        tokenized_corpus = [doc.lower().split() for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_corpus)

        print(f"[RAG] Indexados {len(self.documents)} chunks.")

    def _load_data(self):
        """Carga datos desde el archivo de conocimiento."""
        if not os.path.exists(self.data_path):
            return []

        with open(self.data_path, "r", encoding="utf-8") as f:
            text = f.read()

        # Simulación de Chunking
        # Aquí dividimos por párrafos vacíos o saltos de línea para el baseline
        chunks = [line.strip() for line in text.split("\n") if line.strip()]
        return chunks

    def _normalize(self, scores):
        """Normaliza los scores entre 0 y 1 para poder fusionarlos."""
        arr = np.array(scores, dtype="float32")
        if arr.max() == arr.min():
            return np.ones_like(arr)
        return (arr - arr.min()) / (arr.max() - arr.min())

    def run(self, query: str, k=3, alpha=0.5) -> str:
        """
        Ejecuta búsqueda híbrida.
        alpha: Peso para búsqueda densa (0.0 a 1.0). 0.5 = balanceado.
        """
        if not self.documents:
            return "No knowledge base loaded."

        # Búsqueda Densa (FAISS)
        q_embedding = self.embedder.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(q_embedding)
        dense_scores, dense_indices = self.index.search(
            q_embedding, len(self.documents)
        )

        # Aplanar resultados de FAISS (vienen como matriz)
        dense_scores = dense_scores[0]
        dense_map = {idx: score for idx, score in zip(dense_indices[0], dense_scores)}

        # Búsqueda Sparse (BM25)
        tokenized_query = query.lower().split()
        sparse_scores = self.bm25.get_scores(tokenized_query)

        # Fusión de Scores (Hybrid Fusion)
        # Reordenamos dense_scores para que coincidan con el orden natural de los documentos (0..N)

        dense_scores_aligned = np.zeros(len(self.documents))
        for idx, score in dense_map.items():
            if idx != -1:  # FAISS a veces devuelve -1
                dense_scores_aligned[idx] = score

        # Normalización
        norm_dense = self._normalize(dense_scores_aligned)
        norm_sparse = self._normalize(sparse_scores)

        # Score final
        hybrid_scores = (alpha * norm_dense) + ((1 - alpha) * norm_sparse)

        # Obtener los top K índices
        top_indices = np.argsort(hybrid_scores)[::-1][:k]

        # Recuperar texto
        results = [f"[Doc {i}] {self.documents[i]}" for i in top_indices]

        return "\n\n".join(results)
