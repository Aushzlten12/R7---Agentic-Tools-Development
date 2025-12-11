import os
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader  # <--- NUEVO IMPORT
from src.tools.base import BaseTool
from src.config import Config


class RAGTool(BaseTool):
    def __init__(self, preloaded_docs=None):
        super().__init__(name="rag")
        self.data_dir = Config.DATA_PATH

        print(f"[RAG] Inicializando RAG Híbrido Multi-Universidad...")

        # --- LÓGICA DE INYECCIÓN DE DEPENDENCIAS ---
        if preloaded_docs:
            print("[RAG] MODO TEST: Usando documentos inyectados en memoria.")
            self.documents = preloaded_docs
        else:
            # Modo Producción: Lee los PDFs reales
            self.documents = self._load_pdfs()

        if not self.documents:
            print("[RAG] WARNING: No se encontraron PDFs en la carpeta data/.")
            # Fallback para evitar crash si está vacío
            self.documents = ["Documento vacío por defecto."]

        # 2. Embeddings (Dense)
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = self.embedder.encode(self.documents, convert_to_numpy=True)
        faiss.normalize_L2(embeddings)
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)

        # 3. BM25 (Sparse)
        tokenized_corpus = [doc.lower().split() for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_corpus)

        print(f"[RAG] Indexados {len(self.documents)} fragmentos de planes de estudio.")

    def _load_pdfs(self):
        """
        Lee todos los PDFs del directorio data/, extrae texto e inyecta
        la etiqueta de la universidad para desambiguación.
        """
        chunks = []

        # Mapeo simple: Nombre de archivo -> Etiqueta de Universidad
        # Ajusta esto según los nombres reales de tus archivos
        uni_map = {
            "Plan-estudios": "[UCSP]",  # San Pablo
            "sanMarcos": "[UNMSM]",  # San Marcos
            "2018-N6": "[UNI]",  # UNI (según el código del plan)
            "FDM": "[UPC]",  # Asumiendo UPC/Otra por el nombre
        }

        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir, exist_ok=True)
            return []

        for filename in os.listdir(self.data_dir):
            if filename.endswith(".pdf"):
                file_path = os.path.join(self.data_dir, filename)

                # 1. Detectar etiqueta
                tag = "[GENERAL]"
                for key, val in uni_map.items():
                    if key in filename:
                        tag = val
                        break

                try:
                    # 2. Leer PDF
                    reader = PdfReader(file_path)
                    print(f"Processing {filename} as {tag}...")

                    for page in reader.pages:
                        text = page.extract_text()
                        if not text:
                            continue

                        # 3. Chunking simple por saltos de línea (para E1)
                        # Limpiamos líneas muy cortas o basura
                        lines = [
                            line.strip()
                            for line in text.split("\n")
                            if len(line.strip()) > 20
                        ]

                        # 4. INYECCIÓN DE ETIQUETA (Crucial para BM25)
                        # Transformamos "Curso de Algoritmos" -> "[UCSP] Curso de Algoritmos"
                        tagged_lines = [f"{tag} {line}" for line in lines]

                        chunks.extend(tagged_lines)

                except Exception as e:
                    print(f"Error reading {filename}: {e}")

        return chunks

    def _normalize(self, scores):
        arr = np.array(scores, dtype="float32")
        if arr.max() == arr.min():
            return np.ones_like(arr)
        return (arr - arr.min()) / (arr.max() - arr.min())

    def run(self, query: str, k=3, alpha=0.5) -> str:
        # A. Dense Search
        q_vec = self.embedder.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(q_vec)
        d_scores, d_idx = self.index.search(q_vec, len(self.documents))

        # Aplanar y mapear
        d_scores = d_scores[0]
        dense_map = {idx: score for idx, score in zip(d_idx[0], d_scores)}
        dense_vec = np.zeros(len(self.documents))
        for idx, score in dense_map.items():
            if idx != -1:
                dense_vec[idx] = score

        # B. Sparse Search (BM25)
        # NOTA: BM25 ahora buscará tokens como "[UCSP]" o "[UNMSM]" explícitamente
        tokenized_query = query.lower().split()
        s_scores = self.bm25.get_scores(tokenized_query)

        # C. Fusión
        norm_dense = self._normalize(dense_vec)
        norm_sparse = self._normalize(s_scores)
        hybrid_scores = (alpha * norm_dense) + ((1 - alpha) * norm_sparse)

        top_indices = np.argsort(hybrid_scores)[::-1][:k]

        # Formatear salida
        results = [self.documents[i] for i in top_indices]
        return "\n\n".join(results)
