import os
import re
import unicodedata
import numpy as np
import faiss
import pdfplumber
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from src.tools.base import BaseTool
from src.config import Config

STOPWORDS_ES = {
    "que",
    "de",
    "la",
    "el",
    "en",
    "y",
    "a",
    "los",
    "las",
    "un",
    "una",
    "es",
    "del",
    "al",
    "por",
    "para",
    "cual",
    "cuales",
    "cuál",
    "cuáles",
    "qué",
    "hay",
    "son",
    "donde",
    "dónde",
    "pertenece",
    "pertenecen",
    "curso",
    "cursos",
    "uni",
    "obligatorio",
    "electivo",
    "electivos",
    "obligatoria",
    "obligatorios",
}


def normalize_text(s: str) -> str:
    """Lowercase + sin tildes + sin puntuación (mantiene letras/números)."""
    s = (s or "").lower()
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")  # quita tildes
    s = re.sub(r"[^a-z0-9]+", " ", s)  # quita puntuación
    return s.strip()


def tokenize(s: str):
    """Tokens para BM25: normaliza y filtra stopwords."""
    toks = normalize_text(s).split()
    return [t for t in toks if t and t not in STOPWORDS_ES]


class RAGTool(BaseTool):
    def __init__(self, preloaded_docs=None):
        super().__init__(name="rag")
        self.data_dir = Config.DATA_PATH

        print("[RAG] Inicializando RAG Híbrido con Lógica de Ciclos/Electivos...")

        if preloaded_docs:
            self.documents = preloaded_docs
        else:
            self.documents = self._load_pdfs()

        if not self.documents:
            self.documents = ["Error: No se pudieron cargar documentos."]

        # Embeddings
        self.embedder = SentenceTransformer(Config.EMBEDDING_MODEL_ID)
        embeddings = self.embedder.encode(self.documents, convert_to_numpy=True)
        faiss.normalize_L2(embeddings)
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)

        # BM25 (Sparse) con tokenización mejorada
        tokenized_corpus = [tokenize(doc) for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_corpus)

        print(f"[RAG] Indexados {len(self.documents)} fragmentos enriquecidos.")

    def _detect_header_map(self, row_norm):
        """
        Detecta cabecera de columnas y devuelve un dict con índices:
        {"codigo": i, "nombre": j, "creditos": k, "req": m}
        """

        # Variantes comunes
        def find_idx(candidates):
            for cand in candidates:
                if cand in row_norm:
                    return row_norm.index(cand)
            return None

        idx_codigo = find_idx(["codigo", "código"])
        idx_nombre = find_idx(["nombre del curso", "nombre"])
        idx_creditos = find_idx(["creditos", "créditos"])
        idx_req = find_idx(
            [
                "pre requisitos",
                "pre-requisitos",
                "pre requisitos",
                "prerequisitos",
                "pre requisitios",
            ]
        )

        if idx_codigo is None or idx_nombre is None or idx_creditos is None:
            return None

        colmap = {"codigo": idx_codigo, "nombre": idx_nombre, "creditos": idx_creditos}
        if idx_req is not None:
            colmap["req"] = idx_req
        return colmap

    def _clean_req(self, s: str) -> str:
        s = (s or "").strip()
        s = s.replace(".pdf", "").strip()
        if not s:
            return "Ninguno"
        # a veces se cuela "NINGUNO" u otras cosas con saltos
        if normalize_text(s) in {"ninguno", "none", "na", "n a"}:
            return "Ninguno"
        return s

    def _clean_credits(self, s: str) -> str:
        s = (s or "").strip()
        m = re.search(r"\d+", s)
        return m.group(0) if m else "N/A"

    def _looks_like_code(self, s: str) -> bool:
        """
        Heurística: códigos tipo BFI01, CC0A1, CC202, etc.
        """
        s = (s or "").strip()
        return bool(re.fullmatch(r"[A-Z]{2,4}[0-9A-Z]{2,4}", s))

    def _load_pdfs(self):
        chunks = []
        # En este caso para el E1 solo se usará un PDF del plan de estudio de la UNI
        uni_map = {
            "sanMarcos": "[UNMSM San Marcos]",
            "2018-N6": "[UNI Universidad Nacional de Ingenieria]",
            "FDM": "[UPC]",
            "Plan-estudios": "[UCSP]",
        }

        if not os.path.exists(self.data_dir):
            return []

        for filename in os.listdir(self.data_dir):
            if not filename.endswith(".pdf"):
                continue

            file_path = os.path.join(self.data_dir, filename)

            tag = "[GENERAL]"
            for key, val in uni_map.items():
                if key in filename:
                    tag = val
                    break

            try:
                print(f"Procesando {filename} como {tag}...")

                with pdfplumber.open(file_path) as pdf:
                    current_header = "Desconocido"  # state
                    colmap = None  # mapeo de columnas detectado

                    for page in pdf.pages:
                        tables = page.extract_tables() or []
                        if not tables:
                            continue

                        for table in tables:
                            for row in table:
                                clean_row = [
                                    str(c).replace("\n", " ").strip() if c else ""
                                    for c in row
                                ]
                                if not any(clean_row):
                                    continue

                                row_norm = [normalize_text(c) for c in clean_row]
                                first_cell = row_norm[0] if row_norm else ""

                                # DETECCIÓN DE HEADERS DE SECCIÓN (ciclos / electivos)
                                # "Primer ciclo", "Tercer ciclo", etc.
                                if "ciclo" in first_cell and "total" not in first_cell:
                                    # guarda el texto original
                                    current_header = (
                                        clean_row[0] if clean_row[0] else "Ciclo"
                                    )
                                    continue

                                if "electivos de especialidad" in first_cell:
                                    current_header = "ELECTIVOS DE ESPECIALIDAD"
                                    continue

                                if "electivos complementarios" in first_cell:
                                    current_header = "ELECTIVOS COMPLEMENTARIOS"
                                    continue

                                # DETECCIÓN DE CABECERA DE COLUMNAS
                                newmap = self._detect_header_map(row_norm)
                                if newmap:
                                    colmap = newmap
                                    continue

                                if first_cell in {"total", "totales"}:
                                    continue

                                # EXTRACCIÓN PRINCIPAL
                                if "UNI" in tag:

                                    if colmap is None:
                                        # Busca una celda que parezca código
                                        code_idx = None
                                        for i, cell in enumerate(clean_row):
                                            if self._looks_like_code(cell):
                                                code_idx = i
                                                break
                                        if code_idx is None:
                                            continue
                                        # asumimos nombre al lado
                                        nombre_idx = (
                                            code_idx + 1
                                            if code_idx + 1 < len(clean_row)
                                            else None
                                        )
                                        if nombre_idx is None:
                                            continue
                                        codigo = clean_row[code_idx]
                                        nombre = clean_row[nombre_idx]
                                        creditos = "N/A"
                                        requisito = "Ninguno"
                                    else:
                                        codigo = (
                                            clean_row[colmap["codigo"]]
                                            if colmap["codigo"] < len(clean_row)
                                            else ""
                                        )
                                        nombre = (
                                            clean_row[colmap["nombre"]]
                                            if colmap["nombre"] < len(clean_row)
                                            else ""
                                        )
                                        creditos_raw = (
                                            clean_row[colmap["creditos"]]
                                            if colmap["creditos"] < len(clean_row)
                                            else ""
                                        )
                                        creditos = self._clean_credits(creditos_raw)

                                        req_raw = ""
                                        if "req" in colmap and colmap["req"] < len(
                                            clean_row
                                        ):
                                            req_raw = clean_row[colmap["req"]]
                                        requisito = self._clean_req(req_raw)

                                    # Validación básica
                                    if (
                                        not codigo
                                        or "codigo" in normalize_text(codigo)
                                        or "código" in normalize_text(codigo)
                                    ):
                                        continue
                                    if "total" in normalize_text(codigo):
                                        continue
                                    if len(codigo.strip()) < 4:
                                        continue
                                    if not nombre:
                                        continue

                                    # ASIGNACIÓN DE TIPO DE CURSO
                                    tipo_curso = "Desconocido"
                                    ciclo_info = current_header

                                    if "ciclo" in normalize_text(current_header):
                                        tipo_curso = "Obligatorio"
                                    elif "especialidad" in normalize_text(
                                        current_header
                                    ):
                                        tipo_curso = "Electivo de Especialidad"
                                        ciclo_info = "Electivos"
                                    elif "complementarios" in normalize_text(
                                        current_header
                                    ):
                                        tipo_curso = "Electivo Complementario"
                                        ciclo_info = "Electivos"

                                    structured_text = (
                                        f"{tag} Curso: {nombre} ({codigo}) | "
                                        f"Ubicación: {ciclo_info} | "
                                        f"Tipo: {tipo_curso} | "
                                        f"Créditos: {creditos} | "
                                        f"Pre-requisito: {requisito}"
                                    )
                                    chunks.append(structured_text)

                                else:
                                    # Otros documentos
                                    row_text = " | ".join(clean_row)
                                    chunks.append(f"{tag} {row_text}")

            except Exception as e:
                print(f"Error reading {filename}: {e}")

        return chunks

    def _normalize_scores(self, scores):
        arr = np.array(scores, dtype="float32")
        if arr.size == 0:
            return arr
        mx, mn = arr.max(), arr.min()
        if mx == mn:
            return np.ones_like(arr)
        return (arr - mn) / (mx - mn)

    def _extract_code_from_query(self, query: str):
        # Captura códigos tipo CC202, CC0A1, BMA02, etc.
        m = re.search(r"\b([A-Za-z]{2,4}[0-9A-Za-z]{2,4})\b", query or "")
        return m.group(1).upper() if m else None

    def run(self, query: str, k=3, alpha=0.45) -> str:
        """
        alpha = peso del dense. (1-alpha) = peso del BM25.
        """
        query_norm = normalize_text(query)

        # Exact match por código si aparece en la query
        code = self._extract_code_from_query(query)
        if code:
            exact = [d for d in self.documents if f"({code})" in d]
            if exact:
                return "\n\n".join(exact[:k])

        # Atajos tipo “filtro” para listados por ciclo/electivos
        cycle_map = {
            "primer ciclo": "Primer ciclo",
            "segundo ciclo": "Segundo ciclo",
            "tercer ciclo": "Tercer ciclo",
            "cuarto ciclo": "Cuarto ciclo",
            "quinto ciclo": "Quinto ciclo",
            "sexto ciclo": "Sexto ciclo",
            "setimo ciclo": "Sétimo ciclo",
            "septimo ciclo": "Sétimo ciclo",
            "octavo ciclo": "Octavo ciclo",
            "noveno ciclo": "Noveno ciclo",
            "decimo ciclo": "Décimo ciclo",
        }

        wants_list = any(
            x in query_norm
            for x in [
                "que cursos hay",
                "cuales cursos",
                "cuáles cursos",
                "lista",
                "listame",
                "muéstrame",
                "mostrar",
            ]
        )
        for key, pretty in cycle_map.items():
            if key in query_norm and (wants_list or "ciclo" in query_norm):
                hits = [d for d in self.documents if f"Ubicación: {pretty}" in d]
                # Devuelve varios (no solo top-k), ajusta si quieres
                return (
                    "\n".join(hits[:60])
                    if hits
                    else "No encontré cursos para ese ciclo."
                )

        if "electivos de especialidad" in query_norm and wants_list:
            hits = [d for d in self.documents if "Tipo: Electivo de Especialidad" in d]
            return (
                "\n".join(hits[:80])
                if hits
                else "No encontré electivos de especialidad."
            )

        if "electivos complementarios" in query_norm and wants_list:
            hits = [d for d in self.documents if "Tipo: Electivo Complementario" in d]
            return (
                "\n".join(hits[:80])
                if hits
                else "No encontré electivos complementarios."
            )

        # Híbrido: Dense + BM25
        q_vec = self.embedder.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(q_vec)
        d_scores, d_idx = self.index.search(q_vec, len(self.documents))

        dense_vec = np.zeros(len(self.documents), dtype="float32")
        for idx, score in zip(d_idx[0], d_scores[0]):
            if idx != -1:
                dense_vec[idx] = float(score)

        s_scores = self.bm25.get_scores(tokenize(query))

        norm_dense = self._normalize_scores(dense_vec)
        norm_sparse = self._normalize_scores(s_scores)

        hybrid_scores = (alpha * norm_dense) + ((1 - alpha) * norm_sparse)
        top_indices = np.argsort(hybrid_scores)[::-1][:k]

        print(f"\n[RAG DEBUG] Recuperado para: '{query}'")
        results = []
        for i in top_indices:
            doc = self.documents[i]
            print(f" -> {doc[:140]}...")
            results.append(doc)

        return "\n\n".join(results)
