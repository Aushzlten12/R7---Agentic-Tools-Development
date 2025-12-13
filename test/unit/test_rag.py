import pytest
from src.tools.rag import RAGTool, tokenize, normalize_text


class TestRAGTool:
    @pytest.fixture
    def mock_knowledge_base(self):
        return [
            "[UCSP] En la Universidad San Pablo el curso de Algoritmos requiere CS101.",
            "[UNMSM] En la Universidad San Marcos el curso de Algoritmos requiere Matematicas Basicas.",
            "[UNI] La nota minima para aprobar en la UNI es 10.",
            "[UCSP] La nota minima para aprobar en San Pablo es 12.",
            "[GENERAL] La inteligencia artificial es el futuro.",
            "[UNI] Curso: Fisica I (BFI01) | Ubicación: Primer ciclo | Tipo: Obligatorio | Créditos: 5 | Pre-requisito: Ninguno",
        ]

    def test_inicializa_con_inyeccion(self, mock_knowledge_base):
        rag = RAGTool(preloaded_docs=mock_knowledge_base)
        assert len(rag.documents) == len(mock_knowledge_base)
        assert rag.index.ntotal == len(mock_knowledge_base)

    def test_normalizacion_y_tokenizacion(self):
        q = "¿Física I es obligatorio?"
        assert normalize_text(q) == "fisica i es obligatorio"
        toks = tokenize(q)
        assert "fisica" in toks
        assert "i" in toks
        assert "es" not in toks
        assert "obligatorio" not in toks

    def test_desambiguacion_por_universidad_solo_sparse(self, mock_knowledge_base):
        rag = RAGTool(preloaded_docs=mock_knowledge_base)
        res = rag.run("requisito algoritmos san marcos", k=1, alpha=0.0)
        assert "[UNMSM]" in res
        assert "Matematicas Basicas" in res

    def test_busqueda_sparse_con_terminos_presentes(self, mock_knowledge_base):
        rag = RAGTool(preloaded_docs=mock_knowledge_base)
        res = rag.run("nota minima aprobar uni", k=1, alpha=0.0)
        assert "[UNI]" in res
        assert "nota minima" in res.lower()

    def test_match_exacto_por_codigo(self, mock_knowledge_base):
        rag = RAGTool(preloaded_docs=mock_knowledge_base)
        res = rag.run("¿Cuantos creditos tiene BFI01?", k=1, alpha=0.45)
        assert "(BFI01)" in res
        assert ("Créditos: 5" in res) or ("Creditos: 5" in res)

    def test_modo_lista_por_ciclo_no_revienta(self, mock_knowledge_base):
        rag = RAGTool(preloaded_docs=mock_knowledge_base)
        res = rag.run("Que cursos hay en el segundo ciclo", k=3, alpha=0.45)
        assert isinstance(res, str)

    def test_query_rara_no_revienta(self, mock_knowledge_base):
        rag = RAGTool(preloaded_docs=mock_knowledge_base)
        res = rag.run("palabra_inexistente_xyz_123", k=3, alpha=0.45)
        assert isinstance(res, str)
        assert len(res) > 0
