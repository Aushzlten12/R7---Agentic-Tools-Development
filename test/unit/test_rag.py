import pytest
from src.tools.rag import RAGTool


class TestRAGTool:

    @pytest.fixture
    def mock_knowledge_base(self):
        """
        Simulamos que _load_pdfs ya hizo su trabajo y devolvió esto.
        Inyectamos las etiquetas [UCSP], [UNMSM] manualmente para probar BM25.
        """
        return [
            "[UCSP] El curso de Algoritmos requiere CS101.",
            "[UNMSM] El curso de Algoritmos requiere Matemáticas Básicas.",
            "[UNI] La nota mínima para aprobar es 10.",
            "[UCSP] La nota mínima para aprobar es 12.",
            "[GENERAL] La inteligencia artificial es el futuro.",
        ]

    def test_initialization_with_injection(self, mock_knowledge_base):
        """Prueba que la inyección funciona y se indexa."""
        rag = RAGTool(preloaded_docs=mock_knowledge_base)
        assert len(rag.documents) == 5
        assert rag.index.ntotal == 5  # FAISS index size

    def test_disambiguation_by_university(self, mock_knowledge_base):
        """
        TEST CRÍTICO PARA R7:
        Si pregunto por 'Algoritmos' en 'San Marcos', ¿BM25 filtra correctamente?
        """
        rag = RAGTool(preloaded_docs=mock_knowledge_base)

        # Query incluye el nombre de la U, lo que activa el BM25 para esa etiqueta
        query = "requisito Algoritmos San Marcos"

        result = rag.run(query, k=1)

        # Debería traer el documento de UNMSM, no el de UCSP
        assert "[UNMSM]" in result
        assert "Matemáticas Básicas" in result

    def test_semantic_search(self, mock_knowledge_base):
        """Prueba que los embeddings funcionan (palabras distintas, mismo sentido)."""
        rag = RAGTool(preloaded_docs=mock_knowledge_base)

        # "calificación" vs "nota"
        query = "calificación aprobatoria en la UNI"

        result = rag.run(query, k=1)

        assert "[UNI]" in result
        assert "nota mínima" in result

    def test_hybrid_behavior(self, mock_knowledge_base):
        """Prueba que el sistema no crashea con queries raras."""
        rag = RAGTool(preloaded_docs=mock_knowledge_base)
        result = rag.run("palabra_inexistente_xyz_123")
        # Debería devolver algo (probablemente por score bajo), pero no romper
        assert isinstance(result, str)
