import pytest
import os
from src.tools.rag import RAGTool
from src.config import Config


# Mock simple
@pytest.fixture
def setup_knowledge_base():
    # Crear datos
    dummy_path = "test_knowledge.txt"
    Config.DATA_PATH = dummy_path
    with open(dummy_path, "w", encoding="utf-8") as f:
        f.write("La Universidad tiene tres campus principales.\n")
        f.write("El curso de Inteligencia Artificial cubre Redes Neuronales.\n")
        f.write("La nota mínima aprobatoria es 11.\n")
        f.write("El examen parcial vale 30% del promedio.\n")

    yield dummy_path

    if os.path.exists(dummy_path):
        os.remove(dummy_path)


def test_rag_initialization(setup_knowledge_base):
    rag = RAGTool()
    assert len(rag.documents) == 4
    assert rag.bm25 is not None
    assert rag.index is not None


def test_rag_hybrid_search_exact_match(setup_knowledge_base):
    rag = RAGTool()
    # Una query que coincide casi exacto con BM25
    query = "nota mínima aprobatoria"
    result = rag.run(query, k=1)
    assert "nota mínima aprobatoria es 11" in result


def test_rag_hybrid_search_semantic(setup_knowledge_base):
    rag = RAGTool()
    # Una query semántica (palabras diferentes, mismo significado)
    # "calificación para pasar" vs "nota aprobatoria"
    query = "calificación necesaria para pasar el curso"
    result = rag.run(query, k=1)
    # Debería recuperar la de la nota 11 gracias a los embeddings
    assert "nota mínima" in result
