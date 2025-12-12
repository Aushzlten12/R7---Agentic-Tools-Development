import pytest
from src.agent.core import AgentEngine
from src.tools.calculator import CalculatorTool
from src.tools.verification import VerificationTool
from src.tools.rag import RAGTool


# --- 1. MOCK LLM SERVICE ---
class MockLLMService:
    def __init__(self):
        self.last_query = ""
        self.last_context = ""

    def generate_response(self, query: str, context: str) -> str:
        self.last_query = query
        self.last_context = context
        return "RESPUESTA_SIMULADA_POR_MOCK"


# --- 2. FIXTURE DE INTEGRACIÓN ---
@pytest.fixture
def agent_with_mock_llm():
    # Mock docs para no leer PDFs reales en el test
    mock_docs = ["[UCSP] El curso de IA requiere 100 créditos."]

    tools = [CalculatorTool(), VerificationTool(), RAGTool(preloaded_docs=mock_docs)]

    mock_llm = MockLLMService()
    agent = AgentEngine(llm_service=mock_llm, tools=tools)

    return agent, mock_llm


# --- 3. TEST CASES (Actualizados al Español) ---


def test_flow_calculator_path(agent_with_mock_llm):
    agent, mock_llm = agent_with_mock_llm

    # ACT
    response, _ = agent.run("Calcula 50 + 50")

    # ASSERT
    assert response == "RESPUESTA_SIMULADA_POR_MOCK"
    assert "100" in mock_llm.last_context
    # CAMBIO: Ahora buscamos la frase en español que pusimos en core.py
    assert "El resultado es" in mock_llm.last_context


def test_flow_verification_path(agent_with_mock_llm):
    agent, mock_llm = agent_with_mock_llm

    # ACT
    response, _ = agent.run("Puedo llevar AI301?")

    # ASSERT
    assert "REJECTED" in mock_llm.last_context
    assert "Missing prerequisites" in mock_llm.last_context
    # (VerificationTool sigue devolviendo inglés por ahora, así que esto está bien)


def test_flow_rag_fallback_path(agent_with_mock_llm):
    agent, mock_llm = agent_with_mock_llm

    # ACT
    response, _ = agent.run("Cuantos creditos pide IA?")

    # ASSERT
    assert "[UCSP]" in mock_llm.last_context
    assert "100 créditos" in mock_llm.last_context
