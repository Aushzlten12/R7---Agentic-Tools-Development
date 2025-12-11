import pytest
from src.agent.core import AgentEngine
from src.tools.calculator import CalculatorTool
from src.tools.verification import VerificationTool
from src.tools.rag import RAGTool


# --- 1. MOCK LLM SERVICE ---
# Simulamos el servicio de IA. No carga Pytorch, solo guarda inputs.
class MockLLMService:
    def __init__(self):
        self.last_query = ""
        self.last_context = ""

    def generate_response(self, query: str, context: str) -> str:
        # Guardamos lo que el Agente nos envió para poder hacer 'assert' luego
        self.last_query = query
        self.last_context = context
        return "RESPUESTA_SIMULADA_POR_MOCK"


# --- 2. FIXTURE DE INTEGRACIÓN ---
@pytest.fixture
def agent_with_mock_llm():
    # A. Inicializamos Tools Reales (Logic-only)
    # Para el RAG, inyectamos datos falsos en memoria para no leer PDFs
    mock_docs = ["[UCSP] El curso de IA requiere 100 créditos."]

    tools = [CalculatorTool(), VerificationTool(), RAGTool(preloaded_docs=mock_docs)]

    # B. Inicializamos el Mock del LLM
    mock_llm = MockLLMService()

    # C. Creamos el Agente conectando el Router Real con el LLM Falso
    agent = AgentEngine(llm_service=mock_llm, tools=tools)

    # Devolvemos tanto el agente como el mock (para inspeccionarlo)
    return agent, mock_llm


# --- 3. TEST CASES DE FLUJO COMPLETO ---


def test_flow_calculator_path(agent_with_mock_llm):
    agent, mock_llm = agent_with_mock_llm

    # ACT: El usuario pide calcular
    response, _ = agent.run("Calcula 50 + 50")

    # ASSERT:
    # 1. El agente debe haber devuelto la respuesta del mock
    assert response == "RESPUESTA_SIMULADA_POR_MOCK"

    # 2. CRITICO: Verificamos que el LLM recibió el resultado de la calculadora
    # El Router debió ejecutar la tool, obtener "100", y pasarlo al contexto
    assert "100" in mock_llm.last_context
    assert "Calculation Result" in mock_llm.last_context


def test_flow_verification_path(agent_with_mock_llm):
    agent, mock_llm = agent_with_mock_llm

    # ACT: El usuario pregunta por un curso
    # (Sabemos que VerificationTool devuelve REJECTED si falta prerequisito)
    response, _ = agent.run("Puedo llevar AI301?")

    # ASSERT:
    # Verificamos que el router detectó el patrón "AI301" y llamó a la tool
    assert "REJECTED" in mock_llm.last_context
    assert "Missing prerequisites" in mock_llm.last_context


def test_flow_rag_fallback_path(agent_with_mock_llm):
    agent, mock_llm = agent_with_mock_llm

    # ACT: Pregunta general
    response, _ = agent.run("Cuantos creditos pide IA?")

    # ASSERT:
    # Verificamos que cayó en el RAG y recuperó el doc inyectado
    assert "[UCSP]" in mock_llm.last_context
    assert "100 créditos" in mock_llm.last_context
