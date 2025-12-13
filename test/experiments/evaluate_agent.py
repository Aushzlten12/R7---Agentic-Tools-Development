import os
import sys
import time
import json
import math

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from src.config import Config
from src.agent.core import AgentEngine
from src.tools.rag import RAGTool
from src.tools.calculator import CalculatorTool
from src.tools.verification import VerificationTool

# Se podría usar el LLM real o un MockLLM
USE_REAL_LLM = os.environ.get("USE_REAL_LLM", "0") == "1"


class MockLLMService:
    """
    LLM rápido para evaluación
    Esto mantiene el flujo del agente
    sin el costo de cargar flan-t5-large, el modelo completo.
    """

    def generate_response(self, query: str, context: str) -> str:
        context = (context or "").strip()
        if not context:
            return "No context provided."
        # 2 lineas del contexto
        lines = [l.strip() for l in context.splitlines() if l.strip()]
        return "\n".join(lines[:2])


def read_last_log_entry():
    """
    Lee la última entrada del archivo logs/execution.jsonl
    """
    log_path = os.path.join(Config.LOG_DIR, "execution.jsonl")
    if not os.path.exists(log_path):
        return None

    with open(log_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        if not lines:
            return None
        return json.loads(lines[-1])


def tool_called_from_entry(entry):
    """
    Extrae el tool usado del steps_trace. El agente usa un maximo de una herramienta por query
    """
    if not entry:
        return None, None

    steps = entry.get("steps_trace", [])
    if not steps:
        return None, None

    tool_name = steps[0].get("tool")
    tool_output = steps[0].get("output", "")
    return tool_name, tool_output


def contains_case_insensitive(haystack: str, needle: str) -> bool:
    return (needle or "").lower() in (haystack or "").lower()


def safe_float(s: str):
    try:
        return float(str(s).strip())
    except Exception:
        return None


def approx_equal(a: float, b: float, tol: float = 1e-6) -> bool:
    return (
        a is not None
        and b is not None
        and math.isfinite(a)
        and math.isfinite(b)
        and abs(a - b) <= tol
    )


def evaluate_cases(agent: AgentEngine, cases, section_name: str):
    """
    cases: lista con:
      - query
      - expected_tool
      - check_type: "contains" | "float"
      - expected: string o float
    """
    routing_hits = 0
    output_hits = 0
    latencies = []

    print(f"\n=== Evaluación: {section_name} ===")
    print("-" * 95)
    print(
        f"{'Query':<45} | {'Tool':<12} | {'RouteOK':<7} | {'OutputOK':<8} | {'t(s)':<6}"
    )
    print("-" * 95)

    for c in cases:
        query = c["query"]
        expected_tool = c["expected_tool"]
        check_type = c["check_type"]
        expected = c["expected"]

        t0 = time.time()
        _response, latency = agent.run(query)
        # el agente loguea steps_trace en execution.jsonl
        dt = time.time() - t0
        latencies.append(dt)

        entry = read_last_log_entry()
        tool_used, tool_output = tool_called_from_entry(entry)

        route_ok = tool_used == expected_tool
        if route_ok:
            routing_hits += 1

        # Validación del output, solo si el router funciono bien
        output_ok = False
        if route_ok:
            if check_type == "contains":
                output_ok = contains_case_insensitive(tool_output, str(expected))
            elif check_type == "float":
                got = safe_float(tool_output)
                output_ok = approx_equal(got, float(expected), tol=1e-6)

        if output_ok:
            output_hits += 1

        print(
            f"{query[:43]:<45} | {str(tool_used):<12} | {str(route_ok):<7} | {str(output_ok):<8} | {dt:<6.2f}"
        )

    n = len(cases)
    routing_acc = routing_hits / n if n else 0.0
    output_acc = output_hits / n if n else 0.0
    avg_t = sum(latencies) / n if n else 0.0

    print("-" * 95)
    print(f"Casos: {n}")
    print(f"Routing Accuracy: {routing_acc:.2%}")
    print(f"Tool Output Accuracy: {output_acc:.2%}")
    print(f"Avg Latency (wall): {avg_t:.3f}s")

    return {
        "section": section_name,
        "n": n,
        "routing_acc": routing_acc,
        "output_acc": output_acc,
        "avg_latency": avg_t,
    }


def main():
    print("Iniciando Evaluación del Agente (flujo explícito)")

    # LLM real o mock
    if USE_REAL_LLM:
        from src.llm.model_loader import LLMService

        llm = LLMService()
    else:
        llm = MockLLMService()

    # Tools reales
    tools = [
        RAGTool(),
        CalculatorTool(),
        VerificationTool(),
    ]

    # core Agente
    agent = AgentEngine(llm, tools)

    # CASOS POR TOOL
    # Calculator: el agente detecta "calcular" o expresión tipo "20 + 5"
    calculator_cases = [
        {
            "query": "calcular 20 + 5",
            "expected_tool": "calculator",
            "check_type": "float",
            "expected": 25.0,
        },
        {
            "query": "¿Cuánto es 10/4?",
            "expected_tool": "calculator",
            "check_type": "float",
            "expected": 2.5,
        },
        {
            "query": "calcular (3*3) + 1",
            "expected_tool": "calculator",
            "check_type": "float",
            "expected": 10.0,
        },
        {
            "query": "15 - 7",
            "expected_tool": "calculator",
            "check_type": "float",
            "expected": 8.0,
        },
        {
            "query": "calcular 2.5 * 4",
            "expected_tool": "calculator",
            "check_type": "float",
            "expected": 10.0,
        },
    ]

    # Verification
    verification_cases = [
        {
            "query": "¿Puedo llevar CS102?",
            "expected_tool": "verification",
            "check_type": "contains",
            "expected": "APPROVED",
        },
        {
            "query": "¿Puedo llevar CS202?",
            "expected_tool": "verification",
            "check_type": "contains",
            "expected": "REJECTED",
        },
        {
            "query": "¿Soy elegible para AI301?",
            "expected_tool": "verification",
            "check_type": "contains",
            "expected": "REJECTED",
        },
        {
            "query": "¿Puedo llevar CS101 otra vez?",
            "expected_tool": "verification",
            "check_type": "contains",
            "expected": "Status",
        },
        {
            "query": "¿Puedo llevar MA101?",
            "expected_tool": "verification",
            "check_type": "contains",
            "expected": "Status",
        },
    ]

    # RAG: evitar queries que contengan códigos tipo CC202 o códigos que estén en verificación
    rag_cases = [
        {
            "query": "¿A qué ciclo pertenece Cálculo Integral?",
            "expected_tool": "rag",
            "check_type": "contains",
            "expected": "(BMA02)",
        },
        {
            "query": "¿Cuántos créditos tiene Física I?",
            "expected_tool": "rag",
            "check_type": "contains",
            "expected": "(BFI01)",
        },
        {
            "query": "¿Física I es obligatorio o electivo en la UNI?",
            "expected_tool": "rag",
            "check_type": "contains",
            "expected": "(BFI01)",
        },
        {
            "query": "¿Cuál es el pre-requisito de Base de Datos Avanzadas?",
            "expected_tool": "rag",
            "check_type": "contains",
            "expected": "Pre-requisito: CC202",
        },
        {
            "query": "Qué cursos hay en el segundo ciclo",
            "expected_tool": "rag",
            "check_type": "contains",
            "expected": "Ubicación: Segundo ciclo",
        },
    ]

    results = []
    results.append(evaluate_cases(agent, calculator_cases, "Calculator"))
    results.append(evaluate_cases(agent, verification_cases, "Verification"))
    results.append(evaluate_cases(agent, rag_cases, "RAG"))

    print("\nResumen Global")
    print("-" * 70)
    print(
        f"{'Tool':<14} | {'N':<3} | {'RouteAcc':<10} | {'OutAcc':<10} | {'Avg t(s)':<8}"
    )
    print("-" * 70)
    for r in results:
        print(
            f"{r['section']:<14} | {r['n']:<3} | {r['routing_acc']:<10.2%} | {r['output_acc']:<10.2%} | {r['avg_latency']:<8.3f}"
        )
    print("-" * 70)


if __name__ == "__main__":
    main()
