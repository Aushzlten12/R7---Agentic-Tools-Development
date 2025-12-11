import re
import time
from src.utils.logger import AgentLogger


class AgentEngine:
    def __init__(self, llm_service, tools: list):
        self.llm = llm_service
        self.tools = {t.name: t for t in tools}
        self.logger = AgentLogger()

    def run(self, query: str):
        start_time = time.time()
        response = self._execute_explicit_workflow(query)
        latency = time.time() - start_time
        return response, latency

    def _execute_explicit_workflow(self, query: str):
        """
        Router Explícito (Deterministic DFA).
        Decide qué herramienta usar basándose en Regex/Keywords.
        """
        context_messages = []
        trace_steps = []

        # --- LÓGICA DE ENRUTAMIENTO (ROUTER) ---

        # CASO 1: Intención Académica/Verificación (Detectar códigos tipo CS101)
        # Priorizamos esto porque es muy específico
        if (
            re.search(r"\b[A-Z]{2}\d{3}\b", query.upper())
            or "requisito" in query.lower()
        ):
            print("--> Triggering Verification Tool")
            tool_output = self.tools["verification"].run(query)

            context_messages.append(f"Student Record System: {tool_output}")
            trace_steps.append({"tool": "verification", "output": tool_output})

        # CASO 2: Intención Matemática (Calculadora)
        elif "calcular" in query.lower() or re.search(r"\d+\s*[\+\-\*\/]", query):
            print("--> Triggering Calculator Tool")
            tool_output = self.tools["calculator"].run(query)

            context_messages.append(f"Calculation Result: {tool_output}")
            trace_steps.append({"tool": "calculator", "output": tool_output})

        # CASO 3: Fallback / Pregunta General (RAG)
        # Si no es verificación ni cálculo, asumimos que es una pregunta sobre documentos
        else:
            print("--> Triggering RAG Tool")
            tool_output = self.tools["rag"].run(query)

            context_messages.append(f"Retrieved Documents: {tool_output}")
            trace_steps.append({"tool": "rag", "output": tool_output})

        # --- SÍNTESIS CON LLM ---

        full_context = "\n".join(context_messages)

        # Generar respuesta final usando el contexto de la herramienta seleccionada
        final_answer = self.llm.generate_response(query, full_context)

        # Log de auditoría
        self.logger.log_interaction(query, trace_steps, final_answer, 0)

        return final_answer
