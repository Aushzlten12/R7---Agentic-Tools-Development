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
        response, trace_steps = self._execute_explicit_workflow(query)
        latency = time.time() - start_time
        self.logger.log_interaction(query, trace_steps, response, latency)
        return response, latency

    def _execute_explicit_workflow(self, query: str):
        context_messages = []
        trace_steps = []

        # ROUTER EXPLICITO

        # VERIFICACIÓN: Solo si hay un código de curso explícito
        course_match = re.search(r"\b[A-Z]{2}\d{3}\b", query.upper())

        if course_match:
            print("--> Triggering Verification Tool")
            tool_output = self.tools["verification"].run(query)
            print(f"[DEBUG] Tool Output: {tool_output}")
            # Limpiamos el output para el LLM
            context_messages.append(f"{tool_output}")
            trace_steps.append({"tool": "verification", "output": tool_output})

        # CALCULADORA
        elif "calcular" in query.lower() or re.search(r"\d+\s*[\+\-\*\/]", query):
            # ... (código existente) ...
            print("--> Triggering Calculator Tool")
            tool_output = self.tools["calculator"].run(query)
            print(f"[DEBUG] Tool Output: {tool_output}")
            context_messages.append(f"El resultado es: {tool_output}")  # Texto simple
            trace_steps.append({"tool": "calculator", "output": tool_output})

        # RAG
        else:
            print("RAG Tool")

            tool_output = self.tools["rag"].run(query, k=3, alpha=0.45)

            clean_debug = tool_output.replace("\n", " ")[:150]
            print(f"[DEBUG] Tool Output: {clean_debug}...")

            context_messages.append(tool_output)
            trace_steps.append({"tool": "rag", "output": tool_output})

        full_context = "\n".join(context_messages)

        # Generar respuesta final usando el contexto de la herramienta seleccionada
        final_answer = self.llm.generate_response(query, full_context)

        # Log
        self.logger.log_interaction(query, trace_steps, final_answer, 0)

        return final_answer, trace_steps
