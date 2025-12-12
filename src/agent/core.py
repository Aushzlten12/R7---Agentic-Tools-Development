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
        context_messages = []
        trace_steps = []

        # --- LÓGICA DE ENRUTAMIENTO (ROUTER) ---

        # 1. VERIFICACIÓN: Solo si hay un código de curso explícito (Ej: CS101, MA302)
        # Quitamos 'or "requisito"' para que las preguntas generales vayan al RAG
        course_match = re.search(r"\b[A-Z]{2}\d{3}\b", query.upper())

        if course_match:
            print("--> Triggering Verification Tool")
            tool_output = self.tools["verification"].run(query)
            print(f"[DEBUG] Tool Output: {tool_output}")
            # Limpiamos el output para el LLM
            context_messages.append(f"{tool_output}")
            trace_steps.append({"tool": "verification", "output": tool_output})

        # 2. CALCULADORA (Se mantiene igual)
        elif "calcular" in query.lower() or re.search(r"\d+\s*[\+\-\*\/]", query):
            # ... (código existente) ...
            print("--> Triggering Calculator Tool")
            tool_output = self.tools["calculator"].run(query)
            print(f"[DEBUG] Tool Output: {tool_output}")
            context_messages.append(f"El resultado es: {tool_output}")  # Texto simple
            trace_steps.append({"tool": "calculator", "output": tool_output})

        # 3. RAG (Todo lo demás cae aquí)
        else:
            print("--> Triggering RAG Tool")
            # CAMBIO: k=2 (Trae 2 fragmentos) y alpha=0.1 (Prioriza palabras clave al 90%)
            tool_output = self.tools["rag"].run(query, k=2, alpha=0.1)

            # Limpieza para el Debug en consola
            clean_debug = tool_output.replace("\n", " ")[:150]
            print(f"[DEBUG] Tool Output: {clean_debug}...")

            # Pasamos el output al contexto
            context_messages.append(tool_output)
            trace_steps.append({"tool": "rag", "output": tool_output})

        # --- SÍNTESIS CON LLM ---

        full_context = "\n".join(context_messages)

        # Generar respuesta final usando el contexto de la herramienta seleccionada
        final_answer = self.llm.generate_response(query, full_context)

        # Log de auditoría
        self.logger.log_interaction(query, trace_steps, final_answer, 0)

        return final_answer
