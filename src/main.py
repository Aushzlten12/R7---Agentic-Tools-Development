import os
from src.llm.model_loader import LLMService
from src.tools.calculator import CalculatorTool
from src.tools.rag import RAGTool
from src.tools.verification import VerificationTool  # <--- Importación Correcta
from src.agent.core import AgentEngine
from src.config import Config


def main():
    # Directorio de logs
    os.makedirs(Config.LOG_DIR, exist_ok=True)

    print("Iniciando Agente")

    # Inicializar LLM service
    llm_service = LLMService()

    # Inicializar Herramientas
    tools = [CalculatorTool(), RAGTool(), VerificationTool()]

    # Inicializar Agente
    agent = AgentEngine(llm_service, tools)

    print("\nSistema listo. Escribe 'exit' para salir.")
    print("-" * 50)

    # Loop de consola
    while True:
        user_query = input("User> ")
        if user_query.lower() in ["exit", "quit"]:
            break

        try:
            response, latency = agent.run(user_query)
            print(f"Agent> {response}")
            print(f"[Meta] Latency: {latency:.4f}s | Logs saved.")
        except Exception as e:
            print(f"Error: {e}")
        print("-" * 50)


if __name__ == "__main__":
    main()
