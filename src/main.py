import os
from src.llm.model_loader import LLMService
from src.tools.calculator import CalculatorTool
from src.tools.rag import RAGTool
from src.tools.verification import VerificationTool  # <--- [NUEVO] Importar
from src.agent.core import AgentEngine
from src.config import Config


def main():
    # 0. Setup de Data (Igual que antes)
    os.makedirs(os.path.dirname(Config.DATA_PATH), exist_ok=True)
    if not os.path.exists(Config.DATA_PATH):
        with open(Config.DATA_PATH, "w") as f:
            f.write("El curso CC0C2 se imparte en el semestre 2024-2.\n")
            f.write("La nota final depende de E1, E2 y la exposicion oral.\n")
            f.write(
                "Computer Science es el estudio de la computacion y la informacion.\n"
            )

    print("=== Initializing Agentic System (CPU Optimized) ===")

    # 1. Servicios
    llm_service = LLMService()

    # 2. Herramientas: AQUÍ AGREGAMOS LA TERCERA HERRAMIENTA
    tools = [
        CalculatorTool(),
        RAGTool(),
        VerificationTool(),  # <--- [NUEVO] Instancia añadida
    ]

    # 3. Agente
    agent = AgentEngine(llm_service, tools)

    print("\nSistema listo. Escribe 'exit' para salir.")
    print("-" * 50)

    # Loop principal
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
