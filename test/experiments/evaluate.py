import sys
import os
import numpy as np

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from src.tools.rag import RAGTool


def calculate_metrics(retrieved_docs, relevant_text):
    """
    Retrieved docs: Lista de strings recuperados.
    Relevant text: String que DEBE estar en la respuesta.
    """
    # Recall@K: ¿Aparece el texto relevante en algún documento recuperado?
    hit = any(relevant_text.lower() in doc.lower() for doc in retrieved_docs)

    # MRR (Mean Reciprocal Rank): 1/rank de la primera aparición
    rank = 0
    for i, doc in enumerate(retrieved_docs):
        if relevant_text.lower() in doc.lower():
            rank = i + 1
            break

    reciprocal_rank = 1.0 / rank if rank > 0 else 0.0
    return hit, reciprocal_rank


def run_evaluation():
    print("=== Iniciando Evaluación de RAG (E1) ===")
    rag = RAGTool()

    # Dataset de prueba: (Query, Fragmento que debería recuperar)
    test_cases = [
        ("¿Cómo se evalúa el curso?", "examen final"),
        ("¿Qué temas ve el curso?", "Computer Science"),
        ("Información sobre proyectos", "trabajo individual"),
    ]

    hits = 0
    mrr_sum = 0
    k = 3

    print(f"\nEvaluando top-{k} resultados...")
    print("-" * 60)
    print(f"{'Query':<40} | {'Hit?':<5} | {'MRR':<5}")
    print("-" * 60)

    for query, target in test_cases:
        # Ejecutamos el tool y separamos los docs (están unidos por \n\n)
        result_str = rag.run(query, k=k)
        retrieved_docs = result_str.split("\n\n")

        hit, mrr = calculate_metrics(retrieved_docs, target)

        hits += 1 if hit else 0
        mrr_sum += mrr

        print(f"{query[:37]:<40} | {str(hit):<5} | {mrr:.2f}")

    print("-" * 60)
    print(f"Total Queries: {len(test_cases)}")
    print(f"Recall@{k}: {hits / len(test_cases):.2%}")
    print(f"Mean Reciprocal Rank (MRR): {mrr_sum / len(test_cases):.4f}")
    print("=" * 60)


if __name__ == "__main__":
    run_evaluation()
