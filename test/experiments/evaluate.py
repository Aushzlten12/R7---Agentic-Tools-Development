import sys
import os
import time
import numpy as np

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from src.tools.rag import RAGTool


def split_retrieved(result_str: str):
    """
    el RAG a veces retorna docs separados por '\n\n' (top-k normal),
    pero para consultas tipo 'qué cursos hay en X ciclo' retorna líneas con '\n'.
    """
    if "\n\n" in result_str:
        docs = [d.strip() for d in result_str.split("\n\n") if d.strip()]
    else:
        docs = [d.strip() for d in result_str.split("\n") if d.strip()]
    return docs


def calculate_metrics(retrieved_docs, relevant_text):
    # Recall@K: ¿Aparece el texto relevante en algún documento recuperado?
    target = relevant_text.lower()
    hit = any(target in doc.lower() for doc in retrieved_docs)

    # MRR: 1/rank de la primera aparición
    rank = 0
    for i, doc in enumerate(retrieved_docs):
        if target in doc.lower():
            rank = i + 1
            break

    reciprocal_rank = 1.0 / rank if rank > 0 else 0.0
    return hit, reciprocal_rank


def run_mode(rag: RAGTool, test_cases, k: int, alpha: float, mode_name: str):
    hits = 0
    mrr_sum = 0.0
    latencies = []

    print(f"\n=== Modo: {mode_name} | alpha={alpha} | k={k} ===")
    print("-" * 85)
    print(f"{'Query':<48} | {'Target':<18} | {'Hit':<5} | {'MRR':<5} | {'t(s)':<6}")
    print("-" * 85)

    for query, target in test_cases:
        t0 = time.time()
        result_str = rag.run(query, k=k, alpha=alpha)
        dt = time.time() - t0

        retrieved_docs = split_retrieved(result_str)
        hit, mrr = calculate_metrics(retrieved_docs, target)

        hits += 1 if hit else 0
        mrr_sum += mrr
        latencies.append(dt)

        print(
            f"{query[:46]:<48} | {target[:16]:<18} | {str(hit):<5} | {mrr:.2f} | {dt:.2f}"
        )

    recall = hits / len(test_cases)
    mrr_avg = mrr_sum / len(test_cases)
    t_avg = float(np.mean(latencies)) if latencies else 0.0

    print("-" * 85)
    print(f"Total Queries: {len(test_cases)}")
    print(f"Recall@{k}: {recall:.2%}")
    print(f"MRR: {mrr_avg:.4f}")
    print(f"Avg Latency: {t_avg:.3f}s")
    return recall, mrr_avg, t_avg


def run_evaluation():
    print("=== Iniciando Evaluación de RAG (E1) ===")
    rag = RAGTool()

    # PDF uni
    test_cases = [
        ("¿A qué ciclo pertenece Cálculo Integral?", "(BMA02)"),
        ("¿Cuál es el pre-requisito de Cálculo Integral?", "(BMA02)"),
        ("¿Cuántos créditos tiene Física I?", "(BFI01)"),
        ("¿Física I es obligatorio o electivo en la UNI?", "(BFI01)"),
        ("¿A qué ciclo pertenece Cálculo Diferencial?", "(BMA01)"),
        (
            "¿Cuál es el pre-requisito de Base de Datos Avanzadas?",
            "Pre-requisito: CC202",
        ),
        ("¿Cuáles son los créditos de Base de Datos Avanzadas?", "(CC0A1)"),
        ("¿Base de Datos (CC202) es obligatorio o electivo?", "(CC202)"),
        (
            "¿Idioma Extranjero o Lengua Nativa en el Nivel Intermedio es obligatorio?",
            "(BIE01)",
        ),
        ("¿Tópicos de Ciencia de la Computación I es electivo?", "(CC0F4)"),
        ("Qué cursos hay en el segundo ciclo", "Ubicación: Segundo ciclo"),
    ]

    # Se puede modificar el valor de k
    k = 3

    # Comparación de modos usando alpha:
    # sparse-only: alpha=0.0
    # dense-only:  alpha=1.0
    # hybrid:      alpha=0.45
    modes = [
        ("SPARSE (BM25)", 0.0),
        ("DENSE (FAISS)", 1.0),
        ("HYBRID", 0.45),
    ]

    results = []
    for name, alpha in modes:
        recall, mrr, tavg = run_mode(rag, test_cases, k=k, alpha=alpha, mode_name=name)
        results.append((name, recall, mrr, tavg))

    print("\n=== Resumen ===")
    print("-" * 60)
    print(f"{'Modo':<16} | {'Recall@3':<10} | {'MRR':<8} | {'Avg t(s)':<8}")
    print("-" * 60)
    for name, recall, mrr, tavg in results:
        print(f"{name:<16} | {recall:<10.2%} | {mrr:<8.4f} | {tavg:<8.3f}")
    print("-" * 60)


if __name__ == "__main__":
    run_evaluation()
