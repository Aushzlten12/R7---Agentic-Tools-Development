import json
import os
import time
from datetime import datetime
from src.config import Config


class AgentLogger:
    def __init__(self):
        os.makedirs(Config.LOG_DIR, exist_ok=True)
        self.log_file = os.path.join(Config.LOG_DIR, "execution.jsonl")

    def log_interaction(self, query, steps, response, latency):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "steps_trace": steps,  # Lista de herramientas usadas y sus outputs
            "final_response": response,
            "latency_seconds": round(latency, 4),
        }

        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
