from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from src.config import Config


class LLMService:
    def __init__(self):
        print(f"Loading Model {Config.LLM_MODEL_ID} on CPU...")

        self.tokenizer = AutoTokenizer.from_pretrained(Config.LLM_MODEL_ID)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(Config.LLM_MODEL_ID)

        self.pipe = pipeline(
            "text2text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            # CAMBIO: Quitamos max_length de aquí para evitar conflictos
        )

    def generate_response(self, query: str, context: str) -> str:
        # Prompt Estricto
        # Reducimos un poco el contexto para mejorar la velocidad (1200 chars)
        safe_context = context[:1200]

        input_text = (
            f"Información: {safe_context}\n\n"
            f"Instrucción: Responde la pregunta usando solo la información anterior.\n"
            f"Pregunta: {query}\n"
            f"Respuesta:"
        )

        output = self.pipe(
            input_text,
            max_new_tokens=100,  # CAMBIO: Usamos max_new_tokens en vez de max_length
            do_sample=False,
            repetition_penalty=1.2,
        )
        return output[0]["generated_text"]
