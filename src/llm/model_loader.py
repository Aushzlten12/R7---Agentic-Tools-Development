from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from src.config import Config


class LLMService:
    def __init__(self):
        print(f"Loading Model {Config.LLM_MODEL_ID} on CPU...")
        self.tokenizer = AutoTokenizer.from_pretrained(Config.LLM_MODEL_ID)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(Config.LLM_MODEL_ID)

        # Pipeline de generación de texto
        self.pipe = pipeline(
            "text2text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_length=256,
        )

    def generate_response(self, query: str, context: str) -> str:
        # Prompt Engineering básico para FLAN-T5
        input_text = f"Question: {query}\nContext: {context}\nAnswer:"
        output = self.pipe(input_text)
        return output[0]["generated_text"]
