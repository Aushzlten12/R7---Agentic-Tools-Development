import re
from src.tools.base import BaseTool


class CalculatorTool(BaseTool):
    def __init__(self):
        super().__init__(name="calculator")

    def run(self, input_text: str) -> str:
        # Extraer expresión matemática simple (números y operadores)
        # Regex busca patrones como "20 + 5" o "3*3"
        matches = re.findall(r"[\d\+\-\*\/\.\(\)\s]+", input_text)
        valid_exprs = [m for m in matches if any(char.isdigit() for char in m)]

        if not valid_exprs:
            return "No calculation found."

        # Tomamos la coincidencia más larga que parezca una operación
        expr = max(valid_exprs, key=len).strip()

        try:
            # NOTA: En producción, usar una librería como simpleeval por seguridad.
            # Para E1 académico, eval() restringido es aceptable.
            result = eval(expr, {"__builtins__": None}, {})
            return str(result)
        except Exception as e:
            return f"Error in calculation: {e}"
