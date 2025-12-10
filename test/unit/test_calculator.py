import pytest
from src.tools.calculator import CalculatorTool


class TestCalculatorTool:

    @pytest.fixture
    def tool(self):
        """Fixture para instanciar la herramienta una sola vez."""
        return CalculatorTool()

    def test_basic_arithmetic(self, tool):
        """Prueba operaciones aritméticas básicas."""
        assert tool.run("2 + 2") == "4"
        assert tool.run("10 - 4") == "6"
        assert tool.run("5 * 5") == "25"
        assert tool.run("20 / 2") == "10.0"  # Python 3 devuelve float en división

    def test_order_of_operations(self, tool):
        """Prueba precedencia de operadores y paréntesis."""
        # Multiplicación antes que suma: 2 + (3*4) = 14
        assert tool.run("2 + 3 * 4") == "14"
        # Paréntesis fuerzan suma: (2+3) * 4 = 20
        assert tool.run("(2 + 3) * 4") == "20"

    def test_extraction_from_noise(self, tool):
        """Prueba que extraiga la matemática dentro de una frase de lenguaje natural."""
        # El usuario escribe texto, el regex debe aislar "25 * 4"
        assert tool.run("Por favor calcula cuanto es 25 * 4 gracias") == "100"
        assert tool.run("El resultado de 10+10 es requerido") == "20"

    def test_floating_point(self, tool):
        """Prueba manejo de decimales."""
        assert tool.run("2.5 + 2.5") == "5.0"

    def test_error_handling(self, tool):
        """Prueba manejo de errores de sintaxis o matemáticos."""
        # División por cero
        result = tool.run("10 / 0")
        assert "Error" in result or "division by zero" in result

        # Sintaxis inválida que pasa el regex pero falla el eval
        # "1.2.3" pasa el regex [\d\.]+ pero no es un número válido
        result_syntax = tool.run("La versión es 1.2.3")
        assert "Error" in result_syntax

    def test_no_calculation(self, tool):
        """Prueba cuando no hay números."""
        assert tool.run("Hola mundo") == "No calculation found."
