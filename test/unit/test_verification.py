import pytest
from src.tools.verification import VerificationTool


class TestVerificationTool:

    @pytest.fixture
    def tool(self):
        return VerificationTool()

    def test_course_approval_logic(self, tool):
        """Caso feliz: Cumple requisitos (CS102 requiere CS101, y el alumno tiene CS101)"""
        # Asumiendo que en el mock el alumno tiene CS101
        result = tool.run("Quiero llevar CS102")
        assert "APPROVED" in result
        assert "CS102" in result

    def test_course_rejection_logic(self, tool):
        """Caso negativo: Faltan requisitos (AI301 requiere CS202, alumno no lo tiene)"""
        result = tool.run("Puedo matricularme en AI301?")
        assert "REJECTED" in result
        assert "Missing prerequisites" in result

    def test_course_not_found(self, tool):
        """Caso borde: Código inexistente"""
        result = tool.run("Quiero llevar CS999")
        assert "not found" in result

    def test_no_code_provided(self, tool):
        """Caso de error de input"""
        result = tool.run("Verifica mis requisitos por favor")
        assert "No course code detected" in result

    def test_already_passed(self, tool):
        """Caso: Ya llevó el curso"""
        # El alumno mock tiene CS101
        result = tool.run("Puedo llevar CS101 de nuevo?")
        assert "already passed" in result
