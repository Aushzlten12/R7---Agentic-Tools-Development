import pytest
from src.tools.verification import VerificationTool


class TestVerificationTool:
    @pytest.fixture
    def tool(self):
        return VerificationTool()

    def test_aprobado_si_cumple_requisitos(self, tool):
        res = tool.run("Quiero llevar CS102")
        assert "approved" in res.lower()
        assert "cs102" in res.lower()

    def test_rechazado_si_faltan_requisitos(self, tool):
        res = tool.run("Puedo matricularme en AI301?")
        assert "rejected" in res.lower()
        assert ("missing prerequisites" in res.lower()) or (
            "prerrequisitos" in res.lower()
        )

    def test_curso_no_encontrado(self, tool):
        res = tool.run("Quiero llevar CS999")
        assert "no encontrado" in res.lower()
        assert "cs999" in res.lower()

    def test_sin_codigo_en_input(self, tool):
        res = tool.run("Verifica mis requisitos por favor")
        assert "no se detect" in res.lower()
        assert "cs101" in res.lower()

    def test_ya_aprobado(self, tool):
        res = tool.run("Puedo llevar CS101 de nuevo?")
        assert "status" in res.lower()
        assert ("ya aprobaste" in res.lower()) or ("already passed" in res.lower())
