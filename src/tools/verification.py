import re
from src.tools.base import BaseTool


class VerificationTool(BaseTool):
    def __init__(self):
        super().__init__(name="verification")

        # Mock Database: Simulamos el catálogo de cursos y sus requisitos
        self.course_catalog = {
            "CS101": {"name": "Introducción a CS", "prereqs": []},
            "CS102": {"name": "Programación Orientada a Objetos", "prereqs": ["CS101"]},
            "CS202": {"name": "Algoritmos", "prereqs": ["CS102", "MA101"]},
            "MA101": {"name": "Cálculo I", "prereqs": []},
            "AI301": {"name": "Inteligencia Artificial", "prereqs": ["CS202", "MA101"]},
        }

        # Cursos aprobados
        self.student_history = {"CS101", "MA101"}

    def run(self, input_text: str) -> str:
        """
        Analiza el texto buscando códigos de curso (ej. CS102) y verifica elegibilidad.
        """
        # Normalización y extracción mediante Regex
        target_course = re.search(r"\b[A-Z]{2}\d{3}\b", input_text.upper())

        if not target_course:
            return "Error: No se detectó un código de curso (ejemplo: CS101)."

        course_code = target_course.group(0)

        # Verificar existencia del curso
        if course_code not in self.course_catalog:
            return f"Error: Curso {course_code} no encontrado en el catálogo."

        # Verificar si ya lo aprobó
        if course_code in self.student_history:
            return f"Status: Ya aprobaste {course_code}."

        # Verificar prerrequisitos
        requirements = self.course_catalog[course_code]["prereqs"]
        missing_reqs = [req for req in requirements if req not in self.student_history]

        if missing_reqs:
            return (
                f"REJECTED: No puedes llevar {course_code}. "
                f"Missing prerequisites: {', '.join(missing_reqs)}"
            )

        return f"APPROVED: Puedes matricularte en {course_code} ({self.course_catalog[course_code]['name']})."
