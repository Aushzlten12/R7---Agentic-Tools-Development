import re
from src.tools.base import BaseTool


class VerificationTool(BaseTool):
    def __init__(self):
        super().__init__(name="verification")

        # 1. Mock Database: Simulamos el catálogo de cursos y sus requisitos
        # En un sistema real, esto vendría de una consulta SQL o API REST.
        self.course_catalog = {
            "CS101": {"name": "Intro to CS", "prereqs": []},
            "CS102": {"name": "Objects Oriented Programming", "prereqs": ["CS101"]},
            "CS202": {"name": "Algorithms", "prereqs": ["CS102", "MA101"]},
            "MA101": {"name": "Calculus I", "prereqs": []},
            "AI301": {"name": "Artificial Intelligence", "prereqs": ["CS202", "MA101"]},
        }

        # 2. Mock Student Record: El estado actual del estudiante (Hardcoded para E1)
        self.student_history = {"CS101", "MA101"}  # Cursos ya aprobados

    def run(self, input_text: str) -> str:
        """
        Analiza el texto buscando códigos de curso (ej. CS102) y verifica elegibilidad.
        """
        # Normalización y extracción mediante Regex
        # Busca patrones de 2 letras mayúsculas seguidas de 3 dígitos (ej: CS202)
        target_course = re.search(r"\b[A-Z]{2}\d{3}\b", input_text.upper())

        if not target_course:
            return "Error: No course code detected (format example: CS101)."

        course_code = target_course.group(0)

        # A. Verificar existencia del curso
        if course_code not in self.course_catalog:
            return f"Error: Course {course_code} not found in catalog."

        # B. Verificar si ya lo aprobó
        if course_code in self.student_history:
            return f"Status: You have already passed {course_code}."

        # C. Verificar prerrequisitos (Lógica de conjuntos)
        requirements = self.course_catalog[course_code]["prereqs"]
        missing_reqs = [req for req in requirements if req not in self.student_history]

        if missing_reqs:
            return (
                f"REJECTED: You cannot take {course_code}. "
                f"Missing prerequisites: {', '.join(missing_reqs)}"
            )

        return f"APPROVED: You are eligible to enroll in {course_code} ({self.course_catalog[course_code]['name']})."
