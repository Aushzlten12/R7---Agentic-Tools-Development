from abc import ABC, abstractmethod


class BaseTool(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def run(self, input_text: str) -> str:
        pass
