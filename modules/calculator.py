from typing import Optional

class CalculatorModel:
    def __init__(self):
        self.model = None

    def load_model(self,
                   model_path: str,
                   device: str,
                   compute_type: Optional[str] = None,
                   ):
        self.model = 1
        print("Calculator model loaded")

    @staticmethod
    def plus(a, b):
        return a+b

    @staticmethod
    def minus(a, b):
        return a-b
