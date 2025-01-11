from .nodes import *


#  Map all your custom nodes classes with the names that will be displayed in the UI.
NODE_CLASS_MAPPINGS = {
    "(Down)Load My Model": MyModelLoader,
    "Calculate Plus": CalculatePlus,
    "Calculate Minus": CalculateMinus,
    "Example Output Node": ExampleOutputNode,
}


__all__ = ['NODE_CLASS_MAPPINGS']
