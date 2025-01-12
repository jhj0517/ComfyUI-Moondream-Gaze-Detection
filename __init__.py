from .nodes import *


#  Map all your custom nodes classes with the names that will be displayed in the UI.
NODE_CLASS_MAPPINGS = {
    "(Down)Load Moondream Model": MoondreamModelLoader,
    "Gaze Detection": GazeDetection,
}


__all__ = ['NODE_CLASS_MAPPINGS']
