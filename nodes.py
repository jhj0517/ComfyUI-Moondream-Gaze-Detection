#  Package Modules
import os
from typing import (Union, BinaryIO, Dict, List, Tuple, Optional)
import time

#  ComfyUI Modules
import folder_paths
from comfy.utils import ProgressBar

#  Your Modules
from .modules.inferencer.moondream_inferencer import MoondreamInferencer


#  Basic practice to get paths from ComfyUI
custom_nodes_script_dir = os.path.dirname(os.path.abspath(__file__))
custom_nodes_model_dir = os.path.join(folder_paths.models_dir, "moondream")
os.makedirs(custom_nodes_model_dir, exist_ok=True)


def get_category_name():
    return "Moondream Gaze Detection"


# total_steps = 5
# comfy_pbar = ProgressBar(total_steps)
# #  Then, update the progress.
# for i in range(1, total_steps):
#     time.sleep(1)
#     comfy_pbar.update(
#         i)  # Alternatively, you can use `comfy_pbar.update_absolute(value)` to update the progress with absolute value.


class MoondreamModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "device": (['cuda', 'cpu'],),
            },
        }

    RETURN_TYPES = ("MOONDREAM_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = get_category_name()

    def load_model(self,
                   device: str,
                   ) -> Tuple[MoondreamInferencer]:

        model_inferencer = MoondreamInferencer(model_dir=custom_nodes_model_dir)
        model_inferencer.load_model(device=device)

        return (model_inferencer, )


class GazeDetection:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MOONDREAM_MODEL", ),
            },
            #  Specify the parameters with type and default value.
            "optional": {
                "a": ("INT", {"default": 5}),
                "b": ("INT", {"default": 10}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "gaze_detection"
    CATEGORY = get_category_name()

    def gaze_detection(self,
                       model: MoondreamInferencer,
                       image: Optional[int],
                       ) -> Tuple[int]:
        fig, status = model.process_image(image, use_ensemble=False)
        return (fig, )

