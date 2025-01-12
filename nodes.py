import os
from typing import (Union, BinaryIO, Dict, List, Tuple, Optional, Any)
import torch
import time

import folder_paths
from comfy.utils import ProgressBar

from .modules.inferencer.moondream_inferencer import MoondreamInferencer
from .modules.inferencer.pyvips_dll_handler import handle_pyvips_dll_error


custom_nodes_script_dir = os.path.dirname(os.path.abspath(__file__))
custom_nodes_model_dir = os.path.join(folder_paths.models_dir, "moondream")
os.makedirs(custom_nodes_model_dir, exist_ok=True)


def get_category_name():
    return "Moondream Gaze Detection"


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
        handle_pyvips_dll_error(download_dir=custom_nodes_script_dir)
        model_inferencer = MoondreamInferencer(model_dir=custom_nodes_model_dir)
        model_inferencer.load_model(device=device)

        return (model_inferencer, )


class GazeDetection:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MOONDREAM_MODEL", ),
                "image": ("IMAGE", )
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "gaze_detection"
    CATEGORY = get_category_name()

    def gaze_detection(self,
                       model: MoondreamInferencer,
                       image: Any,
                       ) -> Tuple[int]:
        fig, status = model.process_image(image, use_ensemble=False)
        out_img = model.figure_to_tensor(fig)

        return (out_img, )


class GazeDetectionVideo:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MOONDREAM_MODEL", ),
                "video": ("IMAGE", )
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "gaze_detection_video"
    CATEGORY = get_category_name()

    def gaze_detection_video(self,
                             model: MoondreamInferencer,
                             video: Any,
                             ) -> Tuple[int]:
        num_frames = video.shape[0]
        height = video.shape[1]
        width = video.shape[2]
        channels = video.shape[3]

        comfy_pbar = ProgressBar(num_frames)
        out_frames = []
        for f in range(num_frames):
            frame_tensor = video[f]
            fig, status = model.process_image(frame_tensor, use_ensemble=False)
            out_img = model.figure_to_tensor(fig)

            out_img = out_img.squeeze(0)
            out_frames.append(out_img)

            comfy_pbar.update(1)

        out_frames_tensor = torch.stack(out_frames, dim=0)

        return (out_frames_tensor, )
