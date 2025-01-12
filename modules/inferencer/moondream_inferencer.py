import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import PIL
from PIL import Image
from transformers import AutoModelForCausalLM
from typing import (Union, Tuple, List, Dict, Optional, Any)
import cv2
import io

from .pyvips_dll_handler import handle_pyvips_dll_error


class MoondreamInferencer:
    def __init__(self,
                 model_dir: str):
        self.model = None
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)

    def load_model(self,
                   device: str = "cuda"):
        self.model = AutoModelForCausalLM.from_pretrained(
            "vikhyatk/moondream2",
            trust_remote_code=True,
            device_map={"": device},
            cache_dir=self.model_dir,
        )

    def process_image(
        self,
        input_image: Union[np.ndarray, Any],
        use_ensemble: bool
    ):
        if self.model is None:
            self.load_model()

        if input_image is None:
            return None, ""

        try:
            if isinstance(input_image, torch.Tensor):
                if input_image.dim() == 4 and input_image.shape[0] == 1:
                    input_image = input_image[0]  # shape now (H, W, 3)
                image_np = (input_image.cpu().numpy() * 255).astype(np.uint8)

                pil_image = Image.fromarray(image_np).convert("RGB")

            elif isinstance(input_image, np.ndarray):
                pil_image = Image.fromarray(input_image)

            else:
                pil_image = input_image

            enc_image = self.model.encode_image(pil_image)
            if use_ensemble:
                flipped_pil = pil_image.copy().transpose(method=Image.FLIP_LEFT_RIGHT)
                flip_enc_image = self.model.encode_image(flipped_pil)
            else:
                flip_enc_image = None

            faces = self.model.detect(enc_image, "face")["objects"]

            if not faces:
                return None, "No faces detected in the image."

            face_boxes = []
            gaze_points = []

            for face in faces:
                # Add face bounding box regardless of gaze detection
                face_box = (
                    face["x_min"] * pil_image.width,
                    face["y_min"] * pil_image.height,
                    (face["x_max"] - face["x_min"]) * pil_image.width,
                    (face["y_max"] - face["y_min"]) * pil_image.height,
                )
                face_center = (
                    (face["x_min"] + face["x_max"]) / 2,
                    (face["y_min"] + face["y_max"]) / 2
                )
                face_boxes.append(face_box)

                # Try to detect gaze
                gaze_settings = {
                    "prioritize_accuracy": use_ensemble,
                    "flip_enc_img": flip_enc_image
                }
                gaze = self.model.detect_gaze(enc_image, face=face, eye=face_center, unstable_settings=gaze_settings)["gaze"]

                if gaze is not None:
                    gaze_point = (
                        gaze["x"] * pil_image.width,
                        gaze["y"] * pil_image.height,
                    )
                    gaze_points.append(gaze_point)
                else:
                    gaze_points.append(None)

            # Create visualization
            image_array = np.array(pil_image)
            fig = self.visualize_faces_and_gaze(
                face_boxes, gaze_points, image=image_array, show_plot=False
            )

            faces_with_gaze = sum(1 for gp in gaze_points if gp is not None)
            status = f"Found {len(faces)} faces. {len(faces) - faces_with_gaze} gazing out of frame."
            return fig, status

        except Exception as e:
            return None, f"Error processing image: {str(e)}"

    @staticmethod
    def visualize_faces_and_gaze(face_boxes, gaze_points=None, image=None, show_plot=True):
        """Visualization function that can handle faces without gaze data"""
        # Calculate figure size based on image aspect ratio
        if image is not None:
            height, width = image.shape[:2]
            aspect_ratio = width / height
            fig_height = 6  # Base height
            fig_width = fig_height * aspect_ratio
        else:
            width, height = 800, 600
            fig_width, fig_height = 10, 8

        fig = plt.figure(figsize=(fig_width, fig_height))
        ax = fig.add_subplot(111)

        if image is not None:
            ax.imshow(image)
        else:
            ax.set_facecolor("#1a1a1a")
            fig.patch.set_facecolor("#1a1a1a")

        colors = plt.cm.rainbow(np.linspace(0, 1, len(face_boxes)))

        for i, (face_box, color) in enumerate(zip(face_boxes, colors)):
            hex_color = "#{:02x}{:02x}{:02x}".format(
                int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)
            )

            x, y, width_box, height_box = face_box
            face_center_x = x + width_box / 2
            face_center_y = y + height_box / 2

            # Draw face bounding box
            face_rect = plt.Rectangle(
                (x, y), width_box, height_box, fill=False, color=hex_color, linewidth=2
            )
            ax.add_patch(face_rect)

            # Draw gaze line if gaze data is available
            if gaze_points is not None and i < len(gaze_points) and gaze_points[i] is not None:
                gaze_x, gaze_y = gaze_points[i]

                points = 50
                alphas = np.linspace(0.8, 0, points)

                x_points = np.linspace(face_center_x, gaze_x, points)
                y_points = np.linspace(face_center_y, gaze_y, points)

                for j in range(points - 1):
                    ax.plot(
                        [x_points[j], x_points[j + 1]],
                        [y_points[j], y_points[j + 1]],
                        color=hex_color,
                        alpha=alphas[j],
                        linewidth=4,
                    )

                ax.scatter(gaze_x, gaze_y, color=hex_color, s=100, zorder=5)
                ax.scatter(gaze_x, gaze_y, color="white", s=50, zorder=6)

        # Set plot limits and remove axes
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

        # Remove padding around the plot
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)

        return fig

    @staticmethod
    def figure_to_tensor(fig) -> torch.Tensor:
        """
        Converts a matplotlib Figure into a PyTorch tensor of shape.
        """
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
        plt.close(fig)

        buf.seek(0)
        pil_img = Image.open(buf).convert("RGB")

        np_img = np.array(pil_img, dtype=np.float32) / 255.0

        tensor_img = torch.from_numpy(np_img).unsqueeze(0)
        return tensor_img

#
# if __name__ == "__main__":
#
#     matplotlib.use("Agg")
#     handle_pyvips_dll_error(download_dir=os.path.join("."))
#
#     model = MoondreamInferencer(
#         model_dir=os.path.join("models"),
#     )
#
#     image_path = "face.jpg"
#     image_bgr = cv2.imread(image_path)
#     if image_bgr is None:
#         print(f"Could not load image from {image_path}. Please check the path.")
#     else:
#         # Convert to RGB for consistency
#         image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
#
#         # Call process_image
#         fig, status = model.process_image(image_rgb, use_ensemble=False)
#         print(status)
#
#         # Save the figure if one was returned
#         if fig is not None:
#             output_file = "visualization.png"
#             fig.savefig(output_file)
#             print(f"Visualization saved to {output_file}")
#         else:
#             print("No figure to save.")
