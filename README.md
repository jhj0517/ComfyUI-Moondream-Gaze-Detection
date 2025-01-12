# ComfyUI Moondream Gaze Detection

This is the [ComfyUI](https://github.com/comfyanonymous/ComfyUI) custom node for [Moondream](https://github.com/vikhyat/moondream)'s [gaze detection feature](https://huggingface.co/spaces/moondream/gaze-demo).

https://github.com/user-attachments/assets/cc2b59ee-ad53-4b0e-a3f4-6cc4c234bf7e

## Installation

1. Place this repository into `ComfyUI\custom_nodes\`
```
git clone https://github.com/jhj0517/ComfyUI-Moondream-Gaze-Detection.git
```

2. Go to `ComfyUI\custom_nodes\ComfyUI-Moondream-Gaze-Detection` and run
```
pip install -r requirements.txt
```

If you are using the portable version of ComfyUI, do this:
```
python_embeded\python.exe -m pip install -r ComfyUI\custom_nodes\ComfyUI-Moondream-Gaze-Detection\requirements.txt
```

## Workflows
Example workflows that generate from single image and video can be found in the [examples/](https://github.com/jhj0517/ComfyUI-Moondream-Gaze-Detection/tree/master/examples) directory.

## Models

Models are automatically downloaded from:
https://huggingface.co/vikhyatk/moondream2/tree/main

To the path of your `ComfyUI/models/moondream`.

### VRAM Usage
Peak VRAM for the model was 6GB on my end.
