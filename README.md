# ComfyUI CustomNode Template

This is the [ComfyUI](https://github.com/comfyanonymous/ComfyUI) custom node template repository that anyone can use to create their own custom nodes.

## Directory Structure
```
Project-Name/
├── .github/                # GA workflow for publishing the ComfyUI registry 
├── example/                # Example workflows for your custom node
├── modules/                # Your own modules for the custom node
├── .gitignore              # gitignore file 
├── __init__.py             # Map your custom node display names here 
├── nodes.py                # Your custom node classes  
├── README.md               # README file
├── pyproject.toml          # Metadata file for the ComfyUI registry
└── requirements.txt        # Project dependencies 
```

## Custom Node Files

### [nodes.py](https://github.com/jhj0517/ComfyUI-CustomNodes-Template/tree/master/nodes.py)
This file is where your actual custom node classes are defined. The class has specific methods that are called by the ComfyUI engine.
There're some basic custom nodes for the example with some comments, you can modify them as you need.
For detailed information on how to create custom nodes, please refer to the ComfyUI official documentation: 
- https://docs.comfy.org/essentials/custom_node_walkthrough.

### [__init__.py](https://github.com/jhj0517/ComfyUI-CustomNodes-Template/tree/master/__init__.py)
You can map your custom node display names here. It will be used when users search for your custom node in the ComfyUI.

### [pyproject.toml](https://github.com/jhj0517/ComfyUI-CustomNodes-Template/tree/master/pyproject.toml)
This file is used to publish your custom node to the ComfyUI registry. If you want to publish your custom node to the ComfyUI registry, you need to modify this file.

If you wonder what ComfyUI registry is, please read:

- https://docs.comfy.org/registry/overview#why-use-the-registry

### [requirements.txt](https://github.com/jhj0517/ComfyUI-CustomNodes-Template/tree/master/requirements.txt)
This file contains the dependencies needed for your custom node. `torch` is already installed in the ComfyUI, so you only need to add "extra" dependencies here.

### [examples/example-1.json](https://github.com/jhj0517/ComfyUI-CustomNodes-Template/tree/master/examples)
This is optional, but it is recommended to put your ComfyUI workflow json file inside your project so users can easily understand how to use your custom node.

## Github Actions

### [publish-comfyui-registry.yml](https://github.com/jhj0517/ComfyUI-CustomNodes-Template/tree/master/.github/workflows/publish-comfyui-registry.yml)
When you push into the `master` branch, this workflow will be triggered and publish your custom node to the ComfyUI registry, using your [pyproject.toml](https://github.com/jhj0517/ComfyUI-CustomNodes-Template/tree/master/pyproject.toml).
You have to register your "REGISTRY_ACCESS_TOKEN" in the Github Action Secrets which you can get from:
- https://docs.comfy.org/registry/publishing#create-an-api-key-for-publishing

After generating the repository from this template, uncomment the branches to enable the workflow with auto trigger:

https://github.com/jhj0517/ComfyUI-CustomNodes-Template/blob/24941ae8995217490f2f37aceaf5ccf9ae632995/.github/workflows/publish-comfyui-registry.yml#L6-L8

## How to Strat Using Template

Click "Use this template" -> "Create a new repository", then you can create your own custom node from there.

![image](https://github.com/user-attachments/assets/fab4da53-0458-4e88-adc1-5bb5d341a511)


The custom node installation guide below can usually be used for any custom node, you can use it in your README by modifying the repository name and URL.
## Installation

1. git clone repository into `ComfyUI\custom_nodes\`
```
git clone https://github.com/replace-this-with-your-github-repository-url.git
```

2. Go to `ComfyUI\custom_nodes\ComfyUI-Your-CustomNode-Name` and run
```
pip install -r requirements.txt
```

If you are using the portable version of ComfyUI, do this:
```
python_embeded\python.exe -m pip install -r ComfyUI\custom_nodes\ComfyUI-Your-CustomNode-Name\requirements.txt
```


