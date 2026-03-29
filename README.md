# ADLCV-VisualDebuggerVLM
Final project for the Advanced Deep Learning in Computer Vision course.


# Environment setup
We will be using uv as the python package manager. To set up the environment, run the following command in the terminal:

To install the dependencies, run:
```bash
uv sync --extra cpu
```
If you have a compatible NVIDIA GPU and want to install torch with CUDA support, run:
```bash
uv sync --extra cu128
```