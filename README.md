# Research Code for Multi-Targeted Universal Adversarial Perturbations

## Preliminary

### Docker (suggested)
docker pull huggingface/transformers-pytorch-gpu:4.41.3

### UV
```bash
pip install uv
uv sync 
uv tool install ruff
```

# Code Style and Quality
Code style follows PEP8 standards with `Ruff` formatting.
Use the following commands in the project direcotry to check the format.
```bash
# 
ruff format
ruff check
```