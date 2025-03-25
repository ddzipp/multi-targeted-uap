# Research Code for Multi-Targeted Universal Adversarial Perturbations

## Preliminary
There are two ways to construct the running environment.


### UV
```bash
pip install uv
uv sync 
uv pip install ruff
```
### Docker
```bash
docker pull huggingface/transformers-pytorch-gpu:4.41.3
docker compose -f ./docker-compose.yml up -d 
```

## Code Style and Quality
Code style follows PEP8 standards with `Ruff` formatting.
Use the following commands in the project direcotry to check the format.
```bash
ruff format
ruff check
```