# Research Code for Multi-Targeted Universal Adversarial Perturbations

## Preliminary

### Docker (suggested)
docker pull huggingface/transformers-pytorch-gpu:4.41.3

### Conda/pip
Refer to requirements.txt for the required packages.

# Code Style
Code style follows PEP8 standards with Black formatting, Mypy type checks, and isort sorted imports.
Use the following commands in the project direcotry to check the format.
```bash
black --check --verbose .
mypy --install-types --non-interactive .
mypy --ignore-missing-imports .
pylint --disable="invalid-name,missing-module-docstring,W0612,W0631,W0703,W0621,W0613,W0611,W1308,C0411,C0111,C0103,C0301,C0304,C0305,E1101,R0913,R0914,R0915,R0903,R0902" .
```