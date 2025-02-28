from typing import TypedDict

import torch


class VisualDict(TypedDict):
    image: torch.Tensor
    label: int
    question: str
    answer: str


class AttackDict(TypedDict):
    image: torch.Tensor
    label: int
    inputs_ori: torch.Tensor
    label_ids_ori: torch.Tensor
    inputs_trigger: torch.Tensor
    label_ids_trigger: torch.Tensor
