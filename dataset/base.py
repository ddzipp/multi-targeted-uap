from typing import TypedDict

import torch


class VisionData(TypedDict):
    image: torch.Tensor
    label: str
    question: str
    answer: str
