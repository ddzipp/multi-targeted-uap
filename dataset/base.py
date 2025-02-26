from typing import TypedDict
from torch.utils.data import Dataset
import torch
import torchvision


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


def load_dataset(name, path=None, split="val", transform=None):
    if path is None:
        path = f"./data/{name}"
    if transform is None:
        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((299, 299)),
                torchvision.transforms.ToTensor(),
            ]
        )
    if name.lower() == "vqa":
        from dataset.vqa import VQADataset

        return VQADataset(path=path, split=split, transform=transform)
    elif name.lower() == "imagenet":
        from dataset.imagenet import ImageNetDataset

        return ImageNetDataset(path=path, split=split, transform=transform)
    elif name.lower() == "cropa":
        from dataset.cropa import CropaDataset

        return CropaDataset(path=path, split=split, transform=transform)
    else:
        raise ValueError(f"Unknown dataset name {name}")
