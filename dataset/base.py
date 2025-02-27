from typing import TypedDict
from torch.utils.data import Dataset
import torch
from torchvision import transforms


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


def load_dataset(name, transform=None, *, path=None, split="val"):
    if path is None:

        path = f"./data/{name}"
    # Preprocess the image to 299x299
    if transform is None:
        transform = transforms.Compose(
            [
                transforms.Resize((299, 299)),
                transforms.ToTensor(),
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
