import torch
from torchvision import transforms

from dataset.base import AttackDataset
from dataset.cropa import CropaDataset
from dataset.imagenet import ImageNetDataset, ImageNetTestDataset
from dataset.muap import MUAP
from dataset.vqa import VQADataset


def load_dataset(name, *, transform=None, targets=None, path=None, split="val") -> AttackDataset:
    if path is None:
        path = f"./data/{name}"
    # Preprocess the image to 299x299
    if transform is None:
        transform = transforms.Compose([transforms.Resize((299, 299)), transforms.ToTensor()])
    if name.lower() == "vqa":
        dataset = VQADataset(path=path, split=split, transform=transform)
    elif name.lower() == "imagenet" and split == "test":
        dataset = ImageNetTestDataset(path=path, split="test", transform=transform)
    elif name.lower() == "imagenet":
        dataset = ImageNetDataset(path=path, split=split, transform=transform)
    elif name.lower() == "cropa":
        dataset = CropaDataset(path=path, split=split, transform=transform)
    elif name.lower() == "muap":
        dataset = MUAP(path=path, transform=transform)
    else:
        raise ValueError(f"Unknown dataset name {name}")
    return dataset


def collate_fn(batch):
    lable_ids = torch.cat([item["label_ids"] for item in batch])
    inputs_list = [item["inputs"] for item in batch]
    inputs = {}
    for k in inputs_list[0].keys():
        inputs[k] = torch.cat([item[k] for item in inputs_list])

    return {"inputs": inputs, "label_ids": lable_ids}
