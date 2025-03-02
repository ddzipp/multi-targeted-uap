import torch
from torchvision import transforms

from dataset.base import AttackDataset
from dataset.cropa import CropaDataset
from dataset.imagenet import ImageNetDataset
from dataset.vqa import VQADataset


def load_dataset(
    name, *, transform=None, target=None, path=None, split="val"
) -> AttackDataset:
    if path is None:

        path = f"./data/{name}"
    # Preprocess the image to 299x299
    if transform is None:
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        )
    if name.lower() == "vqa":
        dataset = VQADataset(path=path, split=split, transform=transform)
    elif name.lower() == "imagenet":
        dataset = ImageNetDataset(path=path, split=split, transform=transform)
    elif name.lower() == "cropa":
        dataset = CropaDataset(path=path, split=split, transform=transform)
    else:
        raise ValueError(f"Unknown dataset name {name}")
    dataset = AttackDataset(dataset, target)
    return dataset


def collate_fn(batch):
    images = torch.stack([item["image"] for item in batch])
    labels = [item["label"] for item in batch]
    questions = [item["question"] for item in batch]
    answers = [item["answer"] for item in batch]
    targets = [item["target"] for item in batch]

    return {
        "image": images,
        "label": labels,
        "question": questions,
        "answer": answers,
        "target": targets,
    }
