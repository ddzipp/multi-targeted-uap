from typing import TypedDict

import torch


class VisionData(TypedDict):
    image: torch.Tensor
    label: str
    question: str
    answer: str


class AttackDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, targets: dict | None = None):
        self.dataset = dataset
        self.targets = targets

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        target = (
            self.targets
            if not isinstance(self.targets, dict)
            else self.targets.get(item["label"], None)
        )
        return {
            "image": item["image"],
            "label": item["label"],
            "question": item["question"],
            "answer": item["answer"],
            "target": target,
        }
