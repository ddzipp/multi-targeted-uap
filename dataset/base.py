from typing import TypedDict

import torch


class VisionData(TypedDict):
    image: torch.Tensor
    label: str
    question: str
    answer: str


class AttackDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, target=None):
        self.dataset = dataset
        self.target = target

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        return {
            "image": item["image"],
            "label": item["label"],
            "question": item["question"],
            "answer": item["answer"],
            "target": self.target,
        }
