from typing import TypedDict

import torch


class VisionData(TypedDict):
    image: torch.Tensor
    label: str | int
    question: str
    answer: str


class AttackDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, targets: dict | None = None, tokenizer=None):
        self.dataset = dataset
        self.targets = targets if targets is not None else {}
        self.tokenizer = tokenizer
        if self.tokenizer is not None:
            pad_token_ids = self.tokenizer.pad_token_ids
            eos_token_ids = self.tokenizer.eos_token_ids
            self.tokenizer.pad_token_ids = eos_token_ids
            values = list(self.targets.values())
            target_ids = self.tokenizer(values, padding=True, padding_side="right", add_special_tokens=False)
            target_ids = target_ids.input_ids
            for i, key in enumerate(self.targets.keys()):
                target_ids[i].append(eos_token_ids)
                self.targets[key] = target_ids[i]
            self.tokenizer.pad_token_ids = pad_token_ids

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        target: int | list[int] = self.targets.get(item["label"], None)
        return {
            "image": item["image"],
            "label": item["label"],
            "question": item["question"],
            "answer": item["answer"],
            "target": target,
        }
