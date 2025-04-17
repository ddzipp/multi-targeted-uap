from typing import TypedDict

import torch
from tqdm import tqdm


class VisionData(TypedDict):
    image: torch.Tensor
    label: str | int
    question: str
    answer: str


class AttackDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, targets: dict | None = None, processor=None, eval=False):
        self.dataset = dataset
        self.target_dict = targets if targets is not None else {}
        self.processor = processor
        self.eval = eval
        # calc fixed target tokens for target_dict, which map original label to target label
        if self.processor is not None and not self.eval:
            self.tokenizer = self.processor.tokenizer
            self.eos_token_ids = self.tokenizer.eos_token_ids
            for original, target in self.target_dict.items():
                target_ids = self.tokenizer(target, add_special_tokens=False).input_ids
                self.target_dict[original] = target_ids + [self.eos_token_ids]
            # pad target_dict values to max length
            max_len = max([len(v) for v in self.target_dict.values()])
            for original, target in self.target_dict.items():
                self.target_dict[original] = target + [self.eos_token_ids] * (max_len - len(target))

        # preload attack dataset to memory and calculate input_ids and label_ids
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False)
        item = next(iter(dataloader))
        self.images, self.labels = item["image"], item["label"]
        self.targets = [self.target_dict[label] for label in item["label"]]
        self.inputs, self.label_ids = [], []
        if self.processor is not None:
            with tqdm(total=len(self.targets), desc="Preloading dataset to memory") as pbar:
                for i, q, t in zip(self.images, item["question"], self.targets):
                    inputs, label_ids = self.generate_inputs(i, q, t)
                    self.inputs.append(inputs)
                    self.label_ids.append(label_ids)
                    pbar.update()
        else:
            self.inputs, self.label_ids = self.images, self.targets

    def generate_inputs(self, images, questions, targets, generation=True):
        prompt = []
        questions = [questions] if isinstance(questions, str) else questions
        for q in questions:
            conv = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": q}, {"type": "image"}],
                },
            ]
            prompt.append(conv)
        conversation = self.processor.apply_chat_template(prompt, add_generation_prompt=generation)
        # the image has already rescaled to [0, 1]
        inputs = self.processor(images=images, text=conversation, return_tensors="pt", padding=True, do_rescale=False)
        if targets is None:
            return inputs, None
        targets = torch.tensor(targets)
        if targets.ndim == 1:
            targets = targets.unsqueeze(0)
        if not self.eval:
            inputs["input_ids"] = torch.cat([inputs["input_ids"], targets], dim=-1)
            inputs["attention_mask"] = torch.ones_like(inputs["input_ids"], dtype=torch.int64)
            label_ids = inputs["input_ids"].clone()
            label_ids[:, : -targets.shape[-1]] = -100
        else:
            label_ids = torch.full_like(inputs["input_ids"], -100)
        return inputs, label_ids

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return {
            "image": self.images[idx],
            "label": self.labels[idx],
            "target": self.targets[idx],
            "inputs": self.inputs[idx],
            "label_ids": self.label_ids[idx],
        }
