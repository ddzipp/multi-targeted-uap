import os

import torch

from models.base import Model, TimmModel
from utils.constraint import Constraint


class Attacker:
    def __init__(
        self,
        model: Model,
        constraint: Constraint,
        lr=0.1,
        on_normalized=True,  # add noise on normalized pixel_values
        momentum=0.9,
        bound: tuple = (0, 1),
        ref_size=299,
    ):
        super().__init__()
        self.model = model
        self.constraint = constraint
        self.lr = lr
        self.momentum = momentum
        self.on_normalized = on_normalized
        self.bound = bound
        self.ref_shape = (1, 3, ref_size, ref_size)
        self.pert = self.__init_pert__()
        self.velocity = torch.zeros_like(self.pert)

    def __init_pert__(self):
        self.pert = torch.rand(self.ref_shape)
        mask = self.constraint.get_mask(self.ref_shape)
        if self.on_normalized:
            self.pert = self.model.image_preprocess(self.pert)
            if self.pert.ndim == 4:
                # generate mask directly with new shape
                self.constraint.get_mask(self.pert.shape)
            else:  # let the mask go through the same image_preprocess
                self.constraint._mask = self.model.image_preprocess(mask, False)
            # init clip bound for normalized pixel_values
            lower, upper = self.bound
            min_values = lower * torch.ones(self.ref_shape)
            max_values = upper * torch.ones(self.ref_shape)
            min_values = self.model.image_preprocess(min_values)
            max_values = self.model.image_preprocess(max_values)
            self.bound = (min_values, max_values)
        return self.pert

    def clip_image(self, image: torch.Tensor):
        batch = image.shape[0] // self.pert.shape[0]
        if not self.on_normalized:
            return torch.clip(image, *self.bound)
        else:
            lower, upper = self.bound
            lower = lower.repeat((batch,) + (1,) * (image.ndim - 1))
            upper = upper.repeat((batch,) + (1,) * (image.ndim - 1))
            return torch.clip(image, lower, upper)

    def get_adv_inputs(
        self,
        images: torch.Tensor,
        targets: torch.tensor,
        questions: list,
        labels=None,
        answers=None,
    ):
        # add perturbation to pixel_values
        if not self.on_normalized:
            images = self.constraint(images, self.pert)
        inputs, label_ids = self.model.generate_inputs(images, questions, targets=targets)

        # add perturbation to normalized pixel_values
        if self.on_normalized:
            inputs["pixel_values"] = self.constraint(inputs["pixel_values"], self.pert)
        inputs["pixel_values"] = self.clip_image(inputs["pixel_values"])
        return inputs, label_ids

    def step(self, grad):
        self.velocity = self.momentum * self.velocity + grad / torch.norm(grad, p=1)
        self.pert = self.pert - self.lr * self.velocity.sign()
        self.pert = self.clip_image(self.pert)

    def trainer(self, dataloader) -> float:
        total_loss = 0
        for item in dataloader:
            # self.optimizer.zero_grad()
            self.pert = self.pert.detach().requires_grad_()
            inputs, targets = self.get_adv_inputs(**item)
            loss = self.model.calc_loss(inputs, targets)
            loss.backward()
            # self.optimizer.step()
            self.step(self.pert.grad)
            total_loss += loss.item()
        return total_loss / len(dataloader)

    @torch.no_grad()
    def tester(self, dataloader):
        asr = 0
        for item in dataloader:
            inputs, targets = self.get_adv_inputs(**item, generation=True)
            logits = self.model.forward(inputs)
            # label = torch.tensor([int(label) for label in item["label"]], device="cuda")
            if isinstance(self.model, TimmModel):
                pred = logits.argmax(-1)
            else:
                pred = logits.argmax(-1)[:, -1]
                targets = self.model.processor.tokenizer(item["targets"], return_tensors="pt")["input_ids"][:, 0].to(
                    pred.device
                )
            asr += (pred == targets).sum().item()

        return asr / len(dataloader.dataset)

    def saver(self, filename="./save/perturbation.pth"):
        dirpath = os.path.dirname(filename)
        os.makedirs(dirpath, exist_ok=True)
        torch.save(
            {
                "perturbation": self.pert.detach(),
                # "mask": self.constraint.mask,
            },
            filename,
        )
