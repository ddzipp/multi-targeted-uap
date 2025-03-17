import os
import warnings

import torch
from torch import nn

from config.config import Config
from models.base import Model, TimmModel
from utils.constraint import Constraint

# from utils.optimizer import MomentumOptimizer


class Attacker:

    def __init__(
        self,
        model: Model,
        constraint: Constraint,
        lr=0.1,
        on_normalized=True,
        momentum=0.9,
        bound: tuple = (0, 1),
    ):
        super().__init__()
        self.model = model
        self.constraint = constraint
        self.pert = torch.rand([3, 299, 299])
        self.lr = lr
        self.velocity = torch.zeros_like(self.pert)
        self.momentum = momentum
        self.on_normalized = on_normalized
        self.bound = bound
        # self.optimizer = MomentumOptimizer([self.pert], lr=lr, momentum=0.9)

    def get_inputs(
        self,
        images: torch.Tensor,
        targets: list,
        questions: list,
        labels=None,
        answers=None,
        generation=False,
    ):
        # add perturbation to pixel_values
        mask = self.constraint.init_mask(images)
        if not self.on_normalized:
            images = self.constraint(images, self.pert)

        inputs, label_ids = self.model.generate_inputs(
            images,
            targets=targets,
            questions=questions,
            generation=generation,
        )

        # add perturbation to pixel_values
        if self.on_normalized:
            if self.constraint._mask.shape[-1] != inputs["pixel_values"].shape[-1]:
                self.constraint._mask = self.model.image_preprocess(mask).to(float)

            if self.pert.shape[-1] != inputs["pixel_values"].shape[-1]:
                warnings.warn(
                    "The shape of perturbation is not equal to the shape of image, "
                    "Re-init the perturbation."
                )
                self.pert = torch.rand_like(self.constraint._mask, requires_grad=True)
                self.velocity = torch.zeros_like(self.pert)

                # self.optimizer = MomentumOptimizer(
                #     [self.pert], lr=self.lr, momentum=self.momentum
                # )
            inputs["pixel_values"] = self.constraint(inputs["pixel_values"], self.pert)
        inputs["pixel_values"] = self.model.clip_image(
            inputs["pixel_values"], normalized=self.on_normalized, bound=(0, 1)
        )
        return inputs.to("cuda"), label_ids.to("cuda")

    def step(self, grad):
        self.velocity = self.momentum * self.velocity + grad / torch.norm(grad, p=1)
        self.pert = self.pert - self.lr * self.velocity.sign()
        self.pert = self.model.clip_image(
            self.pert, normalized=self.on_normalized, bound=self.bound
        )

    def trainer(self, dataloader) -> float:
        total_loss = 0
        for item in dataloader:
            # self.optimizer.zero_grad()
            self.pert = self.pert.detach().requires_grad_()
            inputs, targets = self.get_inputs(**item)
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
            inputs, targets = self.get_inputs(**item, generation=True)
            logits = self.model.forward(inputs)
            # label = torch.tensor([int(label) for label in item["label"]], device="cuda")
            if isinstance(self.model, TimmModel):
                pred = logits.argmax(-1)
            else:
                pred = logits.argmax(-1)[:, -1]
                targets = self.model.processor.tokenizer(
                    item["targets"], return_tensors="pt"
                )["input_ids"][:, 0].to(pred.device)
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
