import os

import torch

from config.config import Config
from models.base import Model
from utils.constraint import Constraint
from utils.optimizer import Optimizer


class Attacker:

    def __init__(self, model: Model, constraint: Constraint, cfg: Config):
        super().__init__()
        self.model = model
        self.momentum = torch.tensor(0)
        self.constraint = constraint
        self.cfg = cfg
        self.pert = torch.rand([3, 299, 299])
        # self.optimizer = Optimizer([perturbation], method="adam", lr=1.0)

    def step(self, image, target, question, label, answer) -> torch.Tensor:
        self.pert = self.pert.detach().requires_grad_()
        if not self.cfg.on_normalized:
            perturbed_image = self.constraint(image, self.pert)
            inputs, target = self.model.generate_inputs(
                perturbed_image, targets=target, questions=question
            )
        else:
            inputs, target = self.model.generate_inputs(
                image, targets=target, questions=question
            )
            if self.pert.shape[-2:] != inputs["pixel_values"].shape[-2:]:
                self.pert = torch.rand_like(
                    inputs["pixel_values"][0], requires_grad=True
                )
            inputs["pixel_values"] = self.constraint(inputs["pixel_values"], self.pert)
        loss = self.model.calc_loss(inputs, target)
        loss.backward()
        grad = self.pert.grad
        self.momentum = self.cfg.lr * self.momentum + grad / torch.norm(grad, p=1)
        self.pert = self.pert - self.cfg.lr * self.momentum.sign()
        self.pert = self.model.clip_image(self.pert, normalized=self.cfg.on_normalized)
        # optimizer.step()
        return loss

    def trainer(self, dataloader) -> float:
        total_loss = 0
        for item in dataloader:
            loss = self.step(**item)
            total_loss += loss.item()
        return total_loss / len(dataloader)

    def saver(self, filename="./save/perturbation.pth"):
        dirpath = os.path.dirname(filename)
        os.makedirs(dirpath, exist_ok=True)
        torch.save(
            {
                "perturbation": self.pert.detach(),
                "mask": self.constraint.mask,
            },
            filename,
        )
