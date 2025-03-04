import os
import warnings

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

    def get_inputs(
        self, image, target, question, label, answer, generation=False
    ) -> torch.Tensor:
        # add perturbation to pixel_values
        perturbed_image = (
            image if self.cfg.on_normalized else self.constraint(image, self.pert)
        )

        inputs, target = self.model.generate_inputs(
            perturbed_image,
            targets=target,
            questions=question,
            generation=generation,
        )
        # add perturbation to pixel_values
        if self.cfg.on_normalized:
            if self.pert.shape[-2:] != inputs["pixel_values"].shape[-2:]:
                warnings.warn(
                    "The shape of perturbation is not equal to the shape of image, "
                    "Re-init the perturbation."
                )
                self.pert = torch.rand_like(
                    inputs["pixel_values"][0], requires_grad=True
                )
            inputs["pixel_values"] = self.constraint(inputs["pixel_values"], self.pert)
        inputs["pixel_values"] = self.model.clip_image(
            inputs["pixel_values"], normalized=self.cfg.on_normalized
        )
        return inputs, target

    def trainer(self, dataloader) -> float:
        total_loss = 0
        for item in dataloader:
            self.pert = self.pert.detach().requires_grad_()
            inputs, target = self.get_inputs(**item)
            loss = self.model.calc_loss(inputs, target)
            loss.backward()
            # optimizer.step()
            grad = self.pert.grad
            self.momentum = self.cfg.lr * self.momentum + grad / torch.norm(grad, p=1)
            self.pert = self.pert - self.cfg.lr * self.momentum.sign()
            self.pert = self.model.clip_image(
                self.pert, normalized=self.cfg.on_normalized
            )
            total_loss += loss.item()
        return total_loss / len(dataloader)

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
