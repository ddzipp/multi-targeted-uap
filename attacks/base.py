import os
import warnings

import torch

from config.config import Config
from models.base import Model
from utils.constraint import Constraint
from utils.optimizer import Optimizer


class Attacker:

    def __init__(
        self, model: Model, constraint: Constraint, lr=0.1, on_normalized=True
    ):
        super().__init__()
        self.model = model
        self.momentum = torch.tensor(0)
        self.constraint = constraint
        self.pert = torch.rand([3, 299, 299])
        self.lr = lr
        self.on_normalized = on_normalized
        # self.optimizer = Optimizer([perturbation], method="adam", lr=1.0)

    def get_inputs(
        self,
        image: torch.Tensor,
        target: list,
        question: list,
        label=None,
        answer=None,
        generation=False,
    ) -> torch.Tensor:
        # add perturbation to pixel_values
        perturbed_image = (
            image if self.on_normalized else self.constraint(image, self.pert)
        )

        inputs, target = self.model.generate_inputs(
            perturbed_image,
            targets=target,
            questions=question,
            generation=generation,
        )
        # add perturbation to pixel_values
        if self.on_normalized:
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
            inputs["pixel_values"], normalized=self.on_normalized
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
            self.momentum = self.lr * self.momentum + grad / torch.norm(grad, p=1)
            self.pert = self.pert - self.lr * self.momentum.sign()
            self.pert = self.model.clip_image(self.pert, normalized=self.on_normalized)
            total_loss += loss.item()
        return total_loss / len(dataloader)

    @torch.no_grad()
    def tester(self, dataloader):
        asr = 0
        for item in dataloader:
            inputs, target = self.get_inputs(**item, generation=True)
            image, target = inputs["pixel_values"].cuda(), target.cuda()
            # label = torch.tensor([int(label) for label in item["label"]], device="cuda")
            logits = self.model.model(image)
            pred = logits.argmax(-1)
            asr += (pred == target).sum().item()
        return asr / len(dataloader)

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
