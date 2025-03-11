import warnings

import torch

from attacks.base import Attacker
from models.base import Model
from utils.constraint import Constraint


class SplitConstraint(Constraint):
    def __init__(
        self,
        mode: str = "frame",
        *,
        epsilon: float = 1.0,
        norm_type: str = "linf",
        frame_width: int = 6,
        patch_size: tuple = (40, 40),
        patch_location: tuple = (0, 0),
        bound: tuple = (0.0, 1.0),
        ref_size: int | None = None,
        num_targets: int = 2,
    ) -> None:
        super().__init__(
            mode=mode,
            epsilon=epsilon,
            norm_type=norm_type,
            frame_width=frame_width,
            patch_size=patch_size,
            patch_location=patch_location,
            bound=bound,
            ref_size=ref_size,
        )
        self.num_targets = num_targets

    def get_mask(self, image, target: torch.Tensor | int = 0):
        _mask = super().get_mask(image)
        # support the split-optimization by spliting the width of perturbation
        split_width = _mask.shape[-1] // self.num_targets
        # calc the left and right bound
        l = split_width * target
        r = split_width * (target + 1)
        batch_idx = torch.arange(image.shape[0])[:, None]
        width_idx = torch.arange(image.shape[-1])[None, :]
        condition = (width_idx >= l[:, None]) & (width_idx < r[:, None])
        _mask[batch_idx, ..., width_idx] *= condition[..., None, None].to(_mask.device)
        return _mask

    def apply_perturbation(self, image, perturbation, target=0):
        assert image.shape[-2:] == perturbation.shape[-2:]
        # Repeat the perturbation to the same shape of the image
        perturb = perturbation.expand_as(image).to(image.device)
        # Make a copy of the image to avoid modifying the original
        perturbed_image = image.clone()
        _mask = self.get_mask(image, target)
        # Apply the perturbation only to the frame area
        perturbed_image = perturbed_image * (1 - _mask) + perturb * _mask
        # Ensure the resulting image has valid pixel values (assuming 0-1 range)
        # perturbed_image = torch.clamp(perturbed_image, self.bound[0], self.bound[1])
        if self.mode == "pixel":
            perturbed_image = self.clip_perturbation(perturbed_image, image)
        return perturbed_image

    def __call__(self, image, perturbation, target=0):
        return self.apply_perturbation(image, perturbation, target)


class SplitAttacker(Attacker):

    def __init__(
        self, model: Model, constraint: SplitConstraint, lr=0.1, on_normalized=True
    ):
        super().__init__(model, constraint, lr, on_normalized)
        self.targets: dict = {"-1": -1}
        self.constraint: SplitConstraint
        self.pert: torch.Tensor

    def get_inputs(
        self, image, target: list, question, label=None, answer=None, generation=False
    ) -> torch.Tensor:
        # map target to id target
        for i, t in enumerate(target):
            if str(t) not in self.targets:
                self.targets[str(t)] = max(self.targets.values()) + 1
            target[i] = self.targets[str(t)]
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
            inputs["pixel_values"] = self.constraint(
                inputs["pixel_values"], self.pert, target
            )
        inputs["pixel_values"] = self.model.clip_image(
            inputs["pixel_values"], normalized=self.on_normalized
        )
        return inputs, target
