import torch

from attacks.split import SplitAttacker, SplitConstraint
from utils.constraint import Constraint


class UnionSplitConstraint(SplitConstraint, Constraint):
    """
    Append the whole perturbation to the image, while calculating the spliting mask.
    """

    split_mask: torch.Tensor

    def get_mask(self, image):
        _mask = Constraint.get_mask(self, image)
        self.split_mask = SplitConstraint.split_mask(self, _mask)
        return _mask


class UnionSplitAttacker(SplitAttacker):
    """
    Update the perturbation with the split_mask
    """

    constraint: UnionSplitConstraint

    def step(self, grad):
        # split the gradient
        grad = self.constraint.split_mask * grad
        return super().step(grad)
