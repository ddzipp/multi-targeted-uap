import torch

from attacks.base import Attacker
from utils.constraint import Constraint


class SplitConstraint(Constraint):
    def __init__(self, mode: str = "frame", *, num_targets: int = 2, **kwargs) -> None:
        super().__init__(mode, **kwargs)
        self.num_targets = num_targets
        self.target_dict = {"-1": -1}
        self.target: torch.Tensor = 0

    def split_mask(self, _mask):
        """
        Splits the input mask according to the targets specified in self.target.
        Args:
            _mask (torch.Tensor): The input mask tensor to be split. It is expected to have a shape
                                  where the last dimension corresponds to the width that will be split.
        Returns:
            torch.Tensor: The modified mask tensor after applying the split based on the targets.
        """

        target_id = []
        for i, t in enumerate(self.target):
            if str(t) not in self.target_dict:
                self.target_dict[str(t)] = max(self.target_dict.values()) + 1
            target_id.append(self.target_dict[str(t)])
        target_id = torch.tensor(target_id).to(_mask.device)
        split_width = _mask.shape[-1] // self.num_targets
        # calc the left and right bound
        lb = split_width * target_id
        rb = split_width * (target_id + 1)
        batch_idx = torch.arange(_mask.shape[0])[:, None]
        width_idx = torch.arange(_mask.shape[-1])[None, :]
        condition = (width_idx >= lb[:, None]) & (width_idx < rb[:, None])
        _mask[batch_idx, ..., width_idx] *= condition[..., None, None].to(_mask.device)
        return _mask

    def get_mask(self, image):
        _mask = super().get_mask(image)
        _mask = self.split_mask(_mask)
        return _mask


class SplitAttacker(Attacker):
    constraint: SplitConstraint

    def get_adv_inputs(self, image, target: list, question, label=None, answer=None, generation=False) -> torch.Tensor:
        # add split mask to pixel_values
        self.constraint.target = target
        inputs = super().get_adv_inputs(image, target, question, label, answer, generation)
        return inputs
