import torch
from torch import optim


class MomentumOptimizer(optim.Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.9):
        defaults = dict(lr=lr, momentum=momentum)
        super().__init__(params, defaults)
        for group in self.param_groups:
            for p in group["params"]:
                self.state[p]["velocity"] = torch.zeros_like(p.data)
                p.requires_grad = True

    def step(self, closure=None):
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                velocity = state["velocity"]
                grad = p.grad.data / p.grad.data.norm(p=1)
                # v = momentum * v + grad; p -= lr * v
                velocity.mul_(momentum).add_(grad)
                p.data.add_(velocity.sign(), alpha=-lr)
                p.detach_().requires_grad_()
