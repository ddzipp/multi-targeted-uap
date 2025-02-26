from torch import optim


class Optimizer:
    def __init__(self, params, method="Momentum", lr=1e-2, targeted=True, **kwargs):
        maximize = not targeted

        if method.lower() == "sgd":
            optimizer = optim.SGD(params, lr=lr, maximize=maximize, **kwargs)
        elif method.lower() == "momentum":
            optimizer = optim.SGD(
                params, lr=lr, momentum=1, nesterov=False, maximize=maximize, **kwargs
            )
        elif method.lower() == "nesterov":
            optimizer = optim.SGD(
                params, lr=lr, momentum=1, nesterov=True, maximize=maximize, **kwargs
            )
        elif method.lower() == "adam":
            optimizer = optim.Adam(params, lr=lr, maximize=maximize, **kwargs)
        else:
            raise NotImplementedError

        self.optimizer = optimizer
        self._params = params
        for p in self._params:
            p.requires_grad_()

    def zero_grad(self, *args, **kwargs):
        self.optimizer.zero_grad(*args, **kwargs)
        for param in self._params:
            param.detach_()
            param.requires_grad_()

    def step(self, *args, **kwargs):
        self.optimizer.step(*args, **kwargs)
