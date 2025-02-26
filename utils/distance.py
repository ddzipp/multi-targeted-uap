from abc import ABC, abstractmethod
from functools import partial
from typing import TypeVar

import torch


T = TypeVar("T", bound=torch.Tensor)


def atleast_kd(x: T, k: int) -> T:
    shape = x.shape + (1,) * (k - x.ndim)
    return x.reshape(shape)


def flatten(x: T, start_dim: int = 1) -> T:
    return x.flatten(start_dim=start_dim)


class Distance(ABC):
    @abstractmethod
    def __call__(self, reference: T, perturbed: T) -> T: ...

    @abstractmethod
    def clip_perturbation(self, references: T, perturbed: T, epsilon: float) -> T: ...

    @abstractmethod
    def normalize(self, x: T) -> T: ...


class LpDistance(Distance):
    """:attr:`ord` defines the vector norm that is computed. The following norms are supported:
    ======================   ===============================
    :attr:`ord`              vector norm
    ======================   ===============================
    `2` (default)            `2`-norm (see below)
    `inf`                    `max(abs(x))`
    `-inf`                   `min(abs(x))`
    `0`                      `sum(x != 0)`
    other `int` or `float`   `sum(abs(x)^{ord})^{(1 / ord)}`
    ======================   ===============================
    """

    def __init__(self, p: float) -> None:
        self.p = p
        self.norm = partial(torch.linalg.vector_norm, ord=self.p)

    def __repr__(self) -> str:
        return f"LpDistance({self.p})"

    def __str__(self) -> str:
        return f"L{self.p} distance"

    def __call__(self, references: T, perturbed: T) -> T:
        """Calculates the distances from references to perturbed using the Lp norm.

        Args:
            references: A batch of reference inputs.
            perturbed: A batch of perturbed inputs.

        Returns:
            A 1D tensor with the distances from references to perturbed.
        """
        norms = torch.linalg.vector_norm(
            flatten(perturbed - references),
            self.p,
            dim=-1,
        )
        return norms

    def clip_perturbation(
        self, perturbed: T, *, epsilon: float, references: T = None
    ) -> T:
        """Clips the perturbations to epsilon and returns the new perturbed

        Args:
            references: A batch of reference inputs.
            perturbed: A batch of perturbed inputs.

        Returns:
            A tenosr like perturbed but with the perturbation clipped to epsilon.
        """
        if references is None:
            references = torch.zeros_like(perturbed)
        p = perturbed - references

        if self.p == torch.inf:
            if isinstance(epsilon, float):
                clipped_perturbation = torch.clip(p, -epsilon, epsilon)
                return references + clipped_perturbation
            else:
                assert (
                    isinstance(epsilon, torch.Tensor) and epsilon.shape[0] == p.shape[0]
                )
                _eps = atleast_kd(epsilon, p.ndim)
                _eps = _eps.repeat(1, *p.shape[1:])
                p[p > _eps] = _eps[p > _eps]
                p[p < -_eps] = _eps[p < -_eps]
                clipped_perturbation = p
                return references + clipped_perturbation

        norms = torch.linalg.vector_norm(flatten(p), self.p, dim=-1)
        norms = torch.maximum(norms, torch.tensor(1e-12))  # avoid divsion by zero
        factor = epsilon / norms
        factor = torch.minimum(
            torch.tensor(1), factor
        )  # clipping -> decreasing but not increasing
        if self.p == 0:
            if (factor == 1).all():
                return perturbed
            raise NotImplementedError("reducing L0 norms not yet supported")
        factor = factor.reshape(factor.shape + (1,) * (references.ndim - factor.ndim))
        clipped_perturbation = factor * p
        return references + clipped_perturbation

    def normalize(self, x: T) -> T:
        if self.p == torch.inf:
            return x.sign()
        else:
            norms = torch.linalg.vector_norm(flatten(x), ord=self.p, dim=-1)
            norms = torch.maximum(norms, torch.tensor(1e-12, device=norms.device))
            factor = 1 / norms
            factor = atleast_kd(factor, x.ndim)
            return x * factor


def get_distance(constraint: str = "l2") -> LpDistance:
    match constraint:
        case "l0":
            return LpDistance(0)
        case "l1":
            return LpDistance(1)
        case "l2":
            return LpDistance(2)
        case _:
            return LpDistance(torch.inf)
