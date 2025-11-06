"""Adam optimizer kata - reference solution."""

import torch
from jaxtyping import Float


class AdamOptimizer:
    """Simple Adam optimizer implementation."""

    def __init__(
        self,
        params: list[torch.Tensor],
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        # Initialize moment estimates
        self.m = [torch.zeros_like(p) for p in params]
        self.v = [torch.zeros_like(p) for p in params]
        self.t = 0

    def step(self):
        """Perform a single optimization step."""
        self.t += 1

        for i, param in enumerate(self.params):
            if param.grad is None:
                continue

            grad = param.grad

            # Update biased first moment estimate
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad

            # Update biased second raw moment estimate
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grad**2

            # Compute bias-corrected estimates
            m_hat = self.m[i] / (1 - self.beta1**self.t)
            v_hat = self.v[i] / (1 - self.beta2**self.t)

            # Update parameters
            param.data -= self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        """Zero out gradients."""
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()


def sgd_step(
    param: Float[torch.Tensor, "..."],
    grad: Float[torch.Tensor, "..."],
    lr: float,
) -> Float[torch.Tensor, "..."]:
    """Simple SGD update step."""
    return param - lr * grad


def sgd_momentum_step(
    param: Float[torch.Tensor, "..."],
    grad: Float[torch.Tensor, "..."],
    velocity: Float[torch.Tensor, "..."],
    lr: float,
    momentum: float,
) -> tuple[Float[torch.Tensor, "..."], Float[torch.Tensor, "..."]]:
    """SGD with momentum update step."""
    new_velocity = momentum * velocity + grad
    new_param = param - lr * new_velocity
    return new_param, new_velocity
