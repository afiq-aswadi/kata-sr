"""Adam optimizer kata."""

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
        """Initialize Adam optimizer.

        Args:
            params: list of parameters to optimize
            lr: learning rate
            beta1: exponential decay rate for first moment
            beta2: exponential decay rate for second moment
            eps: small constant for numerical stability
        """
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        # TODO: initialize first and second moment estimates (m and v)
        # BLANK_START
        self.m = None
        self.v = None
        self.t = 0  # timestep
        # BLANK_END

    def step(self):
        """Perform a single optimization step."""
        # TODO: update parameters using Adam algorithm
        # 1. Update timestep
        # 2. For each parameter:
        #    - Update first moment (momentum)
        #    - Update second moment (RMSprop)
        #    - Compute bias-corrected estimates
        #    - Update parameter
        # BLANK_START
        pass
        # BLANK_END

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
    """Simple SGD update step.

    Args:
        param: parameter tensor
        grad: gradient tensor
        lr: learning rate

    Returns:
        updated parameter
    """
    # TODO: return param - lr * grad
    # BLANK_START
    pass
    # BLANK_END


def sgd_momentum_step(
    param: Float[torch.Tensor, "..."],
    grad: Float[torch.Tensor, "..."],
    velocity: Float[torch.Tensor, "..."],
    lr: float,
    momentum: float,
) -> tuple[Float[torch.Tensor, "..."], Float[torch.Tensor, "..."]]:
    """SGD with momentum update step.

    Args:
        param: parameter tensor
        grad: gradient tensor
        velocity: velocity (momentum) tensor
        lr: learning rate
        momentum: momentum coefficient

    Returns:
        (updated parameter, updated velocity)
    """
    # TODO: velocity = momentum * velocity + grad, param = param - lr * velocity
    # BLANK_START
    pass
    # BLANK_END
