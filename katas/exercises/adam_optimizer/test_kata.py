"""Tests for Adam optimizer kata."""

import torch


try:
    from user_kata import sgd_step
    from user_kata import sgd_momentum_step
    from user_kata import AdamOptimizer
except ImportError:
    from .reference import sgd_step
    from .reference import sgd_momentum_step
    from .reference import AdamOptimizer


def test_sgd_step():

    param = torch.tensor([1.0, 2.0, 3.0])
    grad = torch.tensor([0.1, 0.2, 0.3])
    lr = 0.1

    result = sgd_step(param, grad, lr)
    expected = torch.tensor([0.99, 1.98, 2.97])

    assert torch.allclose(result, expected)


def test_sgd_momentum_step():

    param = torch.tensor([1.0, 2.0])
    grad = torch.tensor([0.1, 0.2])
    velocity = torch.tensor([0.0, 0.0])

    new_param, new_velocity = sgd_momentum_step(param, grad, velocity, lr=0.1, momentum=0.9)

    assert torch.allclose(new_velocity, grad)
    assert torch.allclose(new_param, param - 0.1 * grad)


def test_adam_initialization():

    params = [torch.randn(3, 3), torch.randn(5)]
    optimizer = AdamOptimizer(params)

    assert len(optimizer.m) == len(params)
    assert len(optimizer.v) == len(params)
    assert optimizer.t == 0


def test_adam_single_step():

    param = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    optimizer = AdamOptimizer([param])

    # Simulate gradient
    param.grad = torch.tensor([0.1, 0.2, 0.3])

    original = param.clone()
    optimizer.step()

    # Parameter should have moved
    assert not torch.allclose(param, original)
    assert param[0] < original[0]  # gradient descent direction


def test_adam_multiple_steps():

    param = torch.tensor([10.0], requires_grad=True)
    optimizer = AdamOptimizer([param], lr=0.1)

    # Minimize f(x) = (x - 5)^2
    for _ in range(50):
        optimizer.zero_grad()
        loss = (param - 5.0) ** 2
        loss.backward()
        optimizer.step()

    # Should converge close to 5
    assert torch.allclose(param, torch.tensor([5.0]), atol=0.5)


def test_adam_vs_pytorch():

    # Test that our implementation is close to PyTorch's
    torch.manual_seed(42)
    param1 = torch.randn(10, 10, requires_grad=True)
    param2 = param1.clone().detach().requires_grad_(True)

    our_adam = AdamOptimizer([param1], lr=0.01)
    pytorch_adam = torch.optim.Adam([param2], lr=0.01)

    for _ in range(5):
        # Our implementation
        our_adam.zero_grad()
        loss1 = param1.sum()
        loss1.backward()
        our_adam.step()

        # PyTorch implementation
        pytorch_adam.zero_grad()
        loss2 = param2.sum()
        loss2.backward()
        pytorch_adam.step()

    # Should be very close
    assert torch.allclose(param1, param2, atol=1e-5)
