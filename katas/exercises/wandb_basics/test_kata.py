"""Tests for W&B basics kata."""

import pytest
import torch
import torch.nn as nn
import wandb


@pytest.fixture(autouse=True)
def wandb_mode():
    """Set W&B to disabled mode for testing."""
    import os

    os.environ["WANDB_MODE"] = "disabled"
    yield
    if wandb.run is not None:
        wandb.finish()


def test_initialize_wandb_run():
    from template import initialize_wandb_run

    config = {"learning_rate": 0.01, "batch_size": 32}
    run = initialize_wandb_run("test-project", config)

    assert run is not None
    assert run.config["learning_rate"] == 0.01
    assert run.config["batch_size"] == 32

    wandb.finish()


def test_log_training_metrics():
    from template import initialize_wandb_run, log_training_metrics

    run = initialize_wandb_run("test-project", {})
    log_training_metrics(epoch=1, loss=0.5, accuracy=0.9)

    # Should not raise any errors
    wandb.finish()


def test_log_model_gradients():
    from template import initialize_wandb_run, log_model_gradients

    run = initialize_wandb_run("test-project", {})

    model = nn.Linear(10, 5)
    x = torch.randn(4, 10)
    y = torch.randn(4, 5)

    output = model(x)
    loss = ((output - y) ** 2).mean()
    loss.backward()

    log_model_gradients(model)

    # Should not raise any errors
    wandb.finish()


def test_save_model_artifact():
    from template import initialize_wandb_run, save_model_artifact

    run = initialize_wandb_run("test-project", {})

    model = nn.Linear(10, 5)
    save_model_artifact(model, "test-model")

    # Should not raise any errors
    wandb.finish()


def test_create_custom_chart():
    from template import create_custom_chart, initialize_wandb_run

    run = initialize_wandb_run("test-project", {})

    data = {"metric1": [1, 2, 3, 4], "metric2": [4, 3, 2, 1]}

    create_custom_chart(data)

    # Should not raise any errors
    wandb.finish()


def test_track_hyperparameter_sweep():
    from template import track_hyperparameter_sweep

    param_grid = {"lr": [0.01, 0.001], "batch_size": [16, 32]}

    def mock_train(config):
        # Simple mock training function
        return config["lr"] * config["batch_size"]

    track_hyperparameter_sweep(param_grid, mock_train)

    # Should not raise any errors
    if wandb.run is not None:
        wandb.finish()


def test_log_training_metrics_multiple_epochs():
    from template import initialize_wandb_run, log_training_metrics

    run = initialize_wandb_run("test-project", {})

    for epoch in range(5):
        log_training_metrics(epoch=epoch, loss=1.0 / (epoch + 1), accuracy=0.5 + epoch * 0.1)

    # Should handle multiple logs
    wandb.finish()
