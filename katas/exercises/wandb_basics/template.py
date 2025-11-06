"""Weights & Biases basics kata."""

import wandb
import torch
import torch.nn as nn


def initialize_wandb_run(project_name: str, config: dict) -> wandb.sdk.wandb_run.Run:
    """Initialize a W&B run with config.

    Args:
        project_name: name of W&B project
        config: hyperparameter config dict

    Returns:
        wandb run object
    """
    # TODO: initialize wandb run with project and config
    # Hint: wandb.init(project=..., config=...)
    # BLANK_START
    pass
    # BLANK_END


def log_training_metrics(epoch: int, loss: float, accuracy: float) -> None:
    """Log training metrics to W&B.

    Args:
        epoch: current epoch number
        loss: training loss
        accuracy: training accuracy
    """
    # TODO: log metrics with wandb.log()
    # BLANK_START
    pass
    # BLANK_END


def log_model_gradients(model: nn.Module) -> None:
    """Log model gradient statistics to W&B.

    Args:
        model: PyTorch model
    """
    # TODO: compute and log gradient norms
    # Hint: iterate over model.parameters(), compute grad norms
    # BLANK_START
    pass
    # BLANK_END


def save_model_artifact(model: nn.Module, name: str) -> None:
    """Save model as W&B artifact.

    Args:
        model: PyTorch model to save
        name: artifact name
    """
    # TODO:
    # 1. Save model to temporary file
    # 2. Create wandb.Artifact
    # 3. Add file to artifact
    # 4. Log artifact
    # BLANK_START
    pass
    # BLANK_END


def create_custom_chart(data: dict[str, list[float]]) -> None:
    """Create custom W&B chart with multiple metrics.

    Args:
        data: dictionary mapping metric names to value lists
    """
    # TODO: log data as custom plot
    # Hint: use wandb.log() with wandb.plot or custom data
    # BLANK_START
    pass
    # BLANK_END


def track_hyperparameter_sweep(
    param_grid: dict[str, list],
    train_fn,
) -> None:
    """Run hyperparameter sweep and track results.

    Args:
        param_grid: dict of parameter names to value lists
        train_fn: training function that takes config dict
    """
    # TODO: manually iterate over param combinations, track each run
    # BLANK_START
    pass
    # BLANK_END
