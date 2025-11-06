"""Weights & Biases basics kata - reference solution."""

import itertools
import os
import tempfile

import torch
import torch.nn as nn
import wandb


def initialize_wandb_run(project_name: str, config: dict) -> wandb.sdk.wandb_run.Run:
    """Initialize a W&B run with config."""
    return wandb.init(project=project_name, config=config, mode="disabled")  # disabled for tests


def log_training_metrics(epoch: int, loss: float, accuracy: float) -> None:
    """Log training metrics to W&B."""
    wandb.log({"epoch": epoch, "loss": loss, "accuracy": accuracy})


def log_model_gradients(model: nn.Module) -> None:
    """Log model gradient statistics to W&B."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2

    total_norm = total_norm**0.5
    wandb.log({"gradient_norm": total_norm})


def save_model_artifact(model: nn.Module, name: str) -> None:
    """Save model as W&B artifact."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as f:
        torch.save(model.state_dict(), f.name)
        temp_path = f.name

    try:
        artifact = wandb.Artifact(name, type="model")
        artifact.add_file(temp_path)
        wandb.log_artifact(artifact)
    finally:
        os.unlink(temp_path)


def create_custom_chart(data: dict[str, list[float]]) -> None:
    """Create custom W&B chart with multiple metrics."""
    # Log each data point
    max_len = max(len(v) for v in data.values())
    for i in range(max_len):
        log_dict = {}
        for key, values in data.items():
            if i < len(values):
                log_dict[key] = values[i]
        wandb.log(log_dict)


def track_hyperparameter_sweep(
    param_grid: dict[str, list],
    train_fn,
) -> None:
    """Run hyperparameter sweep and track results."""
    # Generate all combinations
    keys = list(param_grid.keys())
    values = list(param_grid.values())

    for combination in itertools.product(*values):
        config = dict(zip(keys, combination))

        # Start new run for this config
        with wandb.init(project="sweep", config=config, mode="disabled", reinit=True):
            result = train_fn(config)
            wandb.log({"final_metric": result})
