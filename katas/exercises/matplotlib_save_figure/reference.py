"""Reference solution for saving figures."""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def save_plot_to_file(
    x: np.ndarray,
    y: np.ndarray,
    filepath: str | Path,
    dpi: int = 100,
    title: str = ""
) -> None:
    """Create a line plot and save to file."""
    fig, ax = plt.subplots()
    ax.plot(x, y)
    if title:
        ax.set_title(title)
    plt.savefig(filepath, dpi=dpi)
    plt.close(fig)
