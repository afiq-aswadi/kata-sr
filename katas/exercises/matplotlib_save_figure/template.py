"""Save figures to file."""

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
    """Create a line plot and save to file.

    Args:
        x: X-coordinates
        y: Y-coordinates
        filepath: Path to save file (extension determines format)
        dpi: Resolution for raster formats (PNG, JPG)
        title: Optional plot title
    """
    # BLANK_START
    raise NotImplementedError("Create plot, use plt.savefig(filepath, dpi=dpi)")
    # BLANK_END
