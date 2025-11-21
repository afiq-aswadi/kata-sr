"""Create bar charts."""

import matplotlib.pyplot as plt
import numpy as np


def create_bar_chart(
    categories: list[str],
    values: np.ndarray,
    xlabel: str = "Category",
    ylabel: str = "Value"
) -> plt.Figure:
    """Create a vertical bar chart.

    Args:
        categories: List of category names
        values: Array of values for each category
        xlabel: Label for x-axis
        ylabel: Label for y-axis

    Returns:
        Figure object containing the bar chart
    """
    # BLANK_START
    raise NotImplementedError("Use plt.bar with categories and values, add labels")
    # BLANK_END
