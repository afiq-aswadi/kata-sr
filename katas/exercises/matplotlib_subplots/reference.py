"""Reference solution for subplots."""

import matplotlib.pyplot as plt
import numpy as np


def create_subplot_grid(
    nrows: int,
    ncols: int,
    plot_data: list[dict]
) -> plt.Figure:
    """Create a grid of subplots with different plot types."""
    fig, axes = plt.subplots(nrows, ncols)

    # Handle single subplot case
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1 or ncols == 1:
        axes = axes.reshape(nrows, ncols)

    for plot in plot_data:
        row, col = plot['row'], plot['col']
        ax = axes[row, col]
        plot_type = plot['type']
        data = plot['data']

        if plot_type == 'scatter':
            ax.scatter(data['x'], data['y'])
        elif plot_type == 'line':
            ax.plot(data['x'], data['y'])
        elif plot_type == 'bar':
            ax.bar(data['categories'], data['values'])
        elif plot_type == 'hist':
            ax.hist(data['values'], bins=data.get('bins', 10))

    return fig
