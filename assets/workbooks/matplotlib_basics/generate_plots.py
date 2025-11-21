"""Generate reference plots for matplotlib basics workbook."""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set style for clean plots
plt.style.use('default')

output_dir = Path(__file__).parent / "images"
output_dir.mkdir(exist_ok=True)


def save_figure(filename: str):
    """Save current figure to output directory."""
    filepath = output_dir / filename
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


# 1. Scatter plot
fig, ax = plt.subplots(figsize=(6, 4))
x = np.linspace(0, 10, 50)
y = 2 * x + np.random.randn(50) * 2
ax.scatter(x, y, alpha=0.6)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Basic Scatter Plot')
save_figure('scatter_example.png')

# 2. Line plot with multiple series
fig, ax = plt.subplots(figsize=(6, 4))
x = np.linspace(0, 2 * np.pi, 100)
ax.plot(x, np.sin(x), '-', label='sin(x)')
ax.plot(x, np.cos(x), '--', label='cos(x)')
ax.plot(x, np.sin(x) * np.cos(x), ':', label='sin(x)cos(x)')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Multiple Line Styles')
ax.legend()
save_figure('line_example.png')

# 3. Bar chart
fig, ax = plt.subplots(figsize=(6, 4))
categories = ['Category A', 'Category B', 'Category C', 'Category D']
values = np.array([23, 45, 31, 38])
ax.bar(categories, values)
ax.set_xlabel('Category')
ax.set_ylabel('Value')
ax.set_title('Bar Chart Example')
save_figure('bar_example.png')

# 4. Histogram
fig, ax = plt.subplots(figsize=(6, 4))
data = np.random.normal(100, 15, 1000)
ax.hist(data, bins=20, alpha=0.7, edgecolor='black')
ax.set_xlabel('Value')
ax.set_ylabel('Frequency')
ax.set_title('Histogram with 20 Bins')
save_figure('histogram_example.png')

# 5. Labels and titles (comprehensive)
fig, ax = plt.subplots(figsize=(6, 4))
x = np.linspace(0, 10, 50)
ax.plot(x, x**2, label='Quadratic', marker='o', markersize=4)
ax.plot(x, x**1.5, label='Power 1.5', marker='s', markersize=4)
ax.set_title('Well-Labeled Plot', fontsize=14)
ax.set_xlabel('Time (seconds)', fontsize=12)
ax.set_ylabel('Distance (meters)', fontsize=12)
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)
save_figure('labels_titles_example.png')

# 6. Colors and markers
fig, ax = plt.subplots(figsize=(6, 4))
np.random.seed(42)
x = np.random.rand(30) * 10
y = np.random.rand(30) * 10
sizes = np.random.rand(30) * 200 + 50
ax.scatter(x, y, c='crimson', marker='^', s=sizes, alpha=0.6, edgecolors='black')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Styled Scatter Plot')
save_figure('colors_markers_example.png')

# 7. Error bars
fig, ax = plt.subplots(figsize=(6, 4))
x = np.array([1, 2, 3, 4, 5])
y = np.array([2.3, 4.1, 5.8, 7.9, 10.2])
yerr = np.array([0.5, 0.7, 0.6, 0.8, 0.9])
ax.errorbar(x, y, yerr=yerr, fmt='o-', capsize=5, capthick=2)
ax.set_xlabel('Measurement Number')
ax.set_ylabel('Value')
ax.set_title('Error Bar Plot')
ax.grid(True, alpha=0.3)
save_figure('error_bars_example.png')

# 8. Box plot
fig, ax = plt.subplots(figsize=(6, 4))
data1 = np.random.normal(100, 10, 200)
data2 = np.random.normal(90, 15, 200)
data3 = np.random.normal(105, 8, 200)
ax.boxplot([data1, data2, data3], labels=['Group A', 'Group B', 'Group C'])
ax.set_xlabel('Group')
ax.set_ylabel('Value')
ax.set_title('Box Plot Comparison')
save_figure('box_plot_example.png')

# 9. Heatmap
fig, ax = plt.subplots(figsize=(6, 5))
data = np.random.rand(10, 12)
im = ax.imshow(data, cmap='viridis', aspect='auto')
ax.set_xlabel('Column')
ax.set_ylabel('Row')
ax.set_title('Heatmap Example')
plt.colorbar(im, ax=ax, label='Value')
save_figure('heatmap_example.png')

# 10. Contour plot
fig, ax = plt.subplots(figsize=(6, 5))
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
Z = np.exp(-(X**2 + Y**2)) + 0.3 * np.exp(-((X-1.5)**2 + (Y-1.5)**2))
cs = ax.contourf(X, Y, Z, levels=15, cmap='plasma')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Contour Plot')
plt.colorbar(cs, ax=ax, label='Z value')
save_figure('contour_example.png')

# 11. Subplots
fig, axes = plt.subplots(2, 2, figsize=(8, 6))
x = np.linspace(0, 10, 50)

# Subplot 1: Scatter
axes[0, 0].scatter(x, x**2 + np.random.randn(50)*10, alpha=0.6)
axes[0, 0].set_title('Scatter')

# Subplot 2: Line
axes[0, 1].plot(x, np.sin(x), 'r-')
axes[0, 1].set_title('Line')

# Subplot 3: Bar
axes[1, 0].bar(['A', 'B', 'C'], [20, 35, 30])
axes[1, 0].set_title('Bar')

# Subplot 4: Histogram
axes[1, 1].hist(np.random.randn(200), bins=15, alpha=0.7)
axes[1, 1].set_title('Histogram')

plt.tight_layout()
save_figure('subplots_example.png')

# 12. Save figure (show a plot that would be saved)
fig, ax = plt.subplots(figsize=(6, 4))
x = np.linspace(0, 5, 100)
ax.plot(x, np.exp(-x) * np.sin(2*np.pi*x))
ax.set_xlabel('Time')
ax.set_ylabel('Amplitude')
ax.set_title('Damped Oscillation (saved at high DPI)')
ax.grid(True, alpha=0.3)
save_figure('save_figure_example.png')

print(f"\nAll plots saved to {output_dir}")
