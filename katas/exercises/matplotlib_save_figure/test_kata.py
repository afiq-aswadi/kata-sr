"""Tests for save figure kata."""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import tempfile
import os

try:
    from user_kata import save_plot_to_file
except ModuleNotFoundError:
    import importlib.util

    module_path = Path(__file__).with_name("reference.py")
    spec = importlib.util.spec_from_file_location("reference", module_path)
    reference = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(reference)
    save_plot_to_file = reference.save_plot_to_file  # type: ignore


def test_saves_png_file():
    """Should save PNG file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / "test.png"
        x = np.array([1, 2, 3])
        y = np.array([1, 4, 9])
        save_plot_to_file(x, y, filepath)
        assert filepath.exists()
        assert filepath.stat().st_size > 0


def test_saves_pdf_file():
    """Should save PDF file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / "test.pdf"
        x = np.array([1, 2, 3])
        y = np.array([1, 4, 9])
        save_plot_to_file(x, y, filepath)
        assert filepath.exists()
        assert filepath.stat().st_size > 0


def test_saves_svg_file():
    """Should save SVG file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / "test.svg"
        x = np.array([1, 2, 3])
        y = np.array([1, 4, 9])
        save_plot_to_file(x, y, filepath)
        assert filepath.exists()
        assert filepath.stat().st_size > 0


def test_with_title():
    """Should include title when provided."""
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / "titled.png"
        x = np.array([1, 2, 3])
        y = np.array([1, 4, 9])
        save_plot_to_file(x, y, filepath, title="Test Plot")
        assert filepath.exists()


def test_custom_dpi():
    """Should save with specified DPI."""
    with tempfile.TemporaryDirectory() as tmpdir:
        low_dpi_path = Path(tmpdir) / "low.png"
        high_dpi_path = Path(tmpdir) / "high.png"
        x = np.linspace(0, 10, 100)
        y = np.sin(x)

        save_plot_to_file(x, y, low_dpi_path, dpi=50)
        save_plot_to_file(x, y, high_dpi_path, dpi=300)

        assert low_dpi_path.exists()
        assert high_dpi_path.exists()
        # Higher DPI should produce larger file
        assert high_dpi_path.stat().st_size > low_dpi_path.stat().st_size


def test_string_filepath():
    """Should handle filepath as string."""
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "string_path.png")
        x = np.array([1, 2, 3])
        y = np.array([1, 2, 3])
        save_plot_to_file(x, y, filepath)
        assert os.path.exists(filepath)


def test_no_display_window():
    """Should not open display window (figure is closed)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / "test.png"
        x = np.array([1, 2, 3])
        y = np.array([1, 2, 3])
        before_count = len(plt.get_fignums())
        save_plot_to_file(x, y, filepath)
        after_count = len(plt.get_fignums())
        # No new figures should remain open
        assert after_count == before_count
