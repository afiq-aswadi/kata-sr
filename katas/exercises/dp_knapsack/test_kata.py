"""Tests for 0/1 knapsack kata."""

try:
    from user_kata import knapsack_01
except ModuleNotFoundError:
    import importlib.util
    from pathlib import Path

    module_path = Path(__file__).with_name("reference.py")
    spec = importlib.util.spec_from_file_location("reference", module_path)
    reference = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(reference)
    knapsack_01 = reference.knapsack_01  # type: ignore


def test_empty_items():
    """Empty items should give 0 value."""
    assert knapsack_01([], [], 10) == 0


def test_zero_capacity():
    """Zero capacity should give 0 value."""
    assert knapsack_01([1, 2, 3], [10, 20, 30], 0) == 0


def test_single_item_fits():
    """Single item that fits should be taken."""
    assert knapsack_01([5], [10], 10) == 10


def test_single_item_too_heavy():
    """Single item that doesn't fit should not be taken."""
    assert knapsack_01([15], [10], 10) == 0


def test_simple_case():
    """Test simple knapsack case."""
    weights = [1, 2, 3]
    values = [10, 15, 40]
    capacity = 5
    # Take items 1 and 2: weight=3, value=25
    # Or take item 3: weight=3, value=40
    # Or take items 0 and 3: weight=4, value=50
    assert knapsack_01(weights, values, capacity) == 50


def test_classic_example():
    """Test classic knapsack example."""
    weights = [10, 20, 30]
    values = [60, 100, 120]
    capacity = 50
    # Best: take items 1 and 2 (weight=50, value=220)
    assert knapsack_01(weights, values, capacity) == 220


def test_all_items_fit():
    """If all items fit, take all of them."""
    weights = [1, 2, 3]
    values = [10, 20, 30]
    capacity = 100
    assert knapsack_01(weights, values, capacity) == 60


def test_fractional_weights_not_allowed():
    """Can only take whole items (0 or 1)."""
    weights = [5, 5, 5]
    values = [10, 10, 10]
    capacity = 12
    # Can only take 2 items (weight=10, value=20)
    # Cannot take 2.4 items
    assert knapsack_01(weights, values, capacity) == 20


def test_duplicate_values():
    """Test with duplicate values."""
    weights = [2, 2, 2]
    values = [10, 10, 10]
    capacity = 5
    # Take 2 items (weight=4, value=20)
    assert knapsack_01(weights, values, capacity) == 20
