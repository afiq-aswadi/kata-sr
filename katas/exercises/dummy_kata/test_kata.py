"""Tests for dummy kata - all tests pass by design."""

try:
    from user_kata import dummy_function
except ModuleNotFoundError:  # pragma: no cover - fallback for standalone test runs
    import importlib.util
    from pathlib import Path

    module_path = Path(__file__).with_name("reference.py")
    spec = importlib.util.spec_from_file_location("dummy_reference", module_path)
    reference = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(reference)

    dummy_function = reference.dummy_function  # type: ignore[attr-defined]


def test_returns_true():
    """Test that dummy_function returns True."""
    result = dummy_function()
    assert result is True, "dummy_function should return True"


def test_is_callable():
    """Test that dummy_function is callable."""
    assert callable(dummy_function), "dummy_function should be callable"


def test_returns_bool():
    """Test that dummy_function returns a boolean."""
    result = dummy_function()
    assert isinstance(result, bool), "dummy_function should return a bool"


def test_consistent():
    """Test that dummy_function returns consistent results."""
    result1 = dummy_function()
    result2 = dummy_function()
    assert result1 == result2, "dummy_function should be consistent"


def test_always_passes():
    """This test always passes."""
    assert True, "This test always passes"
