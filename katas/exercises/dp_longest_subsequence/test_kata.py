"""Tests for LCS kata."""

try:
    from user_kata import lcs_length
except ModuleNotFoundError:
    import importlib.util
    from pathlib import Path

    module_path = Path(__file__).with_name("reference.py")
    spec = importlib.util.spec_from_file_location("reference", module_path)
    reference = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(reference)
    lcs_length = reference.lcs_length  # type: ignore


def test_empty_strings():
    """Empty strings should have LCS of 0."""
    assert lcs_length("", "") == 0
    assert lcs_length("abc", "") == 0
    assert lcs_length("", "xyz") == 0


def test_identical_strings():
    """Identical strings should have LCS equal to their length."""
    assert lcs_length("a", "a") == 1
    assert lcs_length("abc", "abc") == 3
    assert lcs_length("hello", "hello") == 5


def test_no_common_subsequence():
    """Strings with no common characters should have LCS of 0."""
    assert lcs_length("abc", "xyz") == 0
    assert lcs_length("123", "456") == 0


def test_simple_cases():
    """Test simple LCS cases."""
    assert lcs_length("ABCD", "ACDF") == 3  # ACD
    assert lcs_length("AGGTAB", "GXTXAYB") == 4  # GTAB
    assert lcs_length("ABC", "AC") == 2  # AC
    assert lcs_length("ABC", "BC") == 2  # BC


def test_longer_strings():
    """Test longer strings."""
    s1 = "ABCBDAB"
    s2 = "BDCABA"
    assert lcs_length(s1, s2) == 4  # BCBA or BDAB


def test_repeated_characters():
    """Test strings with repeated characters."""
    assert lcs_length("AAA", "AA") == 2
    assert lcs_length("AAAA", "AA") == 2
    assert lcs_length("ABABAB", "BABABA") == 5


def test_order_matters():
    """LCS should be same regardless of argument order."""
    s1 = "ABCDEF"
    s2 = "ACDF"
    assert lcs_length(s1, s2) == lcs_length(s2, s1)
