"""Tests for edit distance kata."""

try:
    from user_kata import edit_distance
except ModuleNotFoundError:
    import importlib.util
    from pathlib import Path

    module_path = Path(__file__).with_name("reference.py")
    spec = importlib.util.spec_from_file_location("reference", module_path)
    reference = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(reference)
    edit_distance = reference.edit_distance  # type: ignore


def test_identical_strings():
    """Identical strings should have distance 0."""
    assert edit_distance("", "") == 0
    assert edit_distance("a", "a") == 0
    assert edit_distance("abc", "abc") == 0
    assert edit_distance("hello", "hello") == 0


def test_empty_strings():
    """Distance from empty string is the length."""
    assert edit_distance("", "abc") == 3
    assert edit_distance("xyz", "") == 3


def test_single_character_difference():
    """Single character changes."""
    assert edit_distance("cat", "hat") == 1  # substitute c->h
    assert edit_distance("cat", "cats") == 1  # insert s
    assert edit_distance("cats", "cat") == 1  # delete s


def test_simple_cases():
    """Test simple edit distance cases."""
    assert edit_distance("kitten", "sitting") == 3  # k->s, e->i, insert g
    assert edit_distance("saturday", "sunday") == 3
    assert edit_distance("abc", "yabd") == 2  # insert y, c->d


def test_completely_different():
    """Completely different strings."""
    assert edit_distance("abc", "xyz") == 3
    assert edit_distance("1234", "abcd") == 4


def test_one_string_prefix_of_other():
    """One string is a prefix of the other."""
    assert edit_distance("test", "testing") == 3  # insert i, n, g
    assert edit_distance("pre", "prefix") == 3  # insert f, i, x


def test_symmetric():
    """Edit distance should be symmetric."""
    s1, s2 = "hello", "world"
    assert edit_distance(s1, s2) == edit_distance(s2, s1)


def test_repeated_characters():
    """Test with repeated characters."""
    assert edit_distance("aaa", "aaa") == 0
    assert edit_distance("aaa", "aaaa") == 1
    assert edit_distance("aaa", "bbb") == 3
