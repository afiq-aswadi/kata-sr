"""Tests for Find Median from Data Stream kata."""

try:
    from user_kata import MedianFinder
except ImportError:
    from .reference import MedianFinder


def test_median_finder_basic():

    mf = MedianFinder()
    mf.add_num(1)
    mf.add_num(2)
    assert mf.find_median() == 1.5
    mf.add_num(3)
    assert mf.find_median() == 2.0

def test_median_finder_single():

    mf = MedianFinder()
    mf.add_num(5)
    assert mf.find_median() == 5.0

def test_median_finder_unordered():

    mf = MedianFinder()
    mf.add_num(6)
    assert mf.find_median() == 6.0
    mf.add_num(10)
    assert mf.find_median() == 8.0
    mf.add_num(2)
    assert mf.find_median() == 6.0
    mf.add_num(6)
    assert mf.find_median() == 6.0
    mf.add_num(5)
    assert mf.find_median() == 6.0

def test_median_finder_negative():

    mf = MedianFinder()
    mf.add_num(-1)
    mf.add_num(-2)
    assert mf.find_median() == -1.5
    mf.add_num(-3)
    assert mf.find_median() == -2.0
