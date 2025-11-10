"""Tests for Find Median from Data Stream kata."""

def test_median_finder_basic():
    from template import MedianFinder

    mf = MedianFinder()
    mf.add_num(1)
    mf.add_num(2)
    assert mf.find_median() == 1.5
    mf.add_num(3)
    assert mf.find_median() == 2.0

def test_median_finder_single():
    from template import MedianFinder

    mf = MedianFinder()
    mf.add_num(5)
    assert mf.find_median() == 5.0

def test_median_finder_unordered():
    from template import MedianFinder

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
    from template import MedianFinder

    mf = MedianFinder()
    mf.add_num(-1)
    mf.add_num(-2)
    assert mf.find_median() == -1.5
    mf.add_num(-3)
    assert mf.find_median() == -2.0
