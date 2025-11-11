"""Tests for Detect Squares kata."""

try:
    from user_kata import DetectSquares
except ImportError:
    from .reference import DetectSquares


def test_detect_squares_example():
    ds = DetectSquares()
    ds.add([3, 10])
    ds.add([11, 2])
    ds.add([3, 2])
    assert ds.count([11, 10]) == 1
    assert ds.count([14, 8]) == 0
    ds.add([11, 2])
    assert ds.count([11, 10]) == 2

def test_detect_squares_no_square():
    ds = DetectSquares()
    ds.add([1, 1])
    ds.add([2, 2])
    assert ds.count([3, 3]) == 0

def test_detect_squares_single_point():
    ds = DetectSquares()
    ds.add([1, 1])
    assert ds.count([1, 1]) == 0

def test_detect_squares_multiple_squares():
    ds = DetectSquares()
    ds.add([0, 0])
    ds.add([0, 1])
    ds.add([1, 0])
    ds.add([1, 1])
    assert ds.count([0, 0]) == 1
