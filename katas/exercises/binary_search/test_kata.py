"""Tests for binary search kata."""



try:
    from user_kata import binary_search
    from user_kata import binary_search_leftmost
    from user_kata import binary_search_rightmost
    from user_kata import binary_search_insert_position
except ImportError:
    from .reference import binary_search
    from .reference import binary_search_leftmost
    from .reference import binary_search_rightmost
    from .reference import binary_search_insert_position


def test_binary_search_found():

    arr = [1, 3, 5, 7, 9, 11]
    assert binary_search(arr, 7) == 3
    assert binary_search(arr, 1) == 0
    assert binary_search(arr, 11) == 5


def test_binary_search_not_found():

    arr = [1, 3, 5, 7, 9]
    assert binary_search(arr, 4) == -1
    assert binary_search(arr, 0) == -1
    assert binary_search(arr, 10) == -1


def test_binary_search_empty():

    assert binary_search([], 5) == -1


def test_binary_search_leftmost():

    arr = [1, 2, 2, 2, 3, 4, 4, 5]
    assert binary_search_leftmost(arr, 2) == 1
    assert binary_search_leftmost(arr, 4) == 5
    assert binary_search_leftmost(arr, 1) == 0


def test_binary_search_rightmost():

    arr = [1, 2, 2, 2, 3, 4, 4, 5]
    assert binary_search_rightmost(arr, 2) == 3
    assert binary_search_rightmost(arr, 4) == 6
    assert binary_search_rightmost(arr, 5) == 7


def test_binary_search_insert_position():

    arr = [1, 3, 5, 7, 9]
    assert binary_search_insert_position(arr, 4) == 2
    assert binary_search_insert_position(arr, 0) == 0
    assert binary_search_insert_position(arr, 10) == 5
    assert binary_search_insert_position(arr, 5) == 2


def test_binary_search_single_element():

    assert binary_search([5], 5) == 0
    assert binary_search([5], 3) == -1
