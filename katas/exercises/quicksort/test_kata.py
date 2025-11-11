"""Tests for quicksort kata."""



try:
    from user_kata import quicksort
    from user_kata import quicksort_inplace
    from user_kata import partition_lomuto
    from user_kata import partition_hoare
except ImportError:
    from .reference import quicksort
    from .reference import quicksort_inplace
    from .reference import partition_lomuto
    from .reference import partition_hoare


def test_quicksort_basic():

    arr = [3, 6, 8, 10, 1, 2, 1]
    result = quicksort(arr)
    assert result == [1, 1, 2, 3, 6, 8, 10]


def test_quicksort_already_sorted():

    arr = [1, 2, 3, 4, 5]
    result = quicksort(arr)
    assert result == [1, 2, 3, 4, 5]


def test_quicksort_reverse_sorted():

    arr = [5, 4, 3, 2, 1]
    result = quicksort(arr)
    assert result == [1, 2, 3, 4, 5]


def test_quicksort_empty():

    assert quicksort([]) == []


def test_quicksort_single():

    assert quicksort([5]) == [5]


def test_quicksort_inplace():

    arr = [3, 6, 8, 10, 1, 2, 1]
    quicksort_inplace(arr)
    assert arr == [1, 1, 2, 3, 6, 8, 10]


def test_partition_lomuto():

    arr = [3, 6, 8, 10, 1, 2, 1]
    pivot_idx = partition_lomuto(arr, 0, len(arr) - 1)

    # All elements before pivot should be <= pivot
    # All elements after pivot should be > pivot
    pivot = arr[pivot_idx]
    assert all(arr[i] <= pivot for i in range(pivot_idx))
    assert all(arr[i] > pivot for i in range(pivot_idx + 1, len(arr)))


def test_partition_hoare():

    arr = [3, 6, 8, 10, 1, 2, 1]
    partition_idx = partition_hoare(arr, 0, len(arr) - 1)

    # Partition index should be valid
    assert 0 <= partition_idx < len(arr)
