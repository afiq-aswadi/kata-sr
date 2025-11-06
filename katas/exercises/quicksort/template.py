"""Quicksort kata."""

from typing import List


def quicksort(arr: List[int]) -> List[int]:
    """Sort array using quicksort algorithm.

    Args:
        arr: unsorted array

    Returns:
        new sorted array (don't modify input)
    """
    # TODO: implement quicksort (can use partition helper)
    # BLANK_START
    pass
    # BLANK_END


def quicksort_inplace(arr: List[int], low: int = 0, high: int | None = None) -> None:
    """Sort array in-place using quicksort.

    Args:
        arr: array to sort (modified in-place)
        low: start index
        high: end index (inclusive)
    """
    if high is None:
        high = len(arr) - 1

    # TODO: implement in-place quicksort
    # BLANK_START
    pass
    # BLANK_END


def partition_lomuto(arr: List[int], low: int, high: int) -> int:
    """Partition array using Lomuto scheme.

    Choose arr[high] as pivot, partition so all elements < pivot are on left.

    Args:
        arr: array to partition (modified in-place)
        low: start index
        high: end index (pivot position)

    Returns:
        final pivot position
    """
    # TODO: implement Lomuto partition
    # BLANK_START
    pass
    # BLANK_END


def partition_hoare(arr: List[int], low: int, high: int) -> int:
    """Partition array using Hoare scheme.

    Use two pointers from both ends, swap elements to partition.

    Args:
        arr: array to partition (modified in-place)
        low: start index
        high: end index

    Returns:
        partition point (not necessarily pivot position)
    """
    # TODO: implement Hoare partition
    # BLANK_START
    pass
    # BLANK_END
