"""Rotate array kata."""


def rotate(nums: list[int], k: int) -> None:
    """Rotate array to the right by k steps in-place.

    Use the reversal algorithm: reverse entire array, then reverse
    first k elements, then reverse remaining elements.

    Args:
        nums: array to rotate (modified in-place)
        k: number of positions to rotate right
    """
    # TODO: implement rotation using reversal algorithm
    # Handle k > len(nums) using modulo
    # BLANK_START
    pass
    # BLANK_END


def rotate_left(nums: list[int], k: int) -> None:
    """Rotate array to the left by k steps in-place.

    Args:
        nums: array to rotate (modified in-place)
        k: number of positions to rotate left
    """
    # TODO: rotate left is equivalent to rotate right by (n - k)
    # BLANK_START
    pass
    # BLANK_END
