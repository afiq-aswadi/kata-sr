"""Rotate array kata - reference solution."""


def rotate(nums: list[int], k: int) -> None:
    """Rotate array to the right by k steps in-place.

    Use the reversal algorithm: reverse entire array, then reverse
    first k elements, then reverse remaining elements.

    Args:
        nums: array to rotate (modified in-place)
        k: number of positions to rotate right
    """
    if not nums:
        return

    n = len(nums)
    k = k % n  # Handle k > n

    # Helper function to reverse portion of array
    def reverse(start: int, end: int) -> None:
        while start < end:
            nums[start], nums[end] = nums[end], nums[start]
            start += 1
            end -= 1

    # Reverse entire array
    reverse(0, n - 1)
    # Reverse first k elements
    reverse(0, k - 1)
    # Reverse remaining elements
    reverse(k, n - 1)


def rotate_left(nums: list[int], k: int) -> None:
    """Rotate array to the left by k steps in-place.

    Args:
        nums: array to rotate (modified in-place)
        k: number of positions to rotate left
    """
    if not nums:
        return

    n = len(nums)
    k = k % n
    # Rotate left by k is same as rotate right by (n - k)
    rotate(nums, n - k)
