"""Remove duplicates kata - reference solution."""


def remove_duplicates(nums: list[int]) -> int:
    """Remove duplicates from sorted array in-place.

    Modify nums in-place so first k elements contain unique values.
    Return k, the number of unique elements.

    Args:
        nums: sorted array (modified in-place)

    Returns:
        number of unique elements
    """
    if not nums:
        return 0

    write_idx = 1  # Position to write next unique element

    for read_idx in range(1, len(nums)):
        if nums[read_idx] != nums[read_idx - 1]:
            nums[write_idx] = nums[read_idx]
            write_idx += 1

    return write_idx


def remove_duplicates_allow_twice(nums: list[int]) -> int:
    """Remove duplicates allowing each element to appear at most twice.

    Args:
        nums: sorted array (modified in-place)

    Returns:
        new length after removing excess duplicates
    """
    if len(nums) <= 2:
        return len(nums)

    write_idx = 2  # First two elements are always kept

    for read_idx in range(2, len(nums)):
        # Check if current element is different from element 2 positions back
        if nums[read_idx] != nums[write_idx - 2]:
            nums[write_idx] = nums[read_idx]
            write_idx += 1

    return write_idx
