"""Maximum subarray kata - reference solution."""


def max_subarray_sum(nums: list[int]) -> int:
    """Find maximum sum of contiguous subarray using Kadane's algorithm.

    Args:
        nums: array of integers (may contain negatives)

    Returns:
        maximum sum of any contiguous subarray
    """
    if not nums:
        return 0

    current_sum = max_sum = nums[0]

    for num in nums[1:]:
        current_sum = max(num, current_sum + num)
        max_sum = max(max_sum, current_sum)

    return max_sum


def max_subarray_indices(nums: list[int]) -> tuple[int, int]:
    """Find the indices [start, end] of the maximum sum subarray.

    Args:
        nums: array of integers

    Returns:
        tuple (start_index, end_index) of maximum subarray
    """
    if not nums:
        return (0, 0)

    current_sum = max_sum = nums[0]
    start = end = 0
    temp_start = 0

    for i in range(1, len(nums)):
        if nums[i] > current_sum + nums[i]:
            current_sum = nums[i]
            temp_start = i
        else:
            current_sum = current_sum + nums[i]

        if current_sum > max_sum:
            max_sum = current_sum
            start = temp_start
            end = i

    return (start, end)
