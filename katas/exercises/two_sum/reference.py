"""Two sum kata - reference solution."""


def two_sum(nums: list[int], target: int) -> list[int]:
    """Find indices of two numbers that add up to target.

    Args:
        nums: array of integers
        target: target sum

    Returns:
        list of two indices [i, j] where nums[i] + nums[j] == target
    """
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []


def two_sum_sorted(nums: list[int], target: int) -> list[int]:
    """Find indices of two numbers in a sorted array that add up to target.

    Use two-pointer technique for O(1) space complexity.

    Args:
        nums: sorted array of integers
        target: target sum

    Returns:
        list of two indices [i, j] where nums[i] + nums[j] == target
    """
    left, right = 0, len(nums) - 1
    while left < right:
        current_sum = nums[left] + nums[right]
        if current_sum == target:
            return [left, right]
        elif current_sum < target:
            left += 1
        else:
            right -= 1
    return []
