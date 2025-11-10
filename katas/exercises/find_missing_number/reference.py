"""Find missing number kata - reference solution."""


def missing_number_sum(nums: list[int]) -> int:
    """Find missing number using sum formula.

    Use the formula: sum(0 to n) = n * (n + 1) / 2
    Missing = expected_sum - actual_sum

    Args:
        nums: array of n distinct numbers from range [0, n]

    Returns:
        the missing number
    """
    n = len(nums)
    expected_sum = n * (n + 1) // 2
    actual_sum = sum(nums)
    return expected_sum - actual_sum


def missing_number_xor(nums: list[int]) -> int:
    """Find missing number using XOR operation.

    XOR has property: a ^ a = 0 and a ^ 0 = a
    XOR all numbers and all indices to find missing number.

    Args:
        nums: array of n distinct numbers from range [0, n]

    Returns:
        the missing number
    """
    result = len(nums)  # Start with n

    for i, num in enumerate(nums):
        result ^= i ^ num

    return result
