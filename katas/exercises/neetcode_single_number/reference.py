"""Single Number - LeetCode 136 - Reference Solution"""

def single_number(nums: list[int]) -> int:
    result = 0
    for num in nums:
        result ^= num
    return result
