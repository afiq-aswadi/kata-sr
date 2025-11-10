"""Jump Game - LeetCode 55 - Reference Solution"""

def can_jump(nums: list[int]) -> bool:
    max_reach = 0

    for i in range(len(nums)):
        if i > max_reach:
            return False
        max_reach = max(max_reach, i + nums[i])
        if max_reach >= len(nums) - 1:
            return True

    return True
