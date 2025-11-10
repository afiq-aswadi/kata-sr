"""House Robber II - LeetCode 213 - Reference Solution"""

def rob(nums: list[int]) -> int:
    if len(nums) == 1:
        return nums[0]
    if len(nums) == 2:
        return max(nums)

    def rob_linear(houses):
        prev2, prev1 = 0, 0
        for num in houses:
            current = max(prev1, prev2 + num)
            prev2, prev1 = prev1, current
        return prev1

    # Either rob houses [0..n-2] or [1..n-1]
    return max(rob_linear(nums[:-1]), rob_linear(nums[1:]))
