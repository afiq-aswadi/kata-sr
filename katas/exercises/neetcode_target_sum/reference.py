"""Target Sum - LeetCode 494 - Reference Solution"""

def find_target_sum_ways(nums: list[int], target: int) -> int:
    total = sum(nums)
    if abs(target) > total or (total + target) % 2 != 0:
        return 0

    # Transform to subset sum problem
    subset_sum = (total + target) // 2
    dp = [0] * (subset_sum + 1)
    dp[0] = 1

    for num in nums:
        for i in range(subset_sum, num - 1, -1):
            dp[i] += dp[i - num]

    return dp[subset_sum]
