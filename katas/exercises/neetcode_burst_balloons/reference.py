"""Burst Balloons - LeetCode 312 - Reference Solution"""

def max_coins(nums: list[int]) -> int:
    # Add virtual balloons with value 1 at boundaries
    nums = [1] + nums + [1]
    n = len(nums)
    dp = [[0] * n for _ in range(n)]

    # Length of subproblem
    for length in range(2, n):
        for left in range(n - length):
            right = left + length
            # Try bursting each balloon last in this range
            for i in range(left + 1, right):
                coins = nums[left] * nums[i] * nums[right]
                coins += dp[left][i] + dp[i][right]
                dp[left][right] = max(dp[left][right], coins)

    return dp[0][n-1]
