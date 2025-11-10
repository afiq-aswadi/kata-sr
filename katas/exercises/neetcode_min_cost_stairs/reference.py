"""Min Cost Climbing Stairs - LeetCode 746 - Reference Solution"""

def min_cost_climbing_stairs(cost: list[int]) -> int:
    n = len(cost)
    if n <= 2:
        return min(cost)

    prev2, prev1 = cost[0], cost[1]
    for i in range(2, n):
        current = cost[i] + min(prev1, prev2)
        prev2, prev1 = prev1, current

    return min(prev1, prev2)
