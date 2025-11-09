"""0/1 Knapsack problem using dynamic programming."""


def knapsack_01(weights: list[int], values: list[int], capacity: int) -> int:
    """Solve 0/1 knapsack problem.

    Given items with weights and values, find maximum value that can
    be achieved with a weight limit. Each item can be taken 0 or 1 times.

    Args:
        weights: weight of each item
        values: value of each item
        capacity: maximum weight capacity

    Returns:
        maximum value achievable
    """
    # TODO: Implement using 2D dynamic programming
    # Hint: dp[i][w] = max value using first i items with capacity w
    # Either take item i: dp[i][w] = dp[i-1][w-weights[i]] + values[i]
    # Or skip it: dp[i][w] = dp[i-1][w]
    # Choose the maximum
    # BLANK_START
    raise NotImplementedError
    # BLANK_END
