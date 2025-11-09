"""Reference solution for Fibonacci DP."""


def fibonacci_dp(n: int) -> int:
    """Compute nth Fibonacci number using dynamic programming."""
    memo = {}

    def fib(k):
        if k in memo:
            return memo[k]
        if k <= 1:
            return k
        memo[k] = fib(k - 1) + fib(k - 2)
        return memo[k]

    return fib(n)
