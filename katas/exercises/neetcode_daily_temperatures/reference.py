"""Daily Temperatures - LeetCode 739 - Reference Solution"""

def daily_temperatures(temperatures: list[int]) -> list[int]:
    n = len(temperatures)
    result = [0] * n
    stack = []

    for i in range(n):
        while stack and temperatures[i] > temperatures[stack[-1]]:
            prev_i = stack.pop()
            result[prev_i] = i - prev_i
        stack.append(i)

    return result
