"""Reverse Integer - LeetCode 7 - Reference Solution"""

def reverse(x: int) -> int:
    sign = -1 if x < 0 else 1
    x = abs(x)

    result = 0
    while x:
        digit = x % 10
        x //= 10

        # Check for overflow before multiplying
        if result > (2**31 - 1) // 10:
            return 0

        result = result * 10 + digit

    result *= sign

    # Check bounds
    if result < -2**31 or result > 2**31 - 1:
        return 0

    return result
