"""Sum of Two Integers - LeetCode 371 - Reference Solution"""

def get_sum(a: int, b: int) -> int:
    mask = 0xFFFFFFFF

    while b != 0:
        carry = ((a & b) << 1) & mask
        a = (a ^ b) & mask
        b = carry

    # Handle negative numbers
    if a > 0x7FFFFFFF:
        return ~(a ^ mask)

    return a
