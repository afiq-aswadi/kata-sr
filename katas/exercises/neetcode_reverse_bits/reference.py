"""Reverse Bits - LeetCode 190 - Reference Solution"""

def reverse_bits(n: int) -> int:
    result = 0
    for i in range(32):
        result = (result << 1) | (n & 1)
        n >>= 1
    return result
