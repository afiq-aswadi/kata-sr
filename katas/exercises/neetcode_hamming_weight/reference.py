"""Number of 1 Bits - LeetCode 191 - Reference Solution"""

def hamming_weight(n: int) -> int:
    count = 0
    while n:
        n &= n - 1
        count += 1
    return count
