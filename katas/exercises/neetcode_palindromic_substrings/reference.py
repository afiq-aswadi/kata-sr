"""Palindromic Substrings - LeetCode 647 - Reference Solution"""

def count_substrings(s: str) -> int:
    count = 0

    def expand_around_center(left, right):
        nonlocal count
        while left >= 0 and right < len(s) and s[left] == s[right]:
            count += 1
            left -= 1
            right += 1

    for i in range(len(s)):
        # Odd length palindromes
        expand_around_center(i, i)
        # Even length palindromes
        expand_around_center(i, i + 1)

    return count
