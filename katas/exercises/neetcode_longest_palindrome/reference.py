"""Longest Palindromic Substring - LeetCode 5 - Reference Solution"""

def longest_palindrome(s: str) -> str:
    if not s:
        return ""

    def expand_around_center(left, right):
        while left >= 0 and right < len(s) and s[left] == s[right]:
            left -= 1
            right += 1
        return right - left - 1

    start, max_len = 0, 0
    for i in range(len(s)):
        # Odd length palindrome
        len1 = expand_around_center(i, i)
        # Even length palindrome
        len2 = expand_around_center(i, i + 1)
        curr_len = max(len1, len2)

        if curr_len > max_len:
            max_len = curr_len
            start = i - (curr_len - 1) // 2

    return s[start:start + max_len]
