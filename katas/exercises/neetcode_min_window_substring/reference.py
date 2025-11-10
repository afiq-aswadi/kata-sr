"""Minimum Window Substring - LeetCode 76 - Reference Solution"""

def min_window(s: str, t: str) -> str:
    if not t or not s:
        return ""

    # Build frequency map for t
    t_count = {}
    for char in t:
        t_count[char] = t_count.get(char, 0) + 1

    required = len(t_count)  # Number of unique characters in t
    formed = 0  # Number of unique characters in current window with desired frequency

    window_counts = {}
    left = 0
    min_len = float('inf')
    min_left = 0

    for right in range(len(s)):
        char = s[right]
        window_counts[char] = window_counts.get(char, 0) + 1

        # Check if the frequency of the current character matches the desired frequency in t
        if char in t_count and window_counts[char] == t_count[char]:
            formed += 1

        # Try to contract the window until it ceases to be 'desirable'
        while formed == required and left <= right:
            # Update result if this window is smaller
            if right - left + 1 < min_len:
                min_len = right - left + 1
                min_left = left

            # Remove from left
            char = s[left]
            window_counts[char] -= 1
            if char in t_count and window_counts[char] < t_count[char]:
                formed -= 1

            left += 1

    return "" if min_len == float('inf') else s[min_left:min_left + min_len]
