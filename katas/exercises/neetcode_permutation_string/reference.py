"""Permutation in String - LeetCode 567 - Reference Solution"""

def check_inclusion(s1: str, s2: str) -> bool:
    if len(s1) > len(s2):
        return False

    s1_count = {}
    s2_count = {}

    # Build frequency map for s1
    for char in s1:
        s1_count[char] = s1_count.get(char, 0) + 1

    # Build initial window
    for i in range(len(s1)):
        s2_count[s2[i]] = s2_count.get(s2[i], 0) + 1

    # Check initial window
    if s1_count == s2_count:
        return True

    # Slide the window
    for i in range(len(s1), len(s2)):
        # Add new character
        s2_count[s2[i]] = s2_count.get(s2[i], 0) + 1

        # Remove old character
        left_char = s2[i - len(s1)]
        s2_count[left_char] -= 1
        if s2_count[left_char] == 0:
            del s2_count[left_char]

        # Check if window matches
        if s1_count == s2_count:
            return True

    return False
