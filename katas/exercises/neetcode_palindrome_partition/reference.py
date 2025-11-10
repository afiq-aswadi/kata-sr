"""Palindrome Partitioning - LeetCode 131 - Reference Solution"""

def partition(s: str) -> list[list[str]]:
    result = []

    def is_palindrome(string: str) -> bool:
        return string == string[::-1]

    def backtrack(start: int, current: list[str]):
        if start == len(s):
            result.append(current[:])
            return

        for end in range(start + 1, len(s) + 1):
            substring = s[start:end]
            if is_palindrome(substring):
                current.append(substring)
                backtrack(end, current)
                current.pop()

    backtrack(0, [])
    return result
