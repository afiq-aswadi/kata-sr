"""Subsets - LeetCode 78 - Reference Solution"""

def subsets(nums: list[int]) -> list[list[int]]:
    result = []

    def backtrack(start: int, current: list[int]):
        result.append(current[:])

        for i in range(start, len(nums)):
            current.append(nums[i])
            backtrack(i + 1, current)
            current.pop()

    backtrack(0, [])
    return result
