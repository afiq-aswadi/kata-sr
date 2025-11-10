"""Subsets II - LeetCode 90 - Reference Solution"""

def subsets_with_dup(nums: list[int]) -> list[list[int]]:
    nums.sort()
    result = []

    def backtrack(start: int, current: list[int]):
        result.append(current[:])

        for i in range(start, len(nums)):
            # Skip duplicates
            if i > start and nums[i] == nums[i-1]:
                continue

            current.append(nums[i])
            backtrack(i + 1, current)
            current.pop()

    backtrack(0, [])
    return result
