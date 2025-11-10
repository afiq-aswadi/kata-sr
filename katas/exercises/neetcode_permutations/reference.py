"""Permutations - LeetCode 46 - Reference Solution"""

def permute(nums: list[int]) -> list[list[int]]:
    result = []

    def backtrack(current: list[int]):
        if len(current) == len(nums):
            result.append(current[:])
            return

        for num in nums:
            if num not in current:
                current.append(num)
                backtrack(current)
                current.pop()

    backtrack([])
    return result
