"""Combination Sum - LeetCode 39 - Reference Solution"""

def combination_sum(candidates: list[int], target: int) -> list[list[int]]:
    result = []

    def backtrack(start: int, current: list[int], total: int):
        if total == target:
            result.append(current[:])
            return

        if total > target:
            return

        for i in range(start, len(candidates)):
            current.append(candidates[i])
            backtrack(i, current, total + candidates[i])
            current.pop()

    backtrack(0, [], 0)
    return result
