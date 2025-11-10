"""Combination Sum II - LeetCode 40 - Reference Solution"""

def combination_sum2(candidates: list[int], target: int) -> list[list[int]]:
    candidates.sort()
    result = []

    def backtrack(start: int, current: list[int], total: int):
        if total == target:
            result.append(current[:])
            return

        if total > target:
            return

        for i in range(start, len(candidates)):
            # Skip duplicates
            if i > start and candidates[i] == candidates[i-1]:
                continue

            current.append(candidates[i])
            backtrack(i + 1, current, total + candidates[i])
            current.pop()

    backtrack(0, [], 0)
    return result
