"""Koko Eating Bananas - LeetCode 875 - Reference Solution"""

import math

def min_eating_speed(piles: list[int], h: int) -> int:
    def can_finish(speed: int) -> bool:
        hours = 0
        for pile in piles:
            hours += math.ceil(pile / speed)
        return hours <= h

    left, right = 1, max(piles)

    while left < right:
        mid = (left + right) // 2
        if can_finish(mid):
            right = mid
        else:
            left = mid + 1

    return left
