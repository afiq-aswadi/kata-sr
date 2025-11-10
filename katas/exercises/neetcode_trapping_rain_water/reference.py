"""Trapping Rain Water - LeetCode 42 - Reference Solution"""

def trap(height: list[int]) -> int:
    if not height:
        return 0

    left, right = 0, len(height) - 1
    max_left, max_right = height[left], height[right]
    water = 0

    while left < right:
        if max_left < max_right:
            left += 1
            max_left = max(max_left, height[left])
            water += max_left - height[left]
        else:
            right -= 1
            max_right = max(max_right, height[right])
            water += max_right - height[right]

    return water
