"""Container With Most Water - LeetCode 11 - Reference Solution"""

def max_area(height: list[int]) -> int:
    left, right = 0, len(height) - 1
    max_water = 0

    while left < right:
        width = right - left
        current_height = min(height[left], height[right])
        water = width * current_height
        max_water = max(max_water, water)

        if height[left] < height[right]:
            left += 1
        else:
            right -= 1

    return max_water
