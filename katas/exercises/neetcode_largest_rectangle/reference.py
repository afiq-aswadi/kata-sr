"""Largest Rectangle in Histogram - LeetCode 84 - Reference Solution"""

def largest_rectangle_area(heights: list[int]) -> int:
    stack = []
    max_area = 0
    heights.append(0)

    for i, h in enumerate(heights):
        while stack and heights[stack[-1]] > h:
            height_idx = stack.pop()
            width = i if not stack else i - stack[-1] - 1
            max_area = max(max_area, heights[height_idx] * width)
        stack.append(i)

    heights.pop()
    return max_area
