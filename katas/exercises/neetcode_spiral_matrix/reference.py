"""Spiral Matrix - LeetCode 54 - Reference Solution"""

def spiral_order(matrix: list[list[int]]) -> list[int]:
    result = []
    top, bottom = 0, len(matrix) - 1
    left, right = 0, len(matrix[0]) - 1

    while top <= bottom and left <= right:
        # Move right
        for col in range(left, right + 1):
            result.append(matrix[top][col])
        top += 1

        # Move down
        for row in range(top, bottom + 1):
            result.append(matrix[row][right])
        right -= 1

        # Move left (if there's a row remaining)
        if top <= bottom:
            for col in range(right, left - 1, -1):
                result.append(matrix[bottom][col])
            bottom -= 1

        # Move up (if there's a column remaining)
        if left <= right:
            for row in range(bottom, top - 1, -1):
                result.append(matrix[row][left])
            left += 1

    return result
