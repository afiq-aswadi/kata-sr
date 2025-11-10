"""Max Area of Island - LeetCode 695 - Reference Solution"""

def max_area_of_island(grid: list[list[int]]) -> int:
    rows, cols = len(grid), len(grid[0])
    visited = set()

    def dfs(r: int, c: int) -> int:
        if (r < 0 or r >= rows or
            c < 0 or c >= cols or
            grid[r][c] == 0 or
            (r, c) in visited):
            return 0

        visited.add((r, c))
        return 1 + dfs(r + 1, c) + dfs(r - 1, c) + dfs(r, c + 1) + dfs(r, c - 1)

    max_area = 0
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 1 and (r, c) not in visited:
                max_area = max(max_area, dfs(r, c))

    return max_area
