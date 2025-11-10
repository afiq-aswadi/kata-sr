"""Number of Islands - LeetCode 200 - Reference Solution"""

def num_islands(grid: list[list[str]]) -> int:
    if not grid:
        return 0

    rows, cols = len(grid), len(grid[0])
    visited = set()
    count = 0

    def dfs(r: int, c: int) -> None:
        if (r < 0 or r >= rows or
            c < 0 or c >= cols or
            grid[r][c] == "0" or
            (r, c) in visited):
            return

        visited.add((r, c))
        dfs(r + 1, c)
        dfs(r - 1, c)
        dfs(r, c + 1)
        dfs(r, c - 1)

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == "1" and (r, c) not in visited:
                dfs(r, c)
                count += 1

    return count
