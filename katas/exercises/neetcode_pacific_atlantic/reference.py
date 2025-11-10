"""Pacific Atlantic Water Flow - LeetCode 417 - Reference Solution"""

def pacific_atlantic(heights: list[list[int]]) -> list[list[int]]:
    if not heights:
        return []

    rows, cols = len(heights), len(heights[0])
    pacific = set()
    atlantic = set()

    def dfs(r: int, c: int, visited: set, prev_height: int) -> None:
        if (r < 0 or r >= rows or
            c < 0 or c >= cols or
            (r, c) in visited or
            heights[r][c] < prev_height):
            return

        visited.add((r, c))
        dfs(r + 1, c, visited, heights[r][c])
        dfs(r - 1, c, visited, heights[r][c])
        dfs(r, c + 1, visited, heights[r][c])
        dfs(r, c - 1, visited, heights[r][c])

    # DFS from Pacific (top and left edges)
    for c in range(cols):
        dfs(0, c, pacific, heights[0][c])
        dfs(rows - 1, c, atlantic, heights[rows - 1][c])

    for r in range(rows):
        dfs(r, 0, pacific, heights[r][0])
        dfs(r, cols - 1, atlantic, heights[r][cols - 1])

    # Find intersection
    result = []
    for r in range(rows):
        for c in range(cols):
            if (r, c) in pacific and (r, c) in atlantic:
                result.append([r, c])

    return result
