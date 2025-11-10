"""Word Search - LeetCode 79 - Reference Solution"""

def exist(board: list[list[str]], word: str) -> bool:
    rows, cols = len(board), len(board[0])
    visited = set()

    def dfs(r: int, c: int, i: int) -> bool:
        if i == len(word):
            return True

        if (r < 0 or r >= rows or
            c < 0 or c >= cols or
            (r, c) in visited or
            board[r][c] != word[i]):
            return False

        visited.add((r, c))
        result = (dfs(r + 1, c, i + 1) or
                  dfs(r - 1, c, i + 1) or
                  dfs(r, c + 1, i + 1) or
                  dfs(r, c - 1, i + 1))
        visited.remove((r, c))

        return result

    for r in range(rows):
        for c in range(cols):
            if dfs(r, c, 0):
                return True

    return False
