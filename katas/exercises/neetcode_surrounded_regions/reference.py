"""Surrounded Regions - LeetCode 130 - Reference Solution"""

def solve(board: list[list[str]]) -> None:
    """
    Do not return anything, modify board in-place instead.
    """
    if not board:
        return

    rows, cols = len(board), len(board[0])

    def dfs(r: int, c: int) -> None:
        if (r < 0 or r >= rows or
            c < 0 or c >= cols or
            board[r][c] != "O"):
            return

        board[r][c] = "T"  # Temporary marker
        dfs(r + 1, c)
        dfs(r - 1, c)
        dfs(r, c + 1)
        dfs(r, c - 1)

    # Mark unsurrounded 'O's from borders
    for r in range(rows):
        dfs(r, 0)
        dfs(r, cols - 1)

    for c in range(cols):
        dfs(0, c)
        dfs(rows - 1, c)

    # Flip surrounded 'O's to 'X' and restore 'T' to 'O'
    for r in range(rows):
        for c in range(cols):
            if board[r][c] == "O":
                board[r][c] = "X"
            elif board[r][c] == "T":
                board[r][c] = "O"
