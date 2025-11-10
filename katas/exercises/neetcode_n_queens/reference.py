"""N-Queens - LeetCode 51 - Reference Solution"""

def solve_n_queens(n: int) -> list[list[str]]:
    result = []
    board = [["."] * n for _ in range(n)]
    cols = set()
    diagonals = set()  # r - c
    anti_diagonals = set()  # r + c

    def backtrack(row: int):
        if row == n:
            result.append(["".join(r) for r in board])
            return

        for col in range(n):
            if col in cols or (row - col) in diagonals or (row + col) in anti_diagonals:
                continue

            cols.add(col)
            diagonals.add(row - col)
            anti_diagonals.add(row + col)
            board[row][col] = "Q"

            backtrack(row + 1)

            cols.remove(col)
            diagonals.remove(row - col)
            anti_diagonals.remove(row + col)
            board[row][col] = "."

    backtrack(0)
    return result
