"""N-Queens II - LeetCode 52 - Reference Solution"""

def total_n_queens(n: int) -> int:
    cols = set()
    diagonals = set()  # r - c
    anti_diagonals = set()  # r + c
    count = 0

    def backtrack(row: int):
        nonlocal count

        if row == n:
            count += 1
            return

        for col in range(n):
            if col in cols or (row - col) in diagonals or (row + col) in anti_diagonals:
                continue

            cols.add(col)
            diagonals.add(row - col)
            anti_diagonals.add(row + col)

            backtrack(row + 1)

            cols.remove(col)
            diagonals.remove(row - col)
            anti_diagonals.remove(row + col)

    backtrack(0)
    return count
