"""Detect Squares - LeetCode 2013 - Reference Solution"""

from collections import defaultdict

class DetectSquares:
    def __init__(self):
        self.points = defaultdict(int)
        self.points_by_x = defaultdict(list)

    def add(self, point: list[int]) -> None:
        x, y = point
        self.points[(x, y)] += 1
        if (x, y) not in [(px, py) for px, py in self.points_by_x[x]]:
            self.points_by_x[x].append((x, y))

    def count(self, point: list[int]) -> int:
        x1, y1 = point
        result = 0

        for x2, y2 in self.points_by_x[x1]:
            if y2 == y1:
                continue

            side = abs(y2 - y1)

            # Check right side
            result += self.points[(x1 + side, y1)] * self.points[(x1 + side, y2)] * self.points[(x2, y2)]

            # Check left side
            result += self.points[(x1 - side, y1)] * self.points[(x1 - side, y2)] * self.points[(x2, y2)]

        return result
