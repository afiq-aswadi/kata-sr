"""Find Median from Data Stream - LeetCode 295 - Reference Solution"""
import heapq

class MedianFinder:
    def __init__(self):
        # Max heap for left half (smaller elements), min heap for right half (larger elements)
        self.left = []  # Max heap (negate values)
        self.right = []  # Min heap

    def add_num(self, num: int) -> None:
        # Add to left (max heap) by default
        heapq.heappush(self.left, -num)

        # Ensure every element in left is <= every element in right
        if self.left and self.right and -self.left[0] > self.right[0]:
            val = -heapq.heappop(self.left)
            heapq.heappush(self.right, val)

        # Balance heaps: left should have at most one more element than right
        if len(self.left) > len(self.right) + 1:
            val = -heapq.heappop(self.left)
            heapq.heappush(self.right, val)
        if len(self.right) > len(self.left):
            val = heapq.heappop(self.right)
            heapq.heappush(self.left, -val)

    def find_median(self) -> float:
        if len(self.left) > len(self.right):
            return float(-self.left[0])
        return (-self.left[0] + self.right[0]) / 2.0
