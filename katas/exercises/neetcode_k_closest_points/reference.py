"""K Closest Points to Origin - LeetCode 973 - Reference Solution"""
import heapq

def k_closest(points: list[list[int]], k: int) -> list[list[int]]:
    # Use min heap with distance squared (no need for sqrt)
    heap = []
    for x, y in points:
        dist = x * x + y * y
        heapq.heappush(heap, (dist, [x, y]))

    result = []
    for _ in range(k):
        result.append(heapq.heappop(heap)[1])

    return result
