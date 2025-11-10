"""Last Stone Weight - LeetCode 1046 - Reference Solution"""
import heapq

def last_stone_weight(stones: list[int]) -> int:
    # Convert to max heap by negating values
    heap = [-stone for stone in stones]
    heapq.heapify(heap)

    while len(heap) > 1:
        first = -heapq.heappop(heap)
        second = -heapq.heappop(heap)

        if first != second:
            heapq.heappush(heap, -(first - second))

    return -heap[0] if heap else 0
