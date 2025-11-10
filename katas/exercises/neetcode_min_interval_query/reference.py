"""Minimum Interval to Include Each Query - LeetCode 1851 - Reference Solution"""

import heapq

def min_interval(intervals: list[list[int]], queries: list[int]) -> list[int]:
    intervals.sort()
    min_heap = []
    result = {}
    i = 0

    for q in sorted(queries):
        # Add all intervals that start before or at query
        while i < len(intervals) and intervals[i][0] <= q:
            left, right = intervals[i]
            heapq.heappush(min_heap, (right - left + 1, right))
            i += 1

        # Remove intervals that end before query
        while min_heap and min_heap[0][1] < q:
            heapq.heappop(min_heap)

        result[q] = min_heap[0][0] if min_heap else -1

    return [result[q] for q in queries]
