"""Task Scheduler - LeetCode 621 - Reference Solution"""
import heapq
from collections import Counter, deque

def least_interval(tasks: list[str], n: int) -> int:
    # Count task frequencies
    counts = Counter(tasks)
    max_heap = [-count for count in counts.values()]
    heapq.heapify(max_heap)

    time = 0
    queue = deque()  # (count, idle_time)

    while max_heap or queue:
        time += 1

        if max_heap:
            count = heapq.heappop(max_heap)
            count += 1  # Decrease count (was negative)
            if count != 0:
                queue.append((count, time + n))

        if queue and queue[0][1] == time:
            heapq.heappush(max_heap, queue.popleft()[0])

    return time
