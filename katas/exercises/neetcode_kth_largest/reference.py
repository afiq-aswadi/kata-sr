"""Kth Largest Element in an Array - LeetCode 215 - Reference Solution"""
import heapq

def find_kth_largest(nums: list[int], k: int) -> int:
    # Use min heap of size k to track k largest elements
    heap = nums[:k]
    heapq.heapify(heap)

    for num in nums[k:]:
        if num > heap[0]:
            heapq.heappop(heap)
            heapq.heappush(heap, num)

    return heap[0]
