"""Sliding Window Maximum - LeetCode 239 - Reference Solution"""

from collections import deque

def max_sliding_window(nums: list[int], k: int) -> list[int]:
    if not nums or k == 0:
        return []

    dq = deque()  # Store indices
    result = []

    for i in range(len(nums)):
        # Remove indices that are out of the current window
        while dq and dq[0] < i - k + 1:
            dq.popleft()

        # Remove indices whose corresponding values are less than nums[i]
        # (they can never be the maximum)
        while dq and nums[dq[-1]] < nums[i]:
            dq.pop()

        dq.append(i)

        # The front of deque is the index of maximum element
        if i >= k - 1:
            result.append(nums[dq[0]])

    return result
