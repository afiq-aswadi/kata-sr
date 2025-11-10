"""Non-overlapping Intervals - LeetCode 435 - Reference Solution"""

def erase_overlap_intervals(intervals: list[list[int]]) -> int:
    intervals.sort(key=lambda x: x[1])
    count = 0
    end = float('-inf')

    for start, interval_end in intervals:
        if start >= end:
            end = interval_end
        else:
            count += 1

    return count
