"""Meeting Rooms II - LeetCode 253 - Reference Solution"""

def min_meeting_rooms(intervals: list[list[int]]) -> int:
    start = sorted([i[0] for i in intervals])
    end = sorted([i[1] for i in intervals])

    rooms = 0
    max_rooms = 0
    s = 0
    e = 0

    while s < len(intervals):
        if start[s] < end[e]:
            rooms += 1
            max_rooms = max(max_rooms, rooms)
            s += 1
        else:
            rooms -= 1
            e += 1

    return max_rooms
