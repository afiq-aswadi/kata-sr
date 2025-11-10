"""Find the Duplicate Number - LeetCode 287 - Reference Solution"""

def find_duplicate(nums: list[int]) -> int:
    # Phase 1: Find intersection point in the cycle
    slow = nums[0]
    fast = nums[0]

    while True:
        slow = nums[slow]
        fast = nums[nums[fast]]
        if slow == fast:
            break

    # Phase 2: Find the entrance to the cycle (duplicate number)
    slow = nums[0]
    while slow != fast:
        slow = nums[slow]
        fast = nums[fast]

    return slow
