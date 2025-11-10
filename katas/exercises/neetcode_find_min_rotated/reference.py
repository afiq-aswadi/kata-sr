"""Find Minimum in Rotated Sorted Array - LeetCode 153 - Reference Solution"""

def find_min(nums: list[int]) -> int:
    left, right = 0, len(nums) - 1

    while left < right:
        mid = (left + right) // 2

        if nums[mid] > nums[right]:
            left = mid + 1
        else:
            right = mid

    return nums[left]
