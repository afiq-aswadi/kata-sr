"""Merge sorted arrays kata - reference solution."""


def merge_sorted(nums1: list[int], nums2: list[int]) -> list[int]:
    """Merge two sorted arrays into a new sorted array.

    Args:
        nums1: first sorted array
        nums2: second sorted array

    Returns:
        new sorted array containing all elements from both arrays
    """
    result = []
    i, j = 0, 0

    while i < len(nums1) and j < len(nums2):
        if nums1[i] <= nums2[j]:
            result.append(nums1[i])
            i += 1
        else:
            result.append(nums2[j])
            j += 1

    # Add remaining elements
    result.extend(nums1[i:])
    result.extend(nums2[j:])

    return result


def merge_in_place(nums1: list[int], m: int, nums2: list[int], n: int) -> None:
    """Merge nums2 into nums1 in-place.

    nums1 has length m+n where last n elements are 0 (placeholders).
    After merging, nums1 should contain all m+n elements sorted.

    Args:
        nums1: first array with extra space (modified in-place)
        m: number of initialized elements in nums1
        nums2: second array
        n: number of elements in nums2
    """
    # Start from the end
    i = m - 1  # Last element in nums1's initialized portion
    j = n - 1  # Last element in nums2
    k = m + n - 1  # Last position in nums1

    # Merge from the end
    while i >= 0 and j >= 0:
        if nums1[i] > nums2[j]:
            nums1[k] = nums1[i]
            i -= 1
        else:
            nums1[k] = nums2[j]
            j -= 1
        k -= 1

    # If nums2 has remaining elements, copy them
    while j >= 0:
        nums1[k] = nums2[j]
        j -= 1
        k -= 1
