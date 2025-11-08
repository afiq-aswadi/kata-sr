"""Quicksort kata - reference solution."""



def quicksort(arr: list[int]) -> list[int]:
    """Sort array using quicksort algorithm."""
    if len(arr) <= 1:
        return arr.copy()

    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]

    return quicksort(left) + middle + quicksort(right)


def quicksort_inplace(arr: list[int], low: int = 0, high: int | None = None) -> None:
    """Sort array in-place using quicksort."""
    if high is None:
        high = len(arr) - 1

    if low < high:
        pivot_idx = partition_lomuto(arr, low, high)
        quicksort_inplace(arr, low, pivot_idx - 1)
        quicksort_inplace(arr, pivot_idx + 1, high)


def partition_lomuto(arr: list[int], low: int, high: int) -> int:
    """Partition array using Lomuto scheme."""
    pivot = arr[high]
    i = low - 1

    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]

    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1


def partition_hoare(arr: list[int], low: int, high: int) -> int:
    """Partition array using Hoare scheme."""
    pivot = arr[(low + high) // 2]
    i = low - 1
    j = high + 1

    while True:
        i += 1
        while arr[i] < pivot:
            i += 1

        j -= 1
        while arr[j] > pivot:
            j -= 1

        if i >= j:
            return j

        arr[i], arr[j] = arr[j], arr[i]
