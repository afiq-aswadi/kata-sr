"""Maximum Product Subarray - LeetCode 152 - Reference Solution"""

def max_product(nums: list[int]) -> int:
    if not nums:
        return 0

    max_prod = min_prod = result = nums[0]

    for i in range(1, len(nums)):
        num = nums[i]
        # Store max_prod before updating (needed for min_prod calculation)
        temp_max = max(num, max_prod * num, min_prod * num)
        min_prod = min(num, max_prod * num, min_prod * num)
        max_prod = temp_max

        result = max(result, max_prod)

    return result
