#!/usr/bin/env python3
"""Generate NeetCode 150 kata files."""

import os
from pathlib import Path

# NeetCode 150 problems organized by category
NEETCODE_150 = {
    "neetcode_arrays_hashing": [
        ("neetcode_contains_duplicate", "Contains Duplicate", 217, 1, [], """
Contains Duplicate (LeetCode 217)

Given an integer array nums, return true if any value appears at least twice in the array,
and return false if every element is distinct.

Example 1:
Input: nums = [1,2,3,1]
Output: true

Example 2:
Input: nums = [1,2,3,4]
Output: false

Constraints:
- 1 <= nums.length <= 10^5
- -10^9 <= nums[i] <= 10^9
""", """def contains_duplicate(nums: list[int]) -> bool:
    # TODO: Use a set to check for duplicates
    # BLANK_START
    pass
    # BLANK_END
""", """def contains_duplicate(nums: list[int]) -> bool:
    return len(nums) != len(set(nums))
""", """def test_contains_duplicate_example1():
    from template import contains_duplicate
    assert contains_duplicate([1,2,3,1]) == True

def test_contains_duplicate_example2():
    from template import contains_duplicate
    assert contains_duplicate([1,2,3,4]) == False

def test_contains_duplicate_empty():
    from template import contains_duplicate
    assert contains_duplicate([1]) == False
"""),
        ("neetcode_valid_anagram", "Valid Anagram", 242, 1, [], """
Valid Anagram (LeetCode 242)

Given two strings s and t, return true if t is an anagram of s, and false otherwise.

Example 1:
Input: s = "anagram", t = "nagaram"
Output: true

Example 2:
Input: s = "rat", t = "car"
Output: false

Constraints:
- 1 <= s.length, t.length <= 5 * 10^4
- s and t consist of lowercase English letters
""", """def is_anagram(s: str, t: str) -> bool:
    # TODO: Compare character counts
    # BLANK_START
    pass
    # BLANK_END
""", """def is_anagram(s: str, t: str) -> bool:
    if len(s) != len(t):
        return False

    count = {}
    for char in s:
        count[char] = count.get(char, 0) + 1

    for char in t:
        if char not in count:
            return False
        count[char] -= 1
        if count[char] < 0:
            return False

    return True
""", """def test_is_anagram_example1():
    from template import is_anagram
    assert is_anagram("anagram", "nagaram") == True

def test_is_anagram_example2():
    from template import is_anagram
    assert is_anagram("rat", "car") == False

def test_is_anagram_empty():
    from template import is_anagram
    assert is_anagram("", "") == True
"""),
        ("neetcode_two_sum", "Two Sum", 1, 1, [], """
Two Sum (LeetCode 1)

Given an array of integers nums and an integer target, return indices of the two numbers
such that they add up to target.

Example 1:
Input: nums = [2,7,11,15], target = 9
Output: [0,1]

Example 2:
Input: nums = [3,2,4], target = 6
Output: [1,2]

Constraints:
- 2 <= nums.length <= 10^4
- -10^9 <= nums[i] <= 10^9
- Only one valid answer exists
""", """def two_sum(nums: list[int], target: int) -> list[int]:
    # TODO: Use a hash map to find complement
    # BLANK_START
    pass
    # BLANK_END
""", """def two_sum(nums: list[int], target: int) -> list[int]:
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []
""", """def test_two_sum_example1():
    from template import two_sum
    assert two_sum([2,7,11,15], 9) == [0,1]

def test_two_sum_example2():
    from template import two_sum
    assert two_sum([3,2,4], 6) == [1,2]

def test_two_sum_negative():
    from template import two_sum
    assert two_sum([-1,-2,-3,-4,-5], -8) == [2,4]
"""),
        ("neetcode_group_anagrams", "Group Anagrams", 49, 2, ["neetcode_valid_anagram"], """
Group Anagrams (LeetCode 49)

Given an array of strings strs, group the anagrams together. You can return the answer in any order.

Example 1:
Input: strs = ["eat","tea","tan","ate","nat","bat"]
Output: [["bat"],["nat","tan"],["ate","eat","tea"]]

Example 2:
Input: strs = [""]
Output: [[""]]

Constraints:
- 1 <= strs.length <= 10^4
- 0 <= strs[i].length <= 100
""", """def group_anagrams(strs: list[str]) -> list[list[str]]:
    # TODO: Use sorted string or character count as key
    # BLANK_START
    pass
    # BLANK_END
""", """from collections import defaultdict

def group_anagrams(strs: list[str]) -> list[list[str]]:
    groups = defaultdict(list)
    for s in strs:
        key = tuple(sorted(s))
        groups[key].append(s)
    return list(groups.values())
""", """def test_group_anagrams_example1():
    from template import group_anagrams
    result = group_anagrams(["eat","tea","tan","ate","nat","bat"])
    result = [sorted(group) for group in result]
    result = sorted(result)
    expected = [["ate","eat","tea"], ["bat"], ["nat","tan"]]
    expected = [sorted(group) for group in expected]
    expected = sorted(expected)
    assert result == expected

def test_group_anagrams_empty():
    from template import group_anagrams
    assert group_anagrams([""]) == [[""]]

def test_group_anagrams_single():
    from template import group_anagrams
    assert group_anagrams(["a"]) == [["a"]]
"""),
        ("neetcode_top_k_frequent", "Top K Frequent Elements", 347, 2, [], """
Top K Frequent Elements (LeetCode 347)

Given an integer array nums and an integer k, return the k most frequent elements.
You may return the answer in any order.

Example 1:
Input: nums = [1,1,1,2,2,3], k = 2
Output: [1,2]

Example 2:
Input: nums = [1], k = 1
Output: [1]

Constraints:
- 1 <= nums.length <= 10^5
- -10^4 <= nums[i] <= 10^4
- k is in the range [1, the number of unique elements in the array]
""", """def top_k_frequent(nums: list[int], k: int) -> list[int]:
    # TODO: Use Counter and heap or bucket sort
    # BLANK_START
    pass
    # BLANK_END
""", """from collections import Counter
import heapq

def top_k_frequent(nums: list[int], k: int) -> list[int]:
    count = Counter(nums)
    return heapq.nlargest(k, count.keys(), key=count.get)
""", """def test_top_k_frequent_example1():
    from template import top_k_frequent
    result = top_k_frequent([1,1,1,2,2,3], 2)
    assert sorted(result) == [1, 2]

def test_top_k_frequent_example2():
    from template import top_k_frequent
    assert top_k_frequent([1], 1) == [1]

def test_top_k_frequent_all():
    from template import top_k_frequent
    result = top_k_frequent([4,1,-1,2,-1,2,3], 2)
    assert sorted(result) == [-1, 2]
"""),
        ("neetcode_product_except_self", "Product of Array Except Self", 238, 2, [], """
Product of Array Except Self (LeetCode 238)

Given an integer array nums, return an array answer such that answer[i] is equal to
the product of all the elements of nums except nums[i].

You must write an algorithm that runs in O(n) time and without using the division operation.

Example 1:
Input: nums = [1,2,3,4]
Output: [24,12,8,6]

Example 2:
Input: nums = [-1,1,0,-3,3]
Output: [0,0,9,0,0]

Constraints:
- 2 <= nums.length <= 10^5
- -30 <= nums[i] <= 30
""", """def product_except_self(nums: list[int]) -> list[int]:
    # TODO: Use prefix and suffix products
    # BLANK_START
    pass
    # BLANK_END
""", """def product_except_self(nums: list[int]) -> list[int]:
    n = len(nums)
    result = [1] * n

    # Calculate prefix products
    prefix = 1
    for i in range(n):
        result[i] = prefix
        prefix *= nums[i]

    # Calculate suffix products and multiply
    suffix = 1
    for i in range(n - 1, -1, -1):
        result[i] *= suffix
        suffix *= nums[i]

    return result
""", """def test_product_except_self_example1():
    from template import product_except_self
    assert product_except_self([1,2,3,4]) == [24,12,8,6]

def test_product_except_self_example2():
    from template import product_except_self
    assert product_except_self([-1,1,0,-3,3]) == [0,0,9,0,0]

def test_product_except_self_two_elements():
    from template import product_except_self
    assert product_except_self([1,2]) == [2,1]
"""),
        ("neetcode_valid_sudoku", "Valid Sudoku", 36, 2, [], """
Valid Sudoku (LeetCode 36)

Determine if a 9 x 9 Sudoku board is valid. Only the filled cells need to be validated.

A Sudoku board is valid if:
1. Each row contains digits 1-9 without repetition
2. Each column contains digits 1-9 without repetition
3. Each 3x3 sub-box contains digits 1-9 without repetition

Example 1:
Input: board =
[["5","3",".",".","7",".",".",".","."]
,["6",".",".","1","9","5",".",".","."]
,[".","9","8",".",".",".",".","6","."]
,["8",".",".",".","6",".",".",".","3"]
,["4",".",".","8",".","3",".",".","1"]
,["7",".",".",".","2",".",".",".","6"]
,[".","6",".",".",".",".","2","8","."]
,[".",".",".","4","1","9",".",".","5"]
,[".",".",".",".","8",".",".","7","9"]]
Output: true

Constraints:
- board.length == 9
- board[i].length == 9
- board[i][j] is a digit 1-9 or '.'
""", """def is_valid_sudoku(board: list[list[str]]) -> bool:
    # TODO: Use sets to track seen digits in rows, columns, and boxes
    # BLANK_START
    pass
    # BLANK_END
""", """def is_valid_sudoku(board: list[list[str]]) -> bool:
    rows = [set() for _ in range(9)]
    cols = [set() for _ in range(9)]
    boxes = [set() for _ in range(9)]

    for r in range(9):
        for c in range(9):
            if board[r][c] == '.':
                continue

            val = board[r][c]
            box_idx = (r // 3) * 3 + (c // 3)

            if val in rows[r] or val in cols[c] or val in boxes[box_idx]:
                return False

            rows[r].add(val)
            cols[c].add(val)
            boxes[box_idx].add(val)

    return True
""", """def test_is_valid_sudoku_valid():
    from template import is_valid_sudoku
    board = [
        ["5","3",".",".","7",".",".",".","."],
        ["6",".",".","1","9","5",".",".","."],
        [".","9","8",".",".",".",".","6","."],
        ["8",".",".",".","6",".",".",".","3"],
        ["4",".",".","8",".","3",".",".","1"],
        ["7",".",".",".","2",".",".",".","6"],
        [".","6",".",".",".",".","2","8","."],
        [".",".",".","4","1","9",".",".","5"],
        [".",".",".",".","8",".",".","7","9"]
    ]
    assert is_valid_sudoku(board) == True

def test_is_valid_sudoku_invalid():
    from template import is_valid_sudoku
    board = [
        ["8","3",".",".","7",".",".",".","."],
        ["6",".",".","1","9","5",".",".","."],
        [".","9","8",".",".",".",".","6","."],
        ["8",".",".",".","6",".",".",".","3"],
        ["4",".",".","8",".","3",".",".","1"],
        ["7",".",".",".","2",".",".",".","6"],
        [".","6",".",".",".",".","2","8","."],
        [".",".",".","4","1","9",".",".","5"],
        [".",".",".",".","8",".",".","7","9"]
    ]
    assert is_valid_sudoku(board) == False
"""),
        ("neetcode_encode_decode_strings", "Encode and Decode Strings", 271, 2, [], """
Encode and Decode Strings (LeetCode 271)

Design an algorithm to encode a list of strings to a single string.
The encoded string is then decoded back to the original list of strings.

Example 1:
Input: ["hello","world"]
Output: ["hello","world"]

Example 2:
Input: [""]
Output: [""]

Constraints:
- 0 <= strs.length <= 200
- 0 <= strs[i].length <= 200
- strs[i] contains any possible characters
""", """def encode(strs: list[str]) -> str:
    # TODO: Encode list of strings with length prefix
    # BLANK_START
    pass
    # BLANK_END

def decode(s: str) -> list[str]:
    # TODO: Decode using length prefix
    # BLANK_START
    pass
    # BLANK_END
""", """def encode(strs: list[str]) -> str:
    result = []
    for s in strs:
        result.append(f"{len(s)}#{s}")
    return "".join(result)

def decode(s: str) -> list[str]:
    result = []
    i = 0
    while i < len(s):
        # Find the delimiter
        j = i
        while s[j] != '#':
            j += 1
        length = int(s[i:j])
        i = j + 1
        result.append(s[i:i + length])
        i += length
    return result
""", """def test_encode_decode_example1():
    from template import encode, decode
    strs = ["hello", "world"]
    assert decode(encode(strs)) == strs

def test_encode_decode_empty():
    from template import encode, decode
    strs = [""]
    assert decode(encode(strs)) == strs

def test_encode_decode_special():
    from template import encode, decode
    strs = ["#","##","a#b"]
    assert decode(encode(strs)) == strs
"""),
        ("neetcode_longest_consecutive", "Longest Consecutive Sequence", 128, 3, [], """
Longest Consecutive Sequence (LeetCode 128)

Given an unsorted array of integers nums, return the length of the longest consecutive elements sequence.

You must write an algorithm that runs in O(n) time.

Example 1:
Input: nums = [100,4,200,1,3,2]
Output: 4
Explanation: The longest consecutive elements sequence is [1, 2, 3, 4]. Therefore its length is 4.

Example 2:
Input: nums = [0,3,7,2,5,8,4,6,0,1]
Output: 9

Constraints:
- 0 <= nums.length <= 10^5
- -10^9 <= nums[i] <= 10^9
""", """def longest_consecutive(nums: list[int]) -> int:
    # TODO: Use a set to check for sequence starts
    # BLANK_START
    pass
    # BLANK_END
""", """def longest_consecutive(nums: list[int]) -> int:
    if not nums:
        return 0

    num_set = set(nums)
    longest = 0

    for num in num_set:
        # Only start counting if it's the start of a sequence
        if num - 1 not in num_set:
            current_num = num
            current_length = 1

            while current_num + 1 in num_set:
                current_num += 1
                current_length += 1

            longest = max(longest, current_length)

    return longest
""", """def test_longest_consecutive_example1():
    from template import longest_consecutive
    assert longest_consecutive([100,4,200,1,3,2]) == 4

def test_longest_consecutive_example2():
    from template import longest_consecutive
    assert longest_consecutive([0,3,7,2,5,8,4,6,0,1]) == 9

def test_longest_consecutive_empty():
    from template import longest_consecutive
    assert longest_consecutive([]) == 0

def test_longest_consecutive_single():
    from template import longest_consecutive
    assert longest_consecutive([1]) == 1
"""),
    ],
    "neetcode_two_pointers": [
        ("neetcode_valid_palindrome", "Valid Palindrome", 125, 1, [], """
Valid Palindrome (LeetCode 125)

A phrase is a palindrome if, after converting all uppercase letters into lowercase letters
and removing all non-alphanumeric characters, it reads the same forward and backward.

Given a string s, return true if it is a palindrome, or false otherwise.

Example 1:
Input: s = "A man, a plan, a canal: Panama"
Output: true

Example 2:
Input: s = "race a car"
Output: false

Constraints:
- 1 <= s.length <= 2 * 10^5
- s consists only of printable ASCII characters
""", """def is_palindrome(s: str) -> bool:
    # TODO: Use two pointers from both ends
    # BLANK_START
    pass
    # BLANK_END
""", """def is_palindrome(s: str) -> bool:
    left, right = 0, len(s) - 1

    while left < right:
        while left < right and not s[left].isalnum():
            left += 1
        while left < right and not s[right].isalnum():
            right -= 1

        if s[left].lower() != s[right].lower():
            return False

        left += 1
        right -= 1

    return True
""", """def test_is_palindrome_example1():
    from template import is_palindrome
    assert is_palindrome("A man, a plan, a canal: Panama") == True

def test_is_palindrome_example2():
    from template import is_palindrome
    assert is_palindrome("race a car") == False

def test_is_palindrome_empty():
    from template import is_palindrome
    assert is_palindrome(" ") == True

def test_is_palindrome_single():
    from template import is_palindrome
    assert is_palindrome("a") == True
"""),
        ("neetcode_3sum", "3Sum", 15, 2, ["neetcode_two_sum"], """
3Sum (LeetCode 15)

Given an integer array nums, return all the triplets [nums[i], nums[j], nums[k]]
such that i != j, i != k, and j != k, and nums[i] + nums[j] + nums[k] == 0.

Notice that the solution set must not contain duplicate triplets.

Example 1:
Input: nums = [-1,0,1,2,-1,-4]
Output: [[-1,-1,2],[-1,0,1]]

Example 2:
Input: nums = [0,1,1]
Output: []

Example 3:
Input: nums = [0,0,0]
Output: [[0,0,0]]

Constraints:
- 3 <= nums.length <= 3000
- -10^5 <= nums[i] <= 10^5
""", """def three_sum(nums: list[int]) -> list[list[int]]:
    # TODO: Sort array and use two pointers for each element
    # BLANK_START
    pass
    # BLANK_END
""", """def three_sum(nums: list[int]) -> list[list[int]]:
    nums.sort()
    result = []

    for i in range(len(nums) - 2):
        # Skip duplicates for first number
        if i > 0 and nums[i] == nums[i-1]:
            continue

        left, right = i + 1, len(nums) - 1

        while left < right:
            total = nums[i] + nums[left] + nums[right]

            if total < 0:
                left += 1
            elif total > 0:
                right -= 1
            else:
                result.append([nums[i], nums[left], nums[right]])

                # Skip duplicates for second and third numbers
                while left < right and nums[left] == nums[left + 1]:
                    left += 1
                while left < right and nums[right] == nums[right - 1]:
                    right -= 1

                left += 1
                right -= 1

    return result
""", """def test_three_sum_example1():
    from template import three_sum
    result = three_sum([-1,0,1,2,-1,-4])
    result = [sorted(triplet) for triplet in result]
    result = sorted(result)
    expected = [[-1,-1,2],[-1,0,1]]
    expected = [sorted(triplet) for triplet in expected]
    expected = sorted(expected)
    assert result == expected

def test_three_sum_example2():
    from template import three_sum
    assert three_sum([0,1,1]) == []

def test_three_sum_example3():
    from template import three_sum
    assert three_sum([0,0,0]) == [[0,0,0]]
"""),
        ("neetcode_container_most_water", "Container With Most Water", 11, 2, [], """
Container With Most Water (LeetCode 11)

You are given an integer array height of length n. There are n vertical lines drawn such that
the two endpoints of the ith line are (i, 0) and (i, height[i]).

Find two lines that together with the x-axis form a container, such that the container contains the most water.

Return the maximum amount of water a container can store.

Example 1:
Input: height = [1,8,6,2,5,4,8,3,7]
Output: 49

Example 2:
Input: height = [1,1]
Output: 1

Constraints:
- n == height.length
- 2 <= n <= 10^5
- 0 <= height[i] <= 10^4
""", """def max_area(height: list[int]) -> int:
    # TODO: Use two pointers from both ends, move the shorter one
    # BLANK_START
    pass
    # BLANK_END
""", """def max_area(height: list[int]) -> int:
    left, right = 0, len(height) - 1
    max_water = 0

    while left < right:
        width = right - left
        current_height = min(height[left], height[right])
        water = width * current_height
        max_water = max(max_water, water)

        if height[left] < height[right]:
            left += 1
        else:
            right -= 1

    return max_water
""", """def test_max_area_example1():
    from template import max_area
    assert max_area([1,8,6,2,5,4,8,3,7]) == 49

def test_max_area_example2():
    from template import max_area
    assert max_area([1,1]) == 1

def test_max_area_increasing():
    from template import max_area
    assert max_area([1,2,3,4,5]) == 6
"""),
        ("neetcode_trapping_rain_water", "Trapping Rain Water", 42, 3, ["neetcode_container_most_water"], """
Trapping Rain Water (LeetCode 42)

Given n non-negative integers representing an elevation map where the width of each bar is 1,
compute how much water it can trap after raining.

Example 1:
Input: height = [0,1,0,2,1,0,1,3,2,1,2,1]
Output: 6

Example 2:
Input: height = [4,2,0,3,2,5]
Output: 9

Constraints:
- n == height.length
- 1 <= n <= 2 * 10^4
- 0 <= height[i] <= 10^5
""", """def trap(height: list[int]) -> int:
    # TODO: Use two pointers with max_left and max_right
    # BLANK_START
    pass
    # BLANK_END
""", """def trap(height: list[int]) -> int:
    if not height:
        return 0

    left, right = 0, len(height) - 1
    max_left, max_right = height[left], height[right]
    water = 0

    while left < right:
        if max_left < max_right:
            left += 1
            max_left = max(max_left, height[left])
            water += max_left - height[left]
        else:
            right -= 1
            max_right = max(max_right, height[right])
            water += max_right - height[right]

    return water
""", """def test_trap_example1():
    from template import trap
    assert trap([0,1,0,2,1,0,1,3,2,1,2,1]) == 6

def test_trap_example2():
    from template import trap
    assert trap([4,2,0,3,2,5]) == 9

def test_trap_no_water():
    from template import trap
    assert trap([1,2,3,4,5]) == 0
"""),
    ],
    "neetcode_sliding_window": [
        ("neetcode_best_time_stock", "Best Time to Buy and Sell Stock", 121, 1, [], """
Best Time to Buy and Sell Stock (LeetCode 121)

You are given an array prices where prices[i] is the price of a given stock on the ith day.

You want to maximize your profit by choosing a single day to buy one stock and choosing
a different day in the future to sell that stock.

Return the maximum profit you can achieve from this transaction. If you cannot achieve any profit, return 0.

Example 1:
Input: prices = [7,1,5,3,6,4]
Output: 5

Example 2:
Input: prices = [7,6,4,3,1]
Output: 0

Constraints:
- 1 <= prices.length <= 10^5
- 0 <= prices[i] <= 10^4
""", """def max_profit(prices: list[int]) -> int:
    # TODO: Track minimum price and maximum profit
    # BLANK_START
    pass
    # BLANK_END
""", """def max_profit(prices: list[int]) -> int:
    min_price = float('inf')
    max_profit = 0

    for price in prices:
        min_price = min(min_price, price)
        max_profit = max(max_profit, price - min_price)

    return max_profit
""", """def test_max_profit_example1():
    from template import max_profit
    assert max_profit([7,1,5,3,6,4]) == 5

def test_max_profit_example2():
    from template import max_profit
    assert max_profit([7,6,4,3,1]) == 0

def test_max_profit_single():
    from template import max_profit
    assert max_profit([1]) == 0
"""),
        ("neetcode_longest_substring", "Longest Substring Without Repeating Characters", 3, 2, [], """
Longest Substring Without Repeating Characters (LeetCode 3)

Given a string s, find the length of the longest substring without repeating characters.

Example 1:
Input: s = "abcabcbb"
Output: 3

Example 2:
Input: s = "bbbbb"
Output: 1

Example 3:
Input: s = "pwwkew"
Output: 3

Constraints:
- 0 <= s.length <= 5 * 10^4
- s consists of English letters, digits, symbols and spaces
""", """def length_of_longest_substring(s: str) -> int:
    # TODO: Use sliding window with a set
    # BLANK_START
    pass
    # BLANK_END
""", """def length_of_longest_substring(s: str) -> int:
    char_set = set()
    left = 0
    max_length = 0

    for right in range(len(s)):
        while s[right] in char_set:
            char_set.remove(s[left])
            left += 1
        char_set.add(s[right])
        max_length = max(max_length, right - left + 1)

    return max_length
""", """def test_length_of_longest_substring_example1():
    from template import length_of_longest_substring
    assert length_of_longest_substring("abcabcbb") == 3

def test_length_of_longest_substring_example2():
    from template import length_of_longest_substring
    assert length_of_longest_substring("bbbbb") == 1

def test_length_of_longest_substring_example3():
    from template import length_of_longest_substring
    assert length_of_longest_substring("pwwkew") == 3

def test_length_of_longest_substring_empty():
    from template import length_of_longest_substring
    assert length_of_longest_substring("") == 0
"""),
    ],
    "neetcode_linked_list": [
        ("neetcode_reverse_linked_list", "Reverse Linked List", 206, 1, [], """
Reverse Linked List (LeetCode 206)

Given the head of a singly linked list, reverse the list, and return the reversed list.

Example 1:
Input: head = [1,2,3,4,5]
Output: [5,4,3,2,1]

Example 2:
Input: head = [1,2]
Output: [2,1]

Example 3:
Input: head = []
Output: []

Constraints:
- The number of nodes in the list is the range [0, 5000]
- -5000 <= Node.val <= 5000
""", """class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def reverse_list(head: ListNode | None) -> ListNode | None:
    # TODO: Reverse the linked list iteratively
    # BLANK_START
    pass
    # BLANK_END
""", """class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def reverse_list(head: ListNode | None) -> ListNode | None:
    prev = None
    current = head

    while current:
        next_node = current.next
        current.next = prev
        prev = current
        current = next_node

    return prev
""", """def test_reverse_list_multiple():
    from template import ListNode, reverse_list

    # Create list 1->2->3->4->5
    head = ListNode(1, ListNode(2, ListNode(3, ListNode(4, ListNode(5)))))
    reversed_head = reverse_list(head)

    # Check reversed list
    vals = []
    current = reversed_head
    while current:
        vals.append(current.val)
        current = current.next
    assert vals == [5, 4, 3, 2, 1]

def test_reverse_list_empty():
    from template import reverse_list
    assert reverse_list(None) is None

def test_reverse_list_single():
    from template import ListNode, reverse_list
    head = ListNode(1)
    reversed_head = reverse_list(head)
    assert reversed_head.val == 1
    assert reversed_head.next is None
"""),
        ("neetcode_merge_two_sorted_lists", "Merge Two Sorted Lists", 21, 1, [], """
Merge Two Sorted Lists (LeetCode 21)

You are given the heads of two sorted linked lists list1 and list2.

Merge the two lists into one sorted list. The list should be made by splicing together
the nodes of the first two lists.

Return the head of the merged linked list.

Example 1:
Input: list1 = [1,2,4], list2 = [1,3,4]
Output: [1,1,2,3,4,4]

Example 2:
Input: list1 = [], list2 = []
Output: []

Example 3:
Input: list1 = [], list2 = [0]
Output: [0]

Constraints:
- The number of nodes in both lists is in the range [0, 50]
- -100 <= Node.val <= 100
- Both list1 and list2 are sorted in non-decreasing order
""", """class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def merge_two_lists(list1: ListNode | None, list2: ListNode | None) -> ListNode | None:
    # TODO: Merge two sorted lists
    # BLANK_START
    pass
    # BLANK_END
""", """class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def merge_two_lists(list1: ListNode | None, list2: ListNode | None) -> ListNode | None:
    dummy = ListNode(0)
    current = dummy

    while list1 and list2:
        if list1.val < list2.val:
            current.next = list1
            list1 = list1.next
        else:
            current.next = list2
            list2 = list2.next
        current = current.next

    current.next = list1 if list1 else list2

    return dummy.next
""", """def test_merge_two_lists_example1():
    from template import ListNode, merge_two_lists

    list1 = ListNode(1, ListNode(2, ListNode(4)))
    list2 = ListNode(1, ListNode(3, ListNode(4)))
    merged = merge_two_lists(list1, list2)

    vals = []
    while merged:
        vals.append(merged.val)
        merged = merged.next
    assert vals == [1, 1, 2, 3, 4, 4]

def test_merge_two_lists_empty():
    from template import merge_two_lists
    assert merge_two_lists(None, None) is None

def test_merge_two_lists_one_empty():
    from template import ListNode, merge_two_lists
    list2 = ListNode(0)
    merged = merge_two_lists(None, list2)
    assert merged.val == 0
"""),
        ("neetcode_linked_list_cycle", "Linked List Cycle", 141, 1, [], """
Linked List Cycle (LeetCode 141)

Given head, the head of a linked list, determine if the linked list has a cycle in it.

Example 1:
Input: head = [3,2,0,-4], pos = 1
Output: true

Example 2:
Input: head = [1,2], pos = 0
Output: true

Example 3:
Input: head = [1], pos = -1
Output: false

Constraints:
- The number of the nodes in the list is in the range [0, 10^4]
- -10^5 <= Node.val <= 10^5
""", """class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def has_cycle(head: ListNode | None) -> bool:
    # TODO: Use fast and slow pointers (Floyd's algorithm)
    # BLANK_START
    pass
    # BLANK_END
""", """class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def has_cycle(head: ListNode | None) -> bool:
    if not head or not head.next:
        return False

    slow = head
    fast = head.next

    while slow != fast:
        if not fast or not fast.next:
            return False
        slow = slow.next
        fast = fast.next.next

    return True
""", """def test_has_cycle_with_cycle():
    from template import ListNode, has_cycle

    node1 = ListNode(3)
    node2 = ListNode(2)
    node3 = ListNode(0)
    node4 = ListNode(-4)
    node1.next = node2
    node2.next = node3
    node3.next = node4
    node4.next = node2  # cycle

    assert has_cycle(node1) == True

def test_has_cycle_no_cycle():
    from template import ListNode, has_cycle

    node1 = ListNode(1)
    node2 = ListNode(2)
    node1.next = node2

    assert has_cycle(node1) == False

def test_has_cycle_single():
    from template import ListNode, has_cycle
    assert has_cycle(ListNode(1)) == False
"""),
        ("neetcode_reorder_list", "Reorder List", 143, 2, ["neetcode_reverse_linked_list"], """
Reorder List (LeetCode 143)

You are given the head of a singly linked-list. Reorder the list to be in the form:
L0 → Ln → L1 → Ln-1 → L2 → Ln-2 → ...

Example 1:
Input: head = [1,2,3,4]
Output: [1,4,2,3]

Example 2:
Input: head = [1,2,3,4,5]
Output: [1,5,2,4,3]

Constraints:
- The number of nodes in the list is in the range [1, 5 * 10^4]
- 1 <= Node.val <= 1000
""", """class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def reorder_list(head: ListNode | None) -> None:
    # TODO: Find middle, reverse second half, merge alternating
    # BLANK_START
    pass
    # BLANK_END
""", """class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def reorder_list(head: ListNode | None) -> None:
    if not head or not head.next:
        return

    # Find middle
    slow, fast = head, head
    while fast.next and fast.next.next:
        slow = slow.next
        fast = fast.next.next

    # Reverse second half
    second = slow.next
    slow.next = None
    prev = None
    while second:
        tmp = second.next
        second.next = prev
        prev = second
        second = tmp

    # Merge two halves
    first, second = head, prev
    while second:
        tmp1, tmp2 = first.next, second.next
        first.next = second
        second.next = tmp1
        first, second = tmp1, tmp2
""", """def test_reorder_list_even():
    from template import ListNode, reorder_list

    head = ListNode(1, ListNode(2, ListNode(3, ListNode(4))))
    reorder_list(head)

    vals = []
    current = head
    while current:
        vals.append(current.val)
        current = current.next
    assert vals == [1, 4, 2, 3]

def test_reorder_list_odd():
    from template import ListNode, reorder_list

    head = ListNode(1, ListNode(2, ListNode(3, ListNode(4, ListNode(5)))))
    reorder_list(head)

    vals = []
    current = head
    while current:
        vals.append(current.val)
        current = current.next
    assert vals == [1, 5, 2, 4, 3]
"""),
    ],
    "neetcode_stack": [
        ("neetcode_valid_parentheses", "Valid Parentheses", 20, 1, [], """
Valid Parentheses (LeetCode 20)

Given a string s containing just the characters '(', ')', '{', '}', '[' and ']',
determine if the input string is valid.

Example 1:
Input: s = "()"
Output: true

Example 2:
Input: s = "()[]{}"
Output: true

Example 3:
Input: s = "(]"
Output: false

Constraints:
- 1 <= s.length <= 10^4
- s consists of parentheses only '()[]{}'
""", """def is_valid(s: str) -> bool:
    # TODO: Use a stack to match brackets
    # BLANK_START
    pass
    # BLANK_END
""", """def is_valid(s: str) -> bool:
    stack = []
    mapping = {')': '(', '}': '{', ']': '['}

    for char in s:
        if char in mapping:
            if not stack or stack[-1] != mapping[char]:
                return False
            stack.pop()
        else:
            stack.append(char)

    return len(stack) == 0
""", """def test_is_valid_simple():
    from template import is_valid
    assert is_valid("()") == True

def test_is_valid_multiple():
    from template import is_valid
    assert is_valid("()[]{}") == True

def test_is_valid_invalid():
    from template import is_valid
    assert is_valid("(]") == False

def test_is_valid_nested():
    from template import is_valid
    assert is_valid("{[]}") == True
"""),
        ("neetcode_min_stack", "Min Stack", 155, 2, [], """
Min Stack (LeetCode 155)

Design a stack that supports push, pop, top, and retrieving the minimum element in constant time.

Implement the MinStack class:
- MinStack() initializes the stack object
- void push(int val) pushes the element val onto the stack
- void pop() removes the element on top of the stack
- int top() gets the top element of the stack
- int getMin() retrieves the minimum element in the stack

Example 1:
Input: ["MinStack","push","push","push","getMin","pop","top","getMin"]
[[],[-2],[0],[-3],[],[],[],[]]
Output: [null,null,null,null,-3,null,0,-2]

Constraints:
- -2^31 <= val <= 2^31 - 1
- pop, top and getMin will always be called on non-empty stacks
""", """class MinStack:
    def __init__(self):
        # TODO: Track both values and minimum at each level
        # BLANK_START
        pass
        # BLANK_END

    def push(self, val: int) -> None:
        # BLANK_START
        pass
        # BLANK_END

    def pop(self) -> None:
        # BLANK_START
        pass
        # BLANK_END

    def top(self) -> int:
        # BLANK_START
        pass
        # BLANK_END

    def get_min(self) -> int:
        # BLANK_START
        pass
        # BLANK_END
""", """class MinStack:
    def __init__(self):
        self.stack = []
        self.min_stack = []

    def push(self, val: int) -> None:
        self.stack.append(val)
        min_val = min(val, self.min_stack[-1] if self.min_stack else val)
        self.min_stack.append(min_val)

    def pop(self) -> None:
        self.stack.pop()
        self.min_stack.pop()

    def top(self) -> int:
        return self.stack[-1]

    def get_min(self) -> int:
        return self.min_stack[-1]
""", """def test_min_stack():
    from template import MinStack

    stack = MinStack()
    stack.push(-2)
    stack.push(0)
    stack.push(-3)
    assert stack.get_min() == -3
    stack.pop()
    assert stack.top() == 0
    assert stack.get_min() == -2
"""),
        ("neetcode_evaluate_rpn", "Evaluate Reverse Polish Notation", 150, 2, [], """
Evaluate Reverse Polish Notation (LeetCode 150)

Evaluate the value of an arithmetic expression in Reverse Polish Notation.

Example 1:
Input: tokens = ["2","1","+","3","*"]
Output: 9
Explanation: ((2 + 1) * 3) = 9

Example 2:
Input: tokens = ["4","13","5","/","+"]
Output: 6

Constraints:
- 1 <= tokens.length <= 10^4
- tokens[i] is either an operator: "+", "-", "*", or "/", or an integer in the range [-200, 200]
""", """def eval_rpn(tokens: list[str]) -> int:
    # TODO: Use a stack to evaluate
    # BLANK_START
    pass
    # BLANK_END
""", """def eval_rpn(tokens: list[str]) -> int:
    stack = []

    for token in tokens:
        if token in "+-*/":
            b = stack.pop()
            a = stack.pop()
            if token == '+':
                stack.append(a + b)
            elif token == '-':
                stack.append(a - b)
            elif token == '*':
                stack.append(a * b)
            else:
                stack.append(int(a / b))
        else:
            stack.append(int(token))

    return stack[0]
""", """def test_eval_rpn_example1():
    from template import eval_rpn
    assert eval_rpn(["2","1","+","3","*"]) == 9

def test_eval_rpn_example2():
    from template import eval_rpn
    assert eval_rpn(["4","13","5","/","+"]) == 6

def test_eval_rpn_single():
    from template import eval_rpn
    assert eval_rpn(["42"]) == 42
"""),
        ("neetcode_generate_parentheses", "Generate Parentheses", 22, 2, [], """
Generate Parentheses (LeetCode 22)

Given n pairs of parentheses, write a function to generate all combinations of well-formed parentheses.

Example 1:
Input: n = 3
Output: ["((()))","(()())","(())()","()(())","()()()"]

Example 2:
Input: n = 1
Output: ["()"]

Constraints:
- 1 <= n <= 8
""", """def generate_parenthesis(n: int) -> list[str]:
    # TODO: Use backtracking with open/close count
    # BLANK_START
    pass
    # BLANK_END
""", """def generate_parenthesis(n: int) -> list[str]:
    result = []

    def backtrack(current, open_count, close_count):
        if len(current) == 2 * n:
            result.append(current)
            return

        if open_count < n:
            backtrack(current + '(', open_count + 1, close_count)
        if close_count < open_count:
            backtrack(current + ')', open_count, close_count + 1)

    backtrack('', 0, 0)
    return result
""", """def test_generate_parenthesis_n3():
    from template import generate_parenthesis
    result = sorted(generate_parenthesis(3))
    expected = sorted(["((()))","(()())","(())()","()(())","()()()"])
    assert result == expected

def test_generate_parenthesis_n1():
    from template import generate_parenthesis
    assert generate_parenthesis(1) == ["()"]
"""),
        ("neetcode_daily_temperatures", "Daily Temperatures", 739, 2, [], """
Daily Temperatures (LeetCode 739)

Given an array of integers temperatures represents the daily temperatures,
return an array answer such that answer[i] is the number of days you have to wait
after the ith day to get a warmer temperature.

Example 1:
Input: temperatures = [73,74,75,71,69,72,76,73]
Output: [1,1,4,2,1,1,0,0]

Example 2:
Input: temperatures = [30,40,50,60]
Output: [1,1,1,0]

Constraints:
- 1 <= temperatures.length <= 10^5
- 30 <= temperatures[i] <= 100
""", """def daily_temperatures(temperatures: list[int]) -> list[int]:
    # TODO: Use a monotonic decreasing stack
    # BLANK_START
    pass
    # BLANK_END
""", """def daily_temperatures(temperatures: list[int]) -> list[int]:
    n = len(temperatures)
    result = [0] * n
    stack = []

    for i in range(n):
        while stack and temperatures[i] > temperatures[stack[-1]]:
            prev_i = stack.pop()
            result[prev_i] = i - prev_i
        stack.append(i)

    return result
""", """def test_daily_temperatures_example1():
    from template import daily_temperatures
    assert daily_temperatures([73,74,75,71,69,72,76,73]) == [1,1,4,2,1,1,0,0]

def test_daily_temperatures_example2():
    from template import daily_temperatures
    assert daily_temperatures([30,40,50,60]) == [1,1,1,0]

def test_daily_temperatures_decreasing():
    from template import daily_temperatures
    assert daily_temperatures([60,50,40,30]) == [0,0,0,0]
"""),
        ("neetcode_car_fleet", "Car Fleet", 853, 2, [], """
Car Fleet (LeetCode 853)

There are n cars at given miles from the starting mile 0, traveling to reach the mile target.

You are given two integer arrays position and speed. The i-th car starts at position[i] miles
and travels at speed[i] miles per hour. A car can never pass another car, but it can catch up to it.

Return the number of car fleets that will arrive at the destination.

Example 1:
Input: target = 12, position = [10,8,0,5,3], speed = [2,4,1,1,3]
Output: 3

Constraints:
- n == position.length == speed.length
- 1 <= n <= 10^5
""", """def car_fleet(target: int, position: list[int], speed: list[int]) -> int:
    # TODO: Sort by position, calculate time to target, use stack
    # BLANK_START
    pass
    # BLANK_END
""", """def car_fleet(target: int, position: list[int], speed: list[int]) -> int:
    cars = sorted(zip(position, speed), reverse=True)
    stack = []

    for pos, spd in cars:
        time = (target - pos) / spd
        if not stack or time > stack[-1]:
            stack.append(time)

    return len(stack)
""", """def test_car_fleet_example1():
    from template import car_fleet
    assert car_fleet(12, [10,8,0,5,3], [2,4,1,1,3]) == 3

def test_car_fleet_single():
    from template import car_fleet
    assert car_fleet(10, [0], [5]) == 1
"""),
        ("neetcode_largest_rectangle", "Largest Rectangle in Histogram", 84, 3, [], """
Largest Rectangle in Histogram (LeetCode 84)

Given an array of integers heights representing the histogram's bar height where the width of each bar is 1,
return the area of the largest rectangle in the histogram.

Example 1:
Input: heights = [2,1,5,6,2,3]
Output: 10

Example 2:
Input: heights = [2,4]
Output: 4

Constraints:
- 1 <= heights.length <= 10^5
- 0 <= heights[i] <= 10^4
""", """def largest_rectangle_area(heights: list[int]) -> int:
    # TODO: Use a monotonic increasing stack
    # BLANK_START
    pass
    # BLANK_END
""", """def largest_rectangle_area(heights: list[int]) -> int:
    stack = []
    max_area = 0
    heights.append(0)

    for i, h in enumerate(heights):
        while stack and heights[stack[-1]] > h:
            height_idx = stack.pop()
            width = i if not stack else i - stack[-1] - 1
            max_area = max(max_area, heights[height_idx] * width)
        stack.append(i)

    heights.pop()
    return max_area
""", """def test_largest_rectangle_area_example1():
    from template import largest_rectangle_area
    assert largest_rectangle_area([2,1,5,6,2,3]) == 10

def test_largest_rectangle_area_example2():
    from template import largest_rectangle_area
    assert largest_rectangle_area([2,4]) == 4
"""),
    ],
    "neetcode_binary_search": [
        ("neetcode_binary_search", "Binary Search", 704, 1, [], """
Binary Search (LeetCode 704)

Given an array of integers nums which is sorted in ascending order, and an integer target,
write a function to search target in nums. If target exists, then return its index. Otherwise, return -1.

Example 1:
Input: nums = [-1,0,3,5,9,12], target = 9
Output: 4

Example 2:
Input: nums = [-1,0,3,5,9,12], target = 2
Output: -1

Constraints:
- 1 <= nums.length <= 10^4
- -10^4 < nums[i], target < 10^4
- All integers in nums are unique
- nums is sorted in ascending order
""", """def search(nums: list[int], target: int) -> int:
    # TODO: Implement binary search
    # BLANK_START
    pass
    # BLANK_END
""", """def search(nums: list[int], target: int) -> int:
    left, right = 0, len(nums) - 1

    while left <= right:
        mid = (left + right) // 2
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1
""", """def test_search_found():
    from template import search
    assert search([-1,0,3,5,9,12], 9) == 4

def test_search_not_found():
    from template import search
    assert search([-1,0,3,5,9,12], 2) == -1

def test_search_single():
    from template import search
    assert search([5], 5) == 0
"""),
        ("neetcode_search_2d_matrix", "Search a 2D Matrix", 74, 2, ["neetcode_binary_search"], """
Search a 2D Matrix (LeetCode 74)

You are given an m x n integer matrix with the following two properties:
- Each row is sorted in non-decreasing order
- The first integer of each row is greater than the last integer of the previous row

Given an integer target, return true if target is in matrix or false otherwise.

Example 1:
Input: matrix = [[1,3,5,7],[10,11,16,20],[23,30,34,60]], target = 3
Output: true

Example 2:
Input: matrix = [[1,3,5,7],[10,11,16,20],[23,30,34,60]], target = 13
Output: false

Constraints:
- m == matrix.length
- n == matrix[i].length
- 1 <= m, n <= 100
""", """def search_matrix(matrix: list[list[int]], target: int) -> bool:
    # TODO: Treat 2D matrix as 1D sorted array
    # BLANK_START
    pass
    # BLANK_END
""", """def search_matrix(matrix: list[list[int]], target: int) -> bool:
    if not matrix or not matrix[0]:
        return False

    m, n = len(matrix), len(matrix[0])
    left, right = 0, m * n - 1

    while left <= right:
        mid = (left + right) // 2
        mid_val = matrix[mid // n][mid % n]

        if mid_val == target:
            return True
        elif mid_val < target:
            left = mid + 1
        else:
            right = mid - 1

    return False
""", """def test_search_matrix_found():
    from template import search_matrix
    matrix = [[1,3,5,7],[10,11,16,20],[23,30,34,60]]
    assert search_matrix(matrix, 3) == True

def test_search_matrix_not_found():
    from template import search_matrix
    matrix = [[1,3,5,7],[10,11,16,20],[23,30,34,60]]
    assert search_matrix(matrix, 13) == False
"""),
        ("neetcode_search_rotated", "Search in Rotated Sorted Array", 33, 2, ["neetcode_binary_search"], """
Search in Rotated Sorted Array (LeetCode 33)

There is an integer array nums sorted in ascending order (with distinct values).
The array is rotated at an unknown pivot. Given the array after rotation and an integer target,
return the index of target if it is in nums, or -1 if it is not in nums.

Example 1:
Input: nums = [4,5,6,7,0,1,2], target = 0
Output: 4

Example 2:
Input: nums = [4,5,6,7,0,1,2], target = 3
Output: -1

Constraints:
- 1 <= nums.length <= 5000
- -10^4 <= nums[i] <= 10^4
- All values of nums are unique
""", """def search(nums: list[int], target: int) -> int:
    # TODO: Binary search with rotation check
    # BLANK_START
    pass
    # BLANK_END
""", """def search(nums: list[int], target: int) -> int:
    left, right = 0, len(nums) - 1

    while left <= right:
        mid = (left + right) // 2

        if nums[mid] == target:
            return mid

        # Left half is sorted
        if nums[left] <= nums[mid]:
            if nums[left] <= target < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
        # Right half is sorted
        else:
            if nums[mid] < target <= nums[right]:
                left = mid + 1
            else:
                right = mid - 1

    return -1
""", """def test_search_rotated_found():
    from template import search
    assert search([4,5,6,7,0,1,2], 0) == 4

def test_search_rotated_not_found():
    from template import search
    assert search([4,5,6,7,0,1,2], 3) == -1

def test_search_rotated_single():
    from template import search
    assert search([1], 1) == 0
"""),
        ("neetcode_find_min_rotated", "Find Minimum in Rotated Sorted Array", 153, 2, ["neetcode_binary_search"], """
Find Minimum in Rotated Sorted Array (LeetCode 153)

Suppose an array of length n sorted in ascending order is rotated between 1 and n times.
Given the sorted rotated array nums of unique elements, return the minimum element of this array.

Example 1:
Input: nums = [3,4,5,1,2]
Output: 1

Example 2:
Input: nums = [4,5,6,7,0,1,2]
Output: 0

Constraints:
- n == nums.length
- 1 <= n <= 5000
- -5000 <= nums[i] <= 5000
- All integers of nums are unique
""", """def find_min(nums: list[int]) -> int:
    # TODO: Binary search for minimum
    # BLANK_START
    pass
    # BLANK_END
""", """def find_min(nums: list[int]) -> int:
    left, right = 0, len(nums) - 1

    while left < right:
        mid = (left + right) // 2

        if nums[mid] > nums[right]:
            left = mid + 1
        else:
            right = mid

    return nums[left]
""", """def test_find_min_example1():
    from template import find_min
    assert find_min([3,4,5,1,2]) == 1

def test_find_min_example2():
    from template import find_min
    assert find_min([4,5,6,7,0,1,2]) == 0

def test_find_min_no_rotation():
    from template import find_min
    assert find_min([1,2,3,4,5]) == 1
"""),
    ],
}


def create_kata(kata_dir, name, title, leetcode_num, difficulty, deps, desc, template_code, reference_code, test_code):
    """Create a single kata with all required files."""
    kata_path = Path(kata_dir) / name
    kata_path.mkdir(parents=True, exist_ok=True)

    # Create manifest.toml
    deps_str = ', '.join([f'"{d}"' for d in deps])
    manifest = f"""[kata]
name = "{name}"
category = "{Path(kata_dir).name}"
base_difficulty = {difficulty}
description = \"\"\"{desc.strip()}
\"\"\"
dependencies = [{deps_str}]
"""
    (kata_path / "manifest.toml").write_text(manifest)

    # Create template.py
    template = f'''"""{title} - LeetCode {leetcode_num}"""

{template_code.strip()}
'''
    (kata_path / "template.py").write_text(template)

    # Create reference.py
    reference = f'''"""{title} - LeetCode {leetcode_num} - Reference Solution"""

{reference_code.strip()}
'''
    (kata_path / "reference.py").write_text(reference)

    # Create test_kata.py
    test = f'''"""Tests for {title} kata."""

{test_code.strip()}
'''
    (kata_path / "test_kata.py").write_text(test)

    print(f"Created kata: {name}")


def main():
    """Generate all NeetCode 150 katas."""
    exercises_dir = Path("katas/exercises")

    for category, katas in NEETCODE_150.items():
        print(f"\nGenerating {category} katas...")
        for kata_data in katas:
            create_kata(exercises_dir, *kata_data)

    print("\nAll katas generated successfully!")


if __name__ == "__main__":
    main()
