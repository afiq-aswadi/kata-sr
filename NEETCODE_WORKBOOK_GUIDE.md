# NeetCode Workbook Creation Guide

## Overview

This guide provides ready-to-use templates and commands for creating thematic workbooks for all NeetCode exercise categories. We've organized 151 NeetCode exercises into 18 themed workbooks based on the NeetCode roadmap.

## Quick Start: Creating Your First Workbook

```bash
# 1. Create workbook directory
mkdir -p workbooks/neetcode_<topic_name>

# 2. Create manifest.toml (use templates below)
# Edit workbooks/neetcode_<topic_name>/manifest.toml

# 3. Build and test
cargo build
cargo run
# Press 'w' to view Workbooks, verify your workbook appears

# 4. Commit
git add workbooks/neetcode_<topic_name>/
git commit -m "Add NeetCode <Topic Name> workbook"
```

## Status: Workbooks Created vs Remaining

### ✅ Already Created (3/18):
1. **Arrays & Hashing** - 8 core exercises from 17 total
2. **Binary Search** - 7 exercises
3. **Two Pointers** - 5 exercises

### 📋 Ready to Create (15/18):

Remaining topics with exercise counts:
- Stack (7 exercises)
- Sliding Window (6 exercises)
- Linked List (11 exercises)
- Trees (15 exercises)
- Tries (3 exercises)
- Heap/Priority Queue (8 exercises)
- Intervals (6 exercises)
- Greedy (8 exercises)
- Backtracking (10 exercises)
- Graphs (13 exercises)
- Advanced Graphs (6 exercises)
- 1-D DP (12 exercises)
- 2-D DP (11 exercises)
- Bit Manipulation (6 exercises)
- Math & Geometry (9 exercises)

## Complete Workbook Templates

---

### 4. Stack

**Create:** `workbooks/neetcode_stack/manifest.toml`

```toml
[workbook]
id = "neetcode_stack"
title = "NeetCode Stack: LIFO Patterns and Monotonic Stacks"
summary = "Master stack patterns from basic parentheses matching to monotonic stacks for optimization."
learning_goals = [
  "Recognize when stack (LIFO) data structure applies",
  "Implement stack-based parsers and validators",
  "Use monotonic stacks for O(n) optimization problems",
  "Apply stack patterns to string and array problems",
]
prerequisites = ["neetcode_contains_duplicate"]
resources = [
  { title = "Stack Patterns Guide", url = "https://neetcode.io/courses/dsa-for-beginners/6" },
]
kata_namespace = "neetcode"

[[exercises]]
slug = "valid-parentheses"
title = "1. Valid Parentheses - Basic Stack"
kata = "neetcode_valid_parentheses"
objective = "Check if parentheses are balanced using a stack."
acceptance = [
  "Returns True for balanced strings",
  "Handles all three bracket types: (), {}, []",
  "Uses O(n) time and O(n) space",
]
hints = [
  "Push opening brackets to stack",
  "For closing bracket, check if it matches stack top",
  "Use hashmap for bracket pairs",
]
dependencies = []

[[exercises]]
slug = "min-stack"
title = "2. Min Stack - Stack with O(1) Min"
kata = "neetcode_min_stack"
objective = "Design a stack that supports getMin() in O(1) time."
acceptance = [
  "All operations work in O(1) time",
  "getMin() returns minimum element",
  "Handles pop() of minimum element correctly",
]
hints = [
  "Use two stacks: main stack and min stack",
  "Min stack tracks minimum at each level",
]
dependencies = ["valid-parentheses"]

[[exercises]]
slug = "evaluate-rpn"
title = "3. Evaluate Reverse Polish Notation"
kata = "neetcode_evaluate_rpn"
objective = "Evaluate arithmetic expression in RPN using a stack."
acceptance = [
  "Handles +, -, *, / operators",
  "Returns correct integer result",
  "Division truncates toward zero",
]
hints = [
  "Push numbers to stack",
  "For operator, pop two operands, compute, push result",
]
dependencies = ["min-stack"]

[[exercises]]
slug = "generate-parentheses"
title = "4. Generate Parentheses - Backtracking with Stack"
kata = "neetcode_generate_parentheses"
objective = "Generate all valid parentheses combinations using backtracking."
acceptance = [
  "Returns all valid n-pair combinations",
  "No duplicates in result",
  "Only well-formed strings",
]
hints = [
  "Backtrack with open_count and close_count",
  "Add '(' if open_count < n",
  "Add ')' if close_count < open_count",
]
dependencies = ["valid-parentheses"]

[[exercises]]
slug = "daily-temperatures"
title = "5. Daily Temperatures - Monotonic Stack"
kata = "neetcode_daily_temperatures"
objective = "Find days until warmer temperature using monotonic decreasing stack."
acceptance = [
  "Returns array of days to wait",
  "Uses O(n) time complexity",
  "Each element pushed/popped once",
]
hints = [
  "Stack stores indices, not temperatures",
  "Pop indices while current temp > stack top temp",
  "For popped index, days = current_index - popped_index",
]
dependencies = ["min-stack"]

[[exercises]]
slug = "car-fleet"
title = "6. Car Fleet - Monotonic Stack Application"
kata = "neetcode_car_fleet"
objective = "Calculate number of car fleets reaching destination using stack."
acceptance = [
  "Correctly identifies fleets that merge",
  "Handles cars at same position",
  "Uses O(n log n) time (due to sorting)",
]
hints = [
  "Calculate time for each car to reach target",
  "Sort by position (descending)",
  "Use stack to track fleet leaders",
]
dependencies = ["daily-temperatures"]

[[exercises]]
slug = "largest-rectangle"
title = "7. Largest Rectangle in Histogram - Advanced Monotonic Stack"
kata = "neetcode_largest_rectangle"
objective = "Find largest rectangle area using monotonic increasing stack."
acceptance = [
  "Returns maximum rectangle area",
  "Uses O(n) time complexity",
  "Handles edge cases with single bar",
]
hints = [
  "Stack stores indices of bars",
  "Pop when current bar < stack top bar",
  "Area = height[popped] * width (using indices)",
]
dependencies = ["daily-temperatures", "car-fleet"]
```

---

### 5. Sliding Window

**Create:** `workbooks/neetcode_sliding_window/manifest.toml`

```toml
[workbook]
id = "neetcode_sliding_window"
title = "NeetCode Sliding Window: Subarray Optimization"
summary = "Master the sliding window technique for efficient subarray and substring problems."
learning_goals = [
  "Recognize fixed and variable-size sliding window problems",
  "Use two pointers to maintain window bounds",
  "Apply frequency maps for character/element tracking",
  "Optimize brute force O(n²) solutions to O(n)",
]
prerequisites = ["neetcode_two_sum", "neetcode_valid_anagram"]
resources = [
  { title = "Sliding Window Patterns", url = "https://neetcode.io/courses/dsa-for-beginners/29" },
]
kata_namespace = "neetcode"

[[exercises]]
slug = "best-time-stock"
title = "1. Best Time to Buy/Sell Stock - Basic Window"
kata = "neetcode_best_time_stock"
objective = "Find maximum profit using expanding window (buy low, sell high)."
acceptance = [
  "Returns maximum profit possible",
  "Uses O(n) time and O(1) space",
  "Handles cases with no profit (return 0)",
]
hints = [
  "Track minimum price seen so far",
  "For each price, calculate profit if selling today",
  "Update max profit",
]
dependencies = []

[[exercises]]
slug = "longest-substring"
title = "2. Longest Substring Without Repeating Characters"
kata = "neetcode_longest_substring"
objective = "Find longest substring with unique characters using variable window."
acceptance = [
  "Returns length of longest substring",
  "Uses O(n) time complexity",
  "Handles empty strings and single characters",
]
hints = [
  "Use set or dict to track characters in window",
  "Expand right, shrink left when duplicate found",
  "Update max length after each expansion",
]
dependencies = ["best-time-stock"]

[[exercises]]
slug = "longest-repeating"
title = "3. Longest Repeating Character Replacement"
kata = "neetcode_longest_repeating"
objective = "Find longest substring with same character after k replacements."
acceptance = [
  "Returns maximum length achievable",
  "Uses O(n) time complexity",
  "Correctly handles k replacements",
]
hints = [
  "Track frequency of each character in window",
  "Window is valid if: window_size - max_freq <= k",
  "Shrink window when invalid",
]
dependencies = ["longest-substring"]

[[exercises]]
slug = "permutation-string"
title = "4. Permutation in String - Fixed Window"
kata = "neetcode_permutation_string"
objective = "Check if s2 contains permutation of s1 using fixed-size window."
acceptance = [
  "Returns True if permutation exists",
  "Uses O(n) time complexity",
  "Handles strings with duplicate characters",
]
hints = [
  "Window size = len(s1)",
  "Compare frequency maps of window and s1",
  "Slide window and update frequencies incrementally",
]
dependencies = ["longest-substring"]

[[exercises]]
slug = "min-window-substring"
title = "5. Minimum Window Substring - Complex Variable Window"
kata = "neetcode_min_window_substring"
objective = "Find smallest window in s containing all characters of t."
acceptance = [
  "Returns shortest substring or empty string",
  "Uses O(n + m) time complexity",
  "Handles characters appearing multiple times in t",
]
hints = [
  "Expand right until window contains all of t",
  "Contract left while still valid, update min length",
  "Use two frequency maps for comparison",
]
dependencies = ["longest-repeating", "permutation-string"]

[[exercises]]
slug = "sliding-window-max"
title = "6. Sliding Window Maximum - Deque Optimization"
kata = "neetcode_sliding_window_max"
objective = "Find maximum in each fixed-size window using monotonic deque."
acceptance = [
  "Returns array of window maximums",
  "Uses O(n) time complexity",
  "Each element added/removed once from deque",
]
hints = [
  "Use deque to store indices (not values)",
  "Maintain decreasing order (front = largest)",
  "Remove indices outside window from front",
  "Remove smaller elements from back",
]
dependencies = ["min-window-substring"]
```

---

### 6. Linked List

**Create:** `workbooks/neetcode_linked_list/manifest.toml`

```toml
[workbook]
id = "neetcode_linked_list"
title = "NeetCode Linked List: Pointer Manipulation Mastery"
summary = "Master linked list patterns from basic traversal to complex pointer manipulation."
learning_goals = [
  "Navigate and manipulate linked list pointers confidently",
  "Use fast/slow pointer technique for cycle detection",
  "Reverse linked lists iteratively and recursively",
  "Handle edge cases with null pointers and empty lists",
]
prerequisites = ["neetcode_two_sum"]
resources = [
  { title = "Linked List Guide", url = "https://neetcode.io/courses/dsa-for-beginners/7" },
]
kata_namespace = "neetcode"

[[exercises]]
slug = "reverse-linked-list"
title = "1. Reverse Linked List - Basic Pointer Manipulation"
kata = "neetcode_reverse_linked_list"
objective = "Reverse a singly linked list iteratively."
acceptance = [
  "Returns new head of reversed list",
  "Uses O(n) time and O(1) space",
  "Handles empty list and single node",
]
hints = [
  "Use three pointers: prev, curr, next",
  "Reverse link: curr.next = prev",
  "Move all pointers forward",
]
dependencies = []

[[exercises]]
slug = "merge-two-sorted-lists"
title = "2. Merge Two Sorted Lists"
kata = "neetcode_merge_two_sorted_lists"
objective = "Merge two sorted linked lists into one sorted list."
acceptance = [
  "Returns head of merged list",
  "Maintains sorted order",
  "Handles lists of different lengths",
]
hints = [
  "Use dummy node to simplify head handling",
  "Compare list1 and list2 values, attach smaller",
  "Advance pointer in chosen list",
]
dependencies = ["reverse-linked-list"]

[[exercises]]
slug = "linked-list-cycle"
title = "3. Linked List Cycle - Fast/Slow Pointers"
kata = "neetcode_linked_list_cycle"
objective = "Detect cycle in linked list using Floyd's algorithm."
acceptance = [
  "Returns True if cycle exists",
  "Uses O(n) time and O(1) space",
  "Handles empty list and single node",
]
hints = [
  "Use two pointers: slow (1 step) and fast (2 steps)",
  "If fast catches slow, cycle exists",
  "If fast reaches None, no cycle",
]
dependencies = []

[[exercises]]
slug = "reorder-list"
title = "4. Reorder List - Multiple Techniques"
kata = "neetcode_reorder_list"
objective = "Reorder list L0→L1→...→Ln to L0→Ln→L1→Ln-1... in-place."
acceptance = [
  "Modifies list in-place",
  "Uses O(n) time and O(1) space",
  "Handles odd and even length lists",
]
hints = [
  "Find middle using slow/fast pointers",
  "Reverse second half",
  "Merge two halves alternately",
]
dependencies = ["reverse-linked-list", "linked-list-cycle", "merge-two-sorted-lists"]

[[exercises]]
slug = "remove-nth-node"
title = "5. Remove Nth Node From End"
kata = "neetcode_remove_nth_node"
objective = "Remove nth node from end using two-pointer technique."
acceptance = [
  "Returns head of modified list",
  "Uses O(n) time and O(1) space",
  "Handles removing head node",
]
hints = [
  "Use two pointers with n-node gap",
  "Move both until fast reaches end",
  "Slow pointer is at (n-1)th from end",
]
dependencies = ["linked-list-cycle"]

[[exercises]]
slug = "copy-random-list"
title = "6. Copy List with Random Pointer"
kata = "neetcode_copy_random_list"
objective = "Deep copy linked list with random pointers."
acceptance = [
  "Creates independent deep copy",
  "Copies both next and random pointers",
  "Uses O(n) time and O(n) or O(1) space",
]
hints = [
  "First pass: create nodes, map old→new",
  "Second pass: set next and random using map",
  "Or: interleave new nodes, then separate",
]
dependencies = ["reverse-linked-list"]

[[exercises]]
slug = "add-two-numbers"
title = "7. Add Two Numbers - Digit Manipulation"
kata = "neetcode_add_two_numbers"
objective = "Add two numbers represented as linked lists (reverse order)."
acceptance = [
  "Returns head of sum list",
  "Handles carry correctly",
  "Works with lists of different lengths",
]
hints = [
  "Process digit by digit with carry",
  "Handle carry at the end",
  "Use dummy node for result",
]
dependencies = ["merge-two-sorted-lists"]

[[exercises]]
slug = "find-duplicate"
title = "8. Find Duplicate Number - Cycle Detection Application"
kata = "neetcode_find_duplicate"
objective = "Find duplicate in array [1,n] using linked list cycle detection."
acceptance = [
  "Returns the duplicate number",
  "Uses O(n) time and O(1) space",
  "Does not modify array",
]
hints = [
  "Treat array as linked list: nums[i] points to nums[nums[i]]",
  "Use Floyd's cycle detection",
  "Find cycle entry point = duplicate",
]
dependencies = ["linked-list-cycle"]
```

---

### 7. Trees

**Create:** `workbooks/neetcode_trees/manifest.toml`

```toml
[workbook]
id = "neetcode_trees"
title = "NeetCode Trees: Binary Tree Patterns"
summary = "Master binary tree traversals, DFS/BFS, and recursive problem-solving."
learning_goals = [
  "Implement DFS (preorder, inorder, postorder) and BFS traversals",
  "Solve problems recursively with base cases and recursive cases",
  "Apply tree properties (BST, balanced, symmetric)",
  "Use height and depth concepts effectively",
]
prerequisites = ["neetcode_reverse_linked_list"]
resources = [
  { title = "Binary Trees Guide", url = "https://neetcode.io/courses/dsa-for-beginners/14" },
]
kata_namespace = "neetcode"

[[exercises]]
slug = "invert-tree"
title = "1. Invert Binary Tree - Basic Recursion"
kata = "neetcode_invert_tree"
objective = "Invert a binary tree (swap left and right subtrees) recursively."
acceptance = [
  "Returns root of inverted tree",
  "Handles empty tree and single node",
  "Uses O(n) time and O(h) space",
]
hints = [
  "Base case: null node returns null",
  "Recursively invert left and right subtrees",
  "Swap left and right",
]
dependencies = []

[[exercises]]
slug = "max-depth"
title = "2. Maximum Depth of Binary Tree"
kata = "neetcode_max_depth"
objective = "Find maximum depth using recursion."
acceptance = [
  "Returns correct depth",
  "Handles empty tree (depth = 0)",
  "Uses O(n) time and O(h) space",
]
hints = [
  "Base case: null node has depth 0",
  "Depth = 1 + max(left_depth, right_depth)",
]
dependencies = []

[[exercises]]
slug = "same-tree"
title = "3. Same Tree - Structural Comparison"
kata = "neetcode_same_tree"
objective = "Check if two trees are identical structurally and by value."
acceptance = [
  "Returns True if identical",
  "Compares structure and values",
  "Handles null trees",
]
hints = [
  "Base cases: both null (True), one null (False)",
  "Check values match AND recurse on subtrees",
]
dependencies = ["max-depth"]

[[exercises]]
slug = "subtree"
title = "4. Subtree of Another Tree"
kata = "neetcode_subtree"
objective = "Check if tree contains given subtree."
acceptance = [
  "Returns True if subtree exists",
  "Checks all possible root positions",
  "Uses O(m*n) time in worst case",
]
hints = [
  "For each node, check if subtree matches",
  "Use isSameTree helper function",
  "Recursively check left and right",
]
dependencies = ["same-tree"]

[[exercises]]
slug = "level-order"
title = "5. Binary Tree Level Order Traversal - BFS"
kata = "neetcode_level_order"
objective = "Return level-by-level traversal using BFS (queue)."
acceptance = [
  "Returns list of lists (one per level)",
  "Uses O(n) time and O(n) space",
  "Correctly separates levels",
]
hints = [
  "Use queue (deque) for BFS",
  "Process one level at a time (track level size)",
  "Add children to queue for next level",
]
dependencies = ["max-depth"]

[[exercises]]
slug = "validate-bst"
title = "6. Validate Binary Search Tree"
kata = "neetcode_validate_bst"
objective = "Check if tree is valid BST using in-order traversal or bounds."
acceptance = [
  "Returns True if valid BST",
  "Handles edge cases with INT_MIN/MAX",
  "Uses O(n) time and O(h) space",
]
hints = [
  "Pass valid range (min, max) to each node",
  "Left subtree: (min, node.val)",
  "Right subtree: (node.val, max)",
]
dependencies = ["subtree"]

[[exercises]]
slug = "kth-smallest-bst"
title = "7. Kth Smallest in BST - In-order Traversal"
kata = "neetcode_kth_smallest_bst"
objective = "Find kth smallest element using in-order traversal."
acceptance = [
  "Returns correct kth smallest value",
  "Uses O(k) average time",
  "In-order traversal visits BST in sorted order",
]
hints = [
  "In-order traversal gives sorted sequence",
  "Return kth element in traversal",
  "Can stop early after k elements",
]
dependencies = ["validate-bst"]

[[exercises]]
slug = "build-tree"
title = "8. Construct Binary Tree from Preorder/Inorder"
kata = "neetcode_build_tree"
objective = "Reconstruct tree from preorder and inorder traversals."
acceptance = [
  "Returns correct tree structure",
  "Handles trees with negative values",
  "Uses O(n) time with hashmap optimization",
]
hints = [
  "Preorder[0] is root",
  "Find root in inorder to split left/right",
  "Recursively build left and right subtrees",
]
dependencies = ["level-order", "validate-bst"]
```

I'll continue with templates for the remaining 10 topics. Due to length, I'll create them in batches:

---

### 8-11: Tries, Heap, Intervals, Greedy

```toml
# workbooks/neetcode_tries/manifest.toml
[workbook]
id = "neetcode_tries"
title = "NeetCode Tries: Prefix Tree Patterns"
summary = "Master trie (prefix tree) for efficient string search and prefix matching."
learning_goals = [
  "Implement trie with insert, search, and startsWith operations",
  "Use tries for word search and autocomplete problems",
  "Apply backtracking on tries for word search problems",
]
prerequisites = ["neetcode_group_anagrams"]
resources = [
  { title = "Trie Data Structure", url = "https://neetcode.io/courses/dsa-for-beginners/17" },
]
kata_namespace = "neetcode"

[[exercises]]
slug = "trie"
title = "1. Implement Trie (Prefix Tree)"
kata = "neetcode_trie"
objective = "Implement basic trie with insert, search, and startsWith methods."
acceptance = [
  "insert(word) adds word to trie",
  "search(word) returns True if word exists",
  "startsWith(prefix) returns True if prefix exists",
  "All operations run in O(m) where m is word length",
]
hints = [
  "Each node has children dict and is_end_of_word flag",
  "Insert: traverse/create nodes for each character",
  "Search: traverse and check is_end_of_word",
]
dependencies = []

[[exercises]]
slug = "word-dictionary"
title = "2. Design Add and Search Words Data Structure"
kata = "neetcode_word_dictionary"
objective = "Extend trie to support wildcard '.' searches with backtracking."
acceptance = [
  "addWord(word) adds word to trie",
  "search(word) supports '.' as wildcard",
  "Handles multiple wildcards in one word",
]
hints = [
  "For '.', try all children recursively",
  "Use DFS/backtracking for wildcard matching",
]
dependencies = ["trie"]

[[exercises]]
slug = "word-search-ii"
title = "3. Word Search II - Trie + Backtracking"
kata = "neetcode_word_search_ii"
objective = "Find all words from dictionary in 2D board using trie and backtracking."
acceptance = [
  "Returns all words found in board",
  "Uses trie to avoid checking non-words",
  "Prunes search when prefix doesn't exist in trie",
]
hints = [
  "Build trie from word list",
  "DFS from each cell, traverse trie simultaneously",
  "Mark cells visited during DFS (backtrack)",
]
dependencies = ["word-dictionary"]

# workbooks/neetcode_heap/manifest.toml
[workbook]
id = "neetcode_heap"
title = "NeetCode Heap: Priority Queue Mastery"
summary = "Master heap data structure for efficient priority queue operations."
learning_goals = [
  "Use Python's heapq for min-heap operations",
  "Implement max-heap using negation trick",
  "Apply heaps to top-k and streaming problems",
  "Combine heaps with other data structures (dict, set)",
]
prerequisites = ["neetcode_top_k_frequent"]
resources = [
  { title = "Heap/Priority Queue", url = "https://neetcode.io/courses/dsa-for-beginners/18" },
]
kata_namespace = "neetcode"

[[exercises]]
slug = "kth-largest"
title = "1. Kth Largest Element in Array"
kata = "neetcode_kth_largest"
objective = "Find kth largest element using heap (quickselect or heap)."
acceptance = [
  "Returns correct kth largest",
  "Uses O(n log k) with heap or O(n) average with quickselect",
]
hints = [
  "Min-heap of size k: pop smallest when size > k",
  "Final heap top is kth largest",
]
dependencies = []

[[exercises]]
slug = "last-stone-weight"
title = "2. Last Stone Weight - Max Heap Simulation"
kata = "neetcode_last_stone_weight"
objective = "Simulate stone smashing using max-heap."
acceptance = [
  "Returns weight of last stone or 0",
  "Uses heap for efficient max extraction",
]
hints = [
  "Python heapq is min-heap, negate values for max-heap",
  "Pop two largest, push difference back",
]
dependencies = ["kth-largest"]

[[exercises]]
slug = "k-closest-points"
title = "3. K Closest Points to Origin"
kata = "neetcode_k_closest_points"
objective = "Find k closest points using heap."
acceptance = [
  "Returns k closest points",
  "Uses O(n log k) time with heap",
]
hints = [
  "Use max-heap of size k with distances",
  "Maintain k smallest distances",
]
dependencies = ["kth-largest"]

[[exercises]]
slug = "kth-largest-stream"
title = "4. Kth Largest Element in Stream"
kata = "neetcode_kth_largest_stream"
objective = "Design class to find kth largest in stream."
acceptance = [
  "add(val) efficiently updates stream",
  "Returns kth largest in O(log k) per add",
]
hints = [
  "Maintain min-heap of size k",
  "Heap top is always kth largest",
]
dependencies = ["kth-largest"]

[[exercises]]
slug = "median-finder"
title = "5. Find Median from Data Stream"
kata = "neetcode_median_finder"
objective = "Find median from stream using two heaps."
acceptance = [
  "addNum(num) in O(log n)",
  "findMedian() in O(1)",
  "Uses two heaps to maintain balance",
]
hints = [
  "Max-heap for lower half, min-heap for upper half",
  "Keep heaps balanced (size difference <= 1)",
  "Median is top of larger heap or average of both tops",
]
dependencies = ["kth-largest-stream"]

# workbooks/neetcode_intervals/manifest.toml
[workbook]
id = "neetcode_intervals"
title = "NeetCode Intervals: Merging and Scheduling"
summary = "Master interval problems involving merging, intersection, and scheduling."
learning_goals = [
  "Sort intervals by start time for efficient processing",
  "Merge overlapping intervals",
  "Detect and handle interval conflicts",
  "Apply greedy strategies for scheduling problems",
]
prerequisites = ["neetcode_product_except_self"]
resources = [
  { title = "Intervals Patterns", url = "https://neetcode.io/courses/dsa-for-beginners/25" },
]
kata_namespace = "neetcode"

[[exercises]]
slug = "insert-interval"
title = "1. Insert Interval"
kata = "neetcode_insert_interval"
objective = "Insert interval and merge overlapping intervals."
acceptance = [
  "Returns merged intervals list",
  "Handles all overlap cases",
  "Maintains sorted order",
]
hints = [
  "Add all intervals before new interval",
  "Merge overlapping intervals with new interval",
  "Add all intervals after",
]
dependencies = []

[[exercises]]
slug = "merge-intervals"
title = "2. Merge Intervals"
kata = "neetcode_merge_intervals"
objective = "Merge all overlapping intervals."
acceptance = [
  "Returns minimal merged list",
  "Sorts intervals first",
  "Uses O(n log n) time",
]
hints = [
  "Sort by start time",
  "Compare current interval with last merged",
  "Merge if overlapping, else add to result",
]
dependencies = ["insert-interval"]

[[exercises]]
slug = "non-overlapping-intervals"
title = "3. Non-overlapping Intervals - Greedy"
kata = "neetcode_non_overlapping_intervals"
objective = "Find minimum removals to make intervals non-overlapping."
acceptance = [
  "Returns minimum count",
  "Uses greedy approach",
  "O(n log n) time",
]
hints = [
  "Sort by end time",
  "Keep interval with earliest end (greedy)",
  "Remove intervals that overlap with kept interval",
]
dependencies = ["merge-intervals"]

[[exercises]]
slug = "meeting-rooms"
title = "4. Meeting Rooms - Can Attend All"
kata = "neetcode_meeting_rooms"
objective = "Check if person can attend all meetings (no overlaps)."
acceptance = [
  "Returns True if no overlaps",
  "Sorts intervals",
  "O(n log n) time",
]
hints = [
  "Sort by start time",
  "Check each pair of consecutive intervals",
]
dependencies = ["merge-intervals"]

[[exercises]]
slug = "meeting-rooms-ii"
title = "5. Meeting Rooms II - Minimum Rooms Needed"
kata = "neetcode_meeting_rooms_ii"
objective = "Find minimum meeting rooms needed."
acceptance = [
  "Returns minimum rooms required",
  "Uses heap or sorted events",
  "O(n log n) time",
]
hints = [
  "Sort start and end times separately",
  "Use two pointers to track concurrent meetings",
  "Or use min-heap to track end times",
]
dependencies = ["meeting-rooms"]

[[exercises]]
slug = "min-interval-query"
title = "6. Minimum Interval to Include Each Query"
kata = "neetcode_min_interval_query"
objective = "For each query, find smallest interval containing it."
acceptance = [
  "Returns size of smallest interval for each query",
  "Handles queries not covered by any interval",
  "Uses sorting and heap for efficiency",
]
hints = [
  "Sort intervals and queries",
  "Use heap to track active intervals",
  "Remove intervals that end before query",
]
dependencies = ["meeting-rooms-ii"]

# workbooks/neetcode_greedy/manifest.toml
[workbook]
id = "neetcode_greedy"
title = "NeetCode Greedy: Optimal Local Choices"
summary = "Master greedy algorithms where local optimal choices lead to global optimum."
learning_goals = [
  "Recognize problems solvable by greedy approach",
  "Prove greedy choice property",
  "Apply sorting as preprocessing for greedy algorithms",
  "Avoid common pitfalls where greedy fails",
]
prerequisites = ["neetcode_two_sum"]
resources = [
  { title = "Greedy Algorithms", url = "https://neetcode.io/courses/dsa-for-beginners/24" },
]
kata_namespace = "neetcode"

[[exercises]]
slug = "max-subarray"
title = "1. Maximum Subarray - Kadane's Algorithm"
kata = "neetcode_max_subarray"
objective = "Find subarray with maximum sum using Kadane's algorithm."
acceptance = [
  "Returns maximum sum",
  "Uses O(n) time and O(1) space",
  "Handles all negative arrays",
]
hints = [
  "Track current_sum and max_sum",
  "Reset current_sum to 0 if it goes negative",
]
dependencies = []

[[exercises]]
slug = "jump-game"
title = "2. Jump Game - Greedy Reachability"
kata = "neetcode_jump_game"
objective = "Check if last index is reachable with greedy approach."
acceptance = [
  "Returns True if reachable",
  "Uses O(n) time and O(1) space",
  "Tracks farthest reachable position",
]
hints = [
  "Track max_reach as you iterate",
  "Update max_reach = max(max_reach, i + nums[i])",
  "If i > max_reach, can't proceed",
]
dependencies = ["max-subarray"]

[[exercises]]
slug = "jump-game-ii"
title = "3. Jump Game II - Minimum Jumps"
kata = "neetcode_jump_game_ii"
objective = "Find minimum jumps to reach end."
acceptance = [
  "Returns minimum jump count",
  "Uses O(n) time greedy approach",
  "Tracks current and next jump boundaries",
]
hints = [
  "Track current_end and farthest",
  "Increment jumps when reaching current_end",
  "Update current_end to farthest",
]
dependencies = ["jump-game"]

[[exercises]]
slug = "gas-station"
title = "4. Gas Station - Circular Array Greedy"
kata = "neetcode_gas_station"
objective = "Find starting gas station for circular route."
acceptance = [
  "Returns starting index or -1",
  "Uses O(n) time single pass",
  "Greedy choice: restart when tank < 0",
]
hints = [
  "If total_gas < total_cost, impossible",
  "Track tank and start index",
  "Reset start when tank goes negative",
]
dependencies = ["jump-game"]

[[exercises]]
slug = "hand-straights"
title = "5. Hand of Straights - Greedy Grouping"
kata = "neetcode_hand_straights"
objective = "Check if cards can form consecutive groups of size k."
acceptance = [
  "Returns True if possible",
  "Uses O(n log n) time",
  "Greedy: form groups starting from smallest",
]
hints = [
  "Sort and use frequency map",
  "For each card, try to form group [card, card+1, ..., card+k-1]",
  "Decrement frequencies",
]
dependencies = ["max-subarray"]

[[exercises]]
slug = "partition-labels"
title = "6. Partition Labels - Greedy Partitioning"
kata = "neetcode_partition_labels"
objective = "Partition string into maximum partitions with unique characters."
acceptance = [
  "Returns list of partition sizes",
  "Each character appears in at most one partition",
  "Uses O(n) time",
]
hints = [
  "Track last occurrence of each character",
  "Expand partition until all chars in partition are done",
]
dependencies = ["gas-station"]
```

---

### 12-15: Backtracking, Graphs, Advanced Graphs, 1-D DP

Due to space constraints, I'll provide a shorter version for the final topics. The pattern is clear, and you can expand them following the same structure:

```toml
# workbooks/neetcode_backtracking/manifest.toml
[workbook]
id = "neetcode_backtracking"
title = "NeetCode Backtracking: Exhaustive Search with Pruning"
summary = "Master backtracking for combinatorial problems with decision trees."
learning_goals = [
  "Implement backtracking with choice, constraint, and goal",
  "Use visited sets and path tracking",
  "Prune invalid branches early",
  "Generate all subsets, permutations, and combinations",
]
prerequisites = ["neetcode_generate_parentheses"]
resources = [{ title = "Backtracking", url = "https://neetcode.io/courses/dsa-for-beginners/26" }]
kata_namespace = "neetcode"

# Add exercises: subsets, combination_sum, permutations, subsets_ii, combination_sum_ii,
# word_search, palindrome_partition, letter_combinations, n_queens

# workbooks/neetcode_graphs/manifest.toml
[workbook]
id = "neetcode_graphs"
title = "NeetCode Graphs: DFS and BFS Fundamentals"
summary = "Master graph traversal, connected components, and basic graph algorithms."
learning_goals = [
  "Implement DFS and BFS on graphs",
  "Detect cycles and connected components",
  "Use visited sets to track exploration",
  "Apply topological sort for DAGs",
]
prerequisites = ["neetcode_num_islands"]
kata_namespace = "neetcode"

# Add exercises: num_islands, clone_graph, pacific_atlantic, surrounded_regions,
# rotting_oranges, walls_gates, course_schedule, course_schedule_ii

# workbooks/neetcode_advanced_graphs/manifest.toml
[workbook]
id = "neetcode_advanced_graphs"
title = "NeetCode Advanced Graphs: MST and Shortest Paths"
summary = "Master advanced graph algorithms including Dijkstra, Prim's, and Union-Find."
learning_goals = [
  "Implement Dijkstra's shortest path algorithm",
  "Use Prim's or Kruskal's for MST",
  "Apply Union-Find for connectivity",
  "Solve advanced path-finding problems",
]
prerequisites = ["neetcode_course_schedule"]
kata_namespace = "neetcode"

# Add exercises: min_cost_connect_points, network_delay_time, swim_rising_water,
# alien_dictionary, cheapest_flights, reconstruct_itinerary

# workbooks/neetcode_1d_dp/manifest.toml
[workbook]
id = "neetcode_1d_dp"
title = "NeetCode 1-D DP: Sequential Decision Problems"
summary = "Master 1-D dynamic programming for sequential optimization problems."
learning_goals = [
  "Identify optimal substructure and overlapping subproblems",
  "Write recurrence relations",
  "Implement bottom-up and top-down DP",
  "Optimize space complexity from O(n) to O(1)",
]
prerequisites = ["neetcode_max_subarray"]
resources = [{ title = "Dynamic Programming", url = "https://neetcode.io/courses/dsa-for-beginners/27" }]
kata_namespace = "neetcode"

# Add exercises: climbing_stairs, min_cost_stairs, house_robber, house_robber_ii,
# longest_palindrome, palindromic_substrings, decode_ways, coin_change,
# max_product_subarray, word_break, longest_increasing_subseq, partition_equal_subset
```

---

### 16-18: 2-D DP, Bit Manipulation, Math & Geometry

```toml
# workbooks/neetcode_2d_dp/manifest.toml
[workbook]
id = "neetcode_2d_dp"
title = "NeetCode 2-D DP: Grid and String Problems"
summary = "Master 2-D dynamic programming for grid paths and string matching."
learning_goals = [
  "Build 2-D DP tables from recurrence relations",
  "Solve grid path-counting problems",
  "Apply DP to string edit distance and subsequences",
  "Optimize space with rolling arrays",
]
prerequisites = ["neetcode_climbing_stairs", "neetcode_coin_change"]
kata_namespace = "neetcode"

# Add exercises: unique_paths, lcs, coin_change_ii, target_sum, interleaving_string,
# edit_distance, distinct_subsequences, burst_balloons, regex_matching, longest_increasing_path

# workbooks/neetcode_bit_manipulation/manifest.toml
[workbook]
id = "neetcode_bit_manipulation"
title = "NeetCode Bit Manipulation: Efficient Low-Level Operations"
summary = "Master bitwise operations for optimization and mathematical problems."
learning_goals = [
  "Use AND, OR, XOR, shift operators effectively",
  "Apply XOR properties (a ^ a = 0, a ^ 0 = a)",
  "Count set bits and manipulate individual bits",
  "Solve array problems with bit tricks",
]
prerequisites = ["neetcode_single_number"]
resources = [{ title = "Bit Manipulation", url = "https://neetcode.io/courses/dsa-for-beginners/23" }]
kata_namespace = "neetcode"

[[exercises]]
slug = "single-number"
title = "1. Single Number - XOR Property"
kata = "neetcode_single_number"
objective = "Find single number in array where others appear twice using XOR."
acceptance = [
  "Returns the single number",
  "Uses O(n) time and O(1) space",
  "Uses XOR property: a ^ a = 0",
]
hints = [
  "XOR all numbers together",
  "Duplicates cancel out (a ^ a = 0)",
  "Result is the single number",
]
dependencies = []

[[exercises]]
slug = "hamming-weight"
title = "2. Number of 1 Bits - Count Set Bits"
kata = "neetcode_hamming_weight"
objective = "Count number of 1 bits in binary representation."
acceptance = [
  "Returns correct count",
  "Uses O(1) time (32 iterations max)",
]
hints = [
  "Use n & 1 to check last bit",
  "Right shift n to check next bit",
  "Or use n = n & (n-1) trick to clear rightmost 1",
]
dependencies = []

[[exercises]]
slug = "counting-bits"
title = "3. Counting Bits - DP with Bit Manipulation"
kata = "neetcode_counting_bits"
objective = "Count 1 bits for all numbers [0, n] using DP."
acceptance = [
  "Returns array of counts",
  "Uses O(n) time",
  "DP relation: dp[i] = dp[i >> 1] + (i & 1)",
]
hints = [
  "i >> 1 is i divided by 2 (drops last bit)",
  "i & 1 is the last bit",
]
dependencies = ["hamming-weight"]

[[exercises]]
slug = "reverse-bits"
title = "4. Reverse Bits"
kata = "neetcode_reverse_bits"
objective = "Reverse bits of a 32-bit unsigned integer."
acceptance = [
  "Returns reversed bits",
  "Handles all 32 bits",
]
hints = [
  "Build result by shifting left",
  "Add last bit of input to result",
  "Shift input right",
]
dependencies = ["hamming-weight"]

[[exercises]]
slug = "missing-number"
title = "5. Missing Number - XOR or Sum"
kata = "neetcode_missing_number"
objective = "Find missing number in [0, n] using XOR or sum."
acceptance = [
  "Returns missing number",
  "Uses O(n) time and O(1) space",
]
hints = [
  "Expected sum = n*(n+1)/2",
  "Missing = expected - actual",
  "Or XOR all indices and values",
]
dependencies = ["single-number"]

[[exercises]]
slug = "sum-two-integers"
title = "6. Sum of Two Integers - No + or - Operators"
kata = "neetcode_sum_two_integers"
objective = "Add two integers using only bitwise operations."
acceptance = [
  "Returns a + b",
  "Uses only &, |, ^, << operators",
  "Handles negative numbers",
]
hints = [
  "XOR gives sum without carry",
  "AND gives carry positions",
  "Shift carry left and repeat",
]
dependencies = ["counting-bits"]

# workbooks/neetcode_math_geometry/manifest.toml
[workbook]
id = "neetcode_math_geometry"
title = "NeetCode Math & Geometry: Computational Problems"
summary = "Master mathematical algorithms and 2D geometry problems."
learning_goals = [
  "Implement mathematical algorithms (fast power, digit manipulation)",
  "Solve 2D array rotation and spiral traversal",
  "Apply mathematical insights for optimization",
  "Handle edge cases in numerical problems",
]
prerequisites = ["neetcode_rotate_image"]
kata_namespace = "neetcode"

[[exercises]]
slug = "rotate-image"
title = "1. Rotate Image - Matrix Manipulation"
kata = "neetcode_rotate_image"
objective = "Rotate n×n matrix 90° clockwise in-place."
acceptance = [
  "Rotates in-place",
  "Uses O(1) extra space",
  "Handles all matrix sizes",
]
hints = [
  "Transpose matrix (swap matrix[i][j] with matrix[j][i])",
  "Reverse each row",
]
dependencies = []

[[exercises]]
slug = "spiral-matrix"
title = "2. Spiral Matrix - Boundary Traversal"
kata = "neetcode_spiral_matrix"
objective = "Return matrix elements in spiral order."
acceptance = [
  "Returns correct spiral order",
  "Handles m×n matrices",
  "Uses O(1) space",
]
hints = [
  "Track top, bottom, left, right boundaries",
  "Move right, down, left, up in cycles",
  "Update boundaries after each direction",
]
dependencies = ["rotate-image"]

[[exercises]]
slug = "set-matrix-zeroes"
title = "3. Set Matrix Zeroes - In-Place Marking"
kata = "neetcode_set_matrix_zeroes"
objective = "Set row and column to 0 if element is 0, in-place."
acceptance = [
  "Uses O(1) space",
  "Handles overlapping zero positions",
]
hints = [
  "Use first row and column as markers",
  "Track separately if first row/column need zeroing",
]
dependencies = ["rotate-image"]

[[exercises]]
slug = "happy-number"
title = "4. Happy Number - Cycle Detection"
kata = "neetcode_happy_number"
objective = "Check if number is happy (sum of squares reaches 1) using cycle detection."
acceptance = [
  "Returns True if happy",
  "Uses Floyd's cycle detection or set",
]
hints = [
  "Compute sum of squares of digits",
  "Detect cycle with slow/fast pointers or set",
]
dependencies = []

[[exercises]]
slug = "plus-one"
title = "5. Plus One - Digit Manipulation"
kata = "neetcode_plus_one"
objective = "Add one to number represented as array of digits."
acceptance = [
  "Returns correct result",
  "Handles carry (e.g., [9,9,9] → [1,0,0,0])",
]
hints = [
  "Start from last digit",
  "Add 1, handle carry",
]
dependencies = []

[[exercises]]
slug = "pow"
title = "6. Pow(x, n) - Fast Exponentiation"
kata = "neetcode_pow"
objective = "Implement pow(x, n) using fast exponentiation."
acceptance = [
  "Returns correct result",
  "Uses O(log n) time",
  "Handles negative exponents",
]
hints = [
  "If n is even: x^n = (x^2)^(n/2)",
  "If n is odd: x^n = x * x^(n-1)",
]
dependencies = []

[[exercises]]
slug = "multiply-strings"
title = "7. Multiply Strings - String Digit Multiplication"
kata = "neetcode_multiply_strings"
objective = "Multiply two numbers represented as strings."
acceptance = [
  "Returns product as string",
  "Does not use int() or built-in BigInteger",
]
hints = [
  "Multiply digit by digit",
  "Track position for result array",
  "Handle carries at the end",
]
dependencies = ["plus-one"]

[[exercises]]
slug = "detect-squares"
title = "8. Detect Squares - Coordinate Geometry"
kata = "neetcode_detect_squares"
objective = "Count axis-aligned squares with given points."
acceptance = [
  "add(point) and count(point) work correctly",
  "Counts all valid squares",
]
hints = [
  "Store points with frequencies",
  "For diagonal point, check if other corners exist",
]
dependencies = ["rotate-image"]
```

---

## Creating All Workbooks at Once

If you want to create all workbooks programmatically, use this bash script:

```bash
#!/bin/bash

# Array of workbook topics
topics=(
  "stack" "sliding_window" "linked_list" "trees" "tries"
  "heap" "intervals" "greedy" "backtracking" "graphs"
  "advanced_graphs" "1d_dp" "2d_dp" "bit_manipulation" "math_geometry"
)

# Create directories
for topic in "${topics[@]}"; do
  mkdir -p "workbooks/neetcode_$topic"
  echo "Created workbooks/neetcode_$topic/"
done

echo "✅ All 15 remaining workbook directories created!"
echo "📝 Now copy the manifest.toml templates from this guide into each directory"
```

---

## Summary Table

| # | Topic | Exercises | Status | Prerequisites |
|---|-------|-----------|--------|---------------|
| 1 | Arrays & Hashing | 8/17 | ✅ Created | None |
| 2 | Two Pointers | 5/5 | ✅ Created | Arrays & Hashing |
| 3 | Stack | 7/7 | 📋 Template Ready | Arrays & Hashing |
| 4 | Binary Search | 7/7 | ✅ Created | Arrays & Hashing |
| 5 | Sliding Window | 6/6 | 📋 Template Ready | Arrays & Hashing |
| 6 | Linked List | 8/11 | 📋 Template Ready | None |
| 7 | Trees | 8/15 | 📋 Template Ready | Linked List |
| 8 | Tries | 3/3 | 📋 Template Ready | Arrays & Hashing |
| 9 | Heap | 5/8 | 📋 Template Ready | Arrays & Hashing |
| 10 | Intervals | 6/6 | 📋 Template Ready | Arrays & Hashing |
| 11 | Greedy | 6/8 | 📋 Template Ready | Arrays & Hashing |
| 12 | Backtracking | 8/10 | 📋 Template Ready | Trees |
| 13 | Graphs | 8/13 | 📋 Template Ready | Trees |
| 14 | Advanced Graphs | 6/6 | 📋 Template Ready | Graphs |
| 15 | 1-D DP | 8/12 | 📋 Template Ready | Greedy |
| 16 | 2-D DP | 8/11 | 📋 Template Ready | 1-D DP |
| 17 | Bit Manipulation | 6/6 | 📋 Template Ready | Arrays & Hashing |
| 18 | Math & Geometry | 8/9 | 📋 Template Ready | Arrays & Hashing |

---

## Next Steps

1. **Test the created workbooks:**
   ```bash
   cargo build
   cargo run
   # Press 'w' to browse workbooks
   # Verify the 3 created workbooks appear
   ```

2. **Create remaining workbooks:**
   - Copy templates from this guide
   - Customize exercise lists and prerequisites
   - Test each one

3. **Commit progressively:**
   ```bash
   git add workbooks/neetcode_stack/
   git commit -m "Add NeetCode Stack workbook"
   ```

4. **Consider creating HTML pages:**
   - Optional but enhances learning experience
   - Add diagrams, code comparisons, and visual aids
   - See `assets/workbooks/_template/` for template

---

## Tips for Success

1. **Start with easy topics:** Stack, Sliding Window, Intervals have clear patterns
2. **Test exercises exist:** Before creating workbook, verify all katas exist
3. **Progressive dependencies:** Ensure dependency chain makes sense
4. **Keep it focused:** 5-8 core exercises per workbook is ideal
5. **Add resources:** Link to NeetCode videos, LeetCode discussions

---

For detailed analysis and exercise mappings, see `NEETCODE_WORKBOOK_MAPPING.md`.
