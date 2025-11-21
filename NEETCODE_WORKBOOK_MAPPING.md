# NeetCode Workbook Mapping

This document provides a comprehensive mapping of all 151 NeetCode exercises organized by topic according to the NeetCode roadmap structure. Each topic includes exercise counts, learning order recommendations, prerequisites, and core exercises for focused workbooks.

## Topics Overview

1. Arrays & Hashing (17 exercises) - Foundation
2. Two Pointers (5 exercises)
3. Stack (7 exercises)
4. Binary Search (7 exercises)
5. Sliding Window (6 exercises)
6. Linked List (11 exercises)
7. Trees (15 exercises)
8. Tries (3 exercises)
9. Heap/Priority Queue (8 exercises)
10. Intervals (6 exercises)
11. Greedy (8 exercises)
12. Advanced Graphs (6 exercises)
13. Backtracking (10 exercises)
14. Graphs (13 exercises)
15. 1-D DP (12 exercises)
16. 2-D DP (11 exercises)
17. Bit Manipulation (6 exercises)
18. Math & Geometry (9 exercises)

**Total: 151 exercises**

---

## 1. Arrays & Hashing (Foundation)

**Prerequisites:** None (this is the foundation topic)

**Exercise Count:** 17

### All Exercises (Easy → Hard):

1. `neetcode_contains_duplicate` ⭐ Easy
2. `neetcode_valid_anagram` ⭐ Easy
3. `neetcode_two_sum` ⭐ Easy
4. `neetcode_plus_one` ⭐ Easy
5. `neetcode_missing_number` ⭐ Easy
6. `neetcode_happy_number` ⭐ Easy
7. `neetcode_group_anagrams` ⭐⭐ Medium
8. `neetcode_top_k_frequent` ⭐⭐ Medium
9. `neetcode_product_except_self` ⭐⭐ Medium
10. `neetcode_longest_consecutive` ⭐⭐ Medium
11. `neetcode_valid_sudoku` ⭐⭐ Medium
12. `neetcode_encode_decode_strings` ⭐⭐ Medium (Premium)
13. `neetcode_rotate_image` ⭐⭐ Medium
14. `neetcode_spiral_matrix` ⭐⭐ Medium
15. `neetcode_set_matrix_zeroes` ⭐⭐ Medium
16. `neetcode_detect_squares` ⭐⭐ Medium
17. `neetcode_twitter` ⭐⭐ Medium (Design)

### Core Workbook Exercises (6):

1. `neetcode_two_sum` - Classic hashmap pattern
2. `neetcode_valid_anagram` - Frequency counting
3. `neetcode_group_anagrams` - Hashmap with key generation
4. `neetcode_top_k_frequent` - Frequency + bucketing
5. `neetcode_product_except_self` - Array manipulation without division
6. `neetcode_longest_consecutive` - Set-based sequential search

### Learning Path:

Start with basic frequency counting (contains_duplicate, valid_anagram), then progress to hashmap patterns (two_sum, group_anagrams), and finally tackle array transformations (product_except_self, rotate_image).

---

## 2. Two Pointers

**Prerequisites:** Arrays & Hashing

**Exercise Count:** 5

### All Exercises (Easy → Hard):

1. `neetcode_valid_palindrome` ⭐ Easy
2. `neetcode_two_sum_ii` ⭐ Easy
3. `neetcode_3sum` ⭐⭐ Medium
4. `neetcode_container_most_water` ⭐⭐ Medium
5. `neetcode_trapping_rain_water` ⭐⭐⭐ Hard

### Core Workbook Exercises (5):

All 5 exercises are core - this is a focused topic that builds systematically.

1. `neetcode_valid_palindrome` - Basic two pointer pattern
2. `neetcode_two_sum_ii` - Sorted array search
3. `neetcode_3sum` - Extending two pointers to three elements
4. `neetcode_container_most_water` - Optimization with greedy movement
5. `neetcode_trapping_rain_water` - Complex two pointer with min/max tracking

### Learning Path:

Master the basic two-pointer pattern first, then tackle sorted array problems, and finally solve optimization problems requiring strategic pointer movement.

---

## 3. Stack

**Prerequisites:** Arrays & Hashing

**Exercise Count:** 7

### All Exercises (Easy → Hard):

1. `neetcode_valid_parentheses` ⭐ Easy
2. `neetcode_min_stack` ⭐⭐ Medium
3. `neetcode_evaluate_rpn` ⭐⭐ Medium
4. `neetcode_generate_parentheses` ⭐⭐ Medium (also Backtracking)
5. `neetcode_daily_temperatures` ⭐⭐ Medium
6. `neetcode_car_fleet` ⭐⭐ Medium
7. `neetcode_largest_rectangle` ⭐⭐⭐ Hard

### Core Workbook Exercises (6):

1. `neetcode_valid_parentheses` - Stack basics
2. `neetcode_min_stack` - Auxiliary stack pattern
3. `neetcode_evaluate_rpn` - Stack for expression evaluation
4. `neetcode_daily_temperatures` - Monotonic stack
5. `neetcode_car_fleet` - Stack with custom logic
6. `neetcode_largest_rectangle` - Advanced monotonic stack

### Learning Path:

Start with basic stack operations (valid_parentheses), learn auxiliary stack patterns (min_stack), then master monotonic stack techniques for next greater/smaller element problems.

---

## 4. Binary Search

**Prerequisites:** Arrays & Hashing

**Exercise Count:** 7

### All Exercises (Easy → Hard):

1. `neetcode_binary_search` ⭐ Easy
2. `neetcode_search_2d_matrix` ⭐⭐ Medium
3. `neetcode_koko_bananas` ⭐⭐ Medium
4. `neetcode_find_min_rotated` ⭐⭐ Medium
5. `neetcode_search_rotated` ⭐⭐ Medium
6. `neetcode_time_map` ⭐⭐ Medium
7. `neetcode_median_two_arrays` ⭐⭐⭐ Hard

### Core Workbook Exercises (6):

1. `neetcode_binary_search` - Classic binary search
2. `neetcode_search_2d_matrix` - 2D binary search
3. `neetcode_find_min_rotated` - Rotated array pattern
4. `neetcode_search_rotated` - Search in rotated array
5. `neetcode_koko_bananas` - Binary search on answer space
6. `neetcode_median_two_arrays` - Advanced partitioning

### Learning Path:

Master classic binary search first, then handle rotated arrays, learn "binary search on answer" pattern, and finally tackle advanced partitioning problems.

---

## 5. Sliding Window

**Prerequisites:** Arrays & Hashing, Two Pointers

**Exercise Count:** 6

### All Exercises (Easy → Hard):

1. `neetcode_best_time_stock` ⭐ Easy
2. `neetcode_longest_substring` ⭐⭐ Medium
3. `neetcode_longest_repeating` ⭐⭐ Medium
4. `neetcode_permutation_string` ⭐⭐ Medium
5. `neetcode_min_window_substring` ⭐⭐⭐ Hard
6. `neetcode_sliding_window_max` ⭐⭐⭐ Hard

### Core Workbook Exercises (6):

All 6 exercises are core - essential patterns for sliding window mastery.

1. `neetcode_best_time_stock` - Simple sliding window
2. `neetcode_longest_substring` - Variable window with set
3. `neetcode_longest_repeating` - Window with character replacement
4. `neetcode_permutation_string` - Fixed window with frequency match
5. `neetcode_min_window_substring` - Minimum window with all characters
6. `neetcode_sliding_window_max` - Window with monotonic deque

### Learning Path:

Start with simple one-pass problems, then learn variable-size windows with hash maps, progress to constraint-based windows, and master advanced patterns with deques.

---

## 6. Linked List

**Prerequisites:** Arrays & Hashing

**Exercise Count:** 11

### All Exercises (Easy → Hard):

1. `neetcode_reverse_linked_list` ⭐ Easy
2. `neetcode_merge_two_sorted_lists` ⭐ Easy
3. `neetcode_linked_list_cycle` ⭐ Easy
4. `neetcode_remove_nth_node` ⭐⭐ Medium
5. `neetcode_reorder_list` ⭐⭐ Medium
6. `neetcode_add_two_numbers` ⭐⭐ Medium
7. `neetcode_copy_random_list` ⭐⭐ Medium
8. `neetcode_find_duplicate` ⭐⭐ Medium (cycle detection)
9. `neetcode_lru_cache` ⭐⭐ Medium (Design)
10. `neetcode_merge_k_lists` ⭐⭐⭐ Hard
11. `neetcode_reverse_k_group` ⭐⭐⭐ Hard

### Core Workbook Exercises (7):

1. `neetcode_reverse_linked_list` - Fundamental pointer manipulation
2. `neetcode_merge_two_sorted_lists` - Two-pointer merge
3. `neetcode_linked_list_cycle` - Floyd's algorithm
4. `neetcode_reorder_list` - Complex multi-step manipulation
5. `neetcode_copy_random_list` - Deep copy with extra pointers
6. `neetcode_lru_cache` - Doubly linked list + hashmap design
7. `neetcode_reverse_k_group` - Advanced reversal pattern

### Learning Path:

Master basic pointer manipulation (reverse), learn merging techniques, understand cycle detection, then tackle complex multi-step operations and design problems.

---

## 7. Trees

**Prerequisites:** Arrays & Hashing

**Exercise Count:** 15

### All Exercises (Easy → Hard):

1. `neetcode_invert_tree` ⭐ Easy
2. `neetcode_max_depth` ⭐ Easy
3. `neetcode_same_tree` ⭐ Easy
4. `neetcode_subtree` ⭐ Easy
5. `neetcode_diameter_tree` ⭐ Easy
6. `neetcode_balanced_tree` ⭐ Easy
7. `neetcode_lca_bst` ⭐⭐ Medium
8. `neetcode_level_order` ⭐⭐ Medium
9. `neetcode_right_side_view` ⭐⭐ Medium
10. `neetcode_good_nodes` ⭐⭐ Medium
11. `neetcode_validate_bst` ⭐⭐ Medium
12. `neetcode_kth_smallest_bst` ⭐⭐ Medium
13. `neetcode_build_tree` ⭐⭐ Medium
14. `neetcode_serialize_tree` ⭐⭐⭐ Hard
15. `neetcode_max_path_sum` ⭐⭐⭐ Hard

### Core Workbook Exercises (8):

1. `neetcode_invert_tree` - Basic recursion
2. `neetcode_max_depth` - DFS depth calculation
3. `neetcode_same_tree` - Tree comparison
4. `neetcode_diameter_tree` - Return multiple values in recursion
5. `neetcode_level_order` - BFS traversal
6. `neetcode_validate_bst` - Range-based validation
7. `neetcode_kth_smallest_bst` - In-order traversal
8. `neetcode_serialize_tree` - Encode/decode pattern

### Learning Path:

Start with basic DFS recursion, learn BFS for level-order traversal, master BST properties, then tackle advanced problems involving tree construction and serialization.

---

## 8. Tries

**Prerequisites:** Trees

**Exercise Count:** 3

### All Exercises (Easy → Hard):

1. `neetcode_trie` ⭐⭐ Medium
2. `neetcode_word_dictionary` ⭐⭐ Medium
3. `neetcode_word_search_ii` ⭐⭐⭐ Hard

### Core Workbook Exercises (3):

All 3 exercises are core - this is a focused topic.

1. `neetcode_trie` - Basic trie implementation
2. `neetcode_word_dictionary` - Trie with wildcard search
3. `neetcode_word_search_ii` - Trie + backtracking on board

### Learning Path:

Build a basic trie first, add wildcard search functionality, then combine trie with backtracking for complex search problems.

---

## 9. Heap/Priority Queue

**Prerequisites:** Arrays & Hashing

**Exercise Count:** 8

### All Exercises (Easy → Hard):

1. `neetcode_kth_largest` ⭐⭐ Medium
2. `neetcode_last_stone_weight` ⭐ Easy
3. `neetcode_k_closest_points` ⭐⭐ Medium
4. `neetcode_kth_largest_stream` ⭐ Easy
5. `neetcode_task_scheduler` ⭐⭐ Medium
6. `neetcode_twitter` ⭐⭐ Medium (Design, also in Arrays)
7. `neetcode_median_finder` ⭐⭐⭐ Hard
8. `neetcode_merge_k_lists` ⭐⭐⭐ Hard (also in Linked List)

### Core Workbook Exercises (6):

1. `neetcode_last_stone_weight` - Max heap basics
2. `neetcode_kth_largest` - Quick select or min heap
3. `neetcode_kth_largest_stream` - Stream processing with heap
4. `neetcode_k_closest_points` - K-closest pattern
5. `neetcode_task_scheduler` - Greedy + heap scheduling
6. `neetcode_median_finder` - Two heaps pattern

### Learning Path:

Learn basic heap operations, understand k-th element patterns, master stream processing, then tackle two-heap problems for median tracking.

---

## 10. Intervals

**Prerequisites:** Arrays & Hashing

**Exercise Count:** 6

### All Exercises (Easy → Hard):

1. `neetcode_meeting_rooms` ⭐ Easy
2. `neetcode_insert_interval` ⭐⭐ Medium
3. `neetcode_merge_intervals` ⭐⭐ Medium
4. `neetcode_non_overlapping_intervals` ⭐⭐ Medium
5. `neetcode_meeting_rooms_ii` ⭐⭐ Medium
6. `neetcode_min_interval_query` ⭐⭐⭐ Hard

### Core Workbook Exercises (5):

1. `neetcode_meeting_rooms` - Basic interval overlap
2. `neetcode_merge_intervals` - Merging pattern
3. `neetcode_insert_interval` - Insertion with merge
4. `neetcode_non_overlapping_intervals` - Greedy removal
5. `neetcode_meeting_rooms_ii` - Minimum rooms (sweep line)

### Learning Path:

Start with basic overlap detection, learn merging techniques, master insertion patterns, and tackle optimization problems with greedy or sweep line algorithms.

---

## 11. Greedy

**Prerequisites:** Arrays & Hashing

**Exercise Count:** 8

### All Exercises (Easy → Hard):

1. `neetcode_max_subarray` ⭐⭐ Medium (Kadane's)
2. `neetcode_jump_game` ⭐⭐ Medium
3. `neetcode_jump_game_ii` ⭐⭐ Medium
4. `neetcode_gas_station` ⭐⭐ Medium
5. `neetcode_hand_straights` ⭐⭐ Medium
6. `neetcode_merge_triplets` ⭐⭐ Medium
7. `neetcode_partition_labels` ⭐⭐ Medium
8. `neetcode_valid_parenthesis_string` ⭐⭐ Medium

### Core Workbook Exercises (6):

1. `neetcode_max_subarray` - Kadane's algorithm
2. `neetcode_jump_game` - Reachability check
3. `neetcode_jump_game_ii` - Minimum jumps with BFS-like approach
4. `neetcode_gas_station` - Circular array greedy
5. `neetcode_partition_labels` - Interval partitioning
6. `neetcode_valid_parenthesis_string` - Two-pass greedy

### Learning Path:

Master Kadane's algorithm first, learn reachability and optimization problems, then tackle problems requiring greedy choice with careful state tracking.

---

## 12. Advanced Graphs

**Prerequisites:** Graphs, Heap/Priority Queue

**Exercise Count:** 6

### All Exercises (Easy → Hard):

1. `neetcode_min_cost_connect_points` ⭐⭐ Medium (MST - Prim's)
2. `neetcode_network_delay_time` ⭐⭐ Medium (Dijkstra)
3. `neetcode_cheapest_flights` ⭐⭐ Medium (Bellman-Ford)
4. `neetcode_swim_rising_water` ⭐⭐⭐ Hard (Binary Search + BFS)
5. `neetcode_alien_dictionary` ⭐⭐⭐ Hard (Topological Sort)
6. `neetcode_reconstruct_itinerary` ⭐⭐⭐ Hard (Eulerian Path)

### Core Workbook Exercises (6):

All 6 exercises are core - these are advanced algorithms.

1. `neetcode_network_delay_time` - Dijkstra's shortest path
2. `neetcode_cheapest_flights` - Bellman-Ford with k stops
3. `neetcode_min_cost_connect_points` - Prim's MST
4. `neetcode_alien_dictionary` - Topological sort on implicit graph
5. `neetcode_swim_rising_water` - Binary search + graph traversal
6. `neetcode_reconstruct_itinerary` - Eulerian path (Hierholzer's)

### Learning Path:

Master Dijkstra's algorithm first, learn variations (k-shortest path), understand MST algorithms, then tackle topological sort and Eulerian path problems.

---

## 13. Backtracking

**Prerequisites:** Arrays & Hashing, Trees (for understanding recursion)

**Exercise Count:** 10

### All Exercises (Easy → Hard):

1. `neetcode_subsets` ⭐⭐ Medium
2. `neetcode_combination_sum` ⭐⭐ Medium
3. `neetcode_permutations` ⭐⭐ Medium
4. `neetcode_subsets_ii` ⭐⭐ Medium
5. `neetcode_combination_sum_ii` ⭐⭐ Medium
6. `neetcode_word_search` ⭐⭐ Medium
7. `neetcode_palindrome_partition` ⭐⭐ Medium
8. `neetcode_letter_combinations` ⭐⭐ Medium
9. `neetcode_n_queens` ⭐⭐⭐ Hard
10. `neetcode_n_queens_ii` ⭐⭐⭐ Hard

### Core Workbook Exercises (7):

1. `neetcode_subsets` - Basic backtracking template
2. `neetcode_permutations` - Permutation pattern
3. `neetcode_combination_sum` - Combination with repetition
4. `neetcode_subsets_ii` - Handling duplicates
5. `neetcode_word_search` - 2D backtracking with visited tracking
6. `neetcode_palindrome_partition` - Backtracking with validation
7. `neetcode_n_queens` - Classic constraint satisfaction

### Learning Path:

Start with subsets to learn the basic template, understand permutations vs combinations, learn duplicate handling, then tackle 2D backtracking and constraint satisfaction problems.

---

## 14. Graphs

**Prerequisites:** Arrays & Hashing, Trees (for DFS/BFS understanding)

**Exercise Count:** 13

### All Exercises (Easy → Hard):

1. `neetcode_num_islands` ⭐⭐ Medium
2. `neetcode_clone_graph` ⭐⭐ Medium
3. `neetcode_max_area_island` ⭐⭐ Medium
4. `neetcode_pacific_atlantic` ⭐⭐ Medium
5. `neetcode_surrounded_regions` ⭐⭐ Medium
6. `neetcode_rotting_oranges` ⭐⭐ Medium
7. `neetcode_walls_gates` ⭐⭐ Medium
8. `neetcode_course_schedule` ⭐⭐ Medium
9. `neetcode_course_schedule_ii` ⭐⭐ Medium
10. `neetcode_connected_components` ⭐⭐ Medium (Union-Find)
11. `neetcode_graph_valid_tree` ⭐⭐ Medium
12. `neetcode_redundant_connection` ⭐⭐ Medium (Union-Find)
13. `neetcode_word_ladder` ⭐⭐⭐ Hard

### Core Workbook Exercises (8):

1. `neetcode_num_islands` - DFS/BFS on 2D grid
2. `neetcode_clone_graph` - Graph cloning with hashmap
3. `neetcode_pacific_atlantic` - Multi-source DFS
4. `neetcode_rotting_oranges` - Multi-source BFS
5. `neetcode_course_schedule` - Cycle detection (topological sort)
6. `neetcode_course_schedule_ii` - Topological sort order
7. `neetcode_graph_valid_tree` - Union-Find or DFS cycle detection
8. `neetcode_word_ladder` - BFS with word transformations

### Learning Path:

Master DFS/BFS on grids first, learn graph cloning and traversal, understand topological sort, learn Union-Find for connected components, then tackle complex BFS problems.

---

## 15. 1-D DP

**Prerequisites:** Arrays & Hashing

**Exercise Count:** 12

### All Exercises (Easy → Hard):

1. `neetcode_climbing_stairs` ⭐ Easy
2. `neetcode_min_cost_stairs` ⭐ Easy
3. `neetcode_house_robber` ⭐⭐ Medium
4. `neetcode_house_robber_ii` ⭐⭐ Medium
5. `neetcode_longest_palindrome` ⭐⭐ Medium
6. `neetcode_palindromic_substrings` ⭐⭐ Medium
7. `neetcode_decode_ways` ⭐⭐ Medium
8. `neetcode_coin_change` ⭐⭐ Medium
9. `neetcode_max_product_subarray` ⭐⭐ Medium
10. `neetcode_word_break` ⭐⭐ Medium
11. `neetcode_longest_increasing_subseq` ⭐⭐ Medium
12. `neetcode_partition_equal_subset` ⭐⭐ Medium

### Core Workbook Exercises (8):

1. `neetcode_climbing_stairs` - Basic DP template
2. `neetcode_house_robber` - Non-adjacent selection
3. `neetcode_house_robber_ii` - Circular array variation
4. `neetcode_decode_ways` - Path counting with constraints
5. `neetcode_coin_change` - Unbounded knapsack (minimum)
6. `neetcode_word_break` - DP with string matching
7. `neetcode_longest_increasing_subseq` - O(n log n) DP
8. `neetcode_partition_equal_subset` - 0/1 knapsack (subset sum)

### Learning Path:

Start with Fibonacci-style DP, learn decision-making patterns (rob/not rob), master unbounded knapsack, then tackle more complex string and subsequence problems.

---

## 16. 2-D DP

**Prerequisites:** 1-D DP

**Exercise Count:** 11

### All Exercises (Easy → Hard):

1. `neetcode_unique_paths` ⭐⭐ Medium
2. `neetcode_longest_increasing_path` ⭐⭐⭐ Hard
3. `neetcode_coin_change_ii` ⭐⭐ Medium
4. `neetcode_target_sum` ⭐⭐ Medium
5. `neetcode_interleaving_string` ⭐⭐ Medium
6. `neetcode_lcs` ⭐⭐ Medium (Longest Common Subsequence)
7. `neetcode_edit_distance` ⭐⭐⭐ Hard
8. `neetcode_distinct_subsequences` ⭐⭐⭐ Hard
9. `neetcode_burst_balloons` ⭐⭐⭐ Hard
10. `neetcode_regex_matching` ⭐⭐⭐ Hard
11. `neetcode_stock_cooldown` ⭐⭐ Medium

### Core Workbook Exercises (7):

1. `neetcode_unique_paths` - Basic 2D DP grid
2. `neetcode_coin_change_ii` - Combination counting
3. `neetcode_lcs` - Classic string DP
4. `neetcode_edit_distance` - Levenshtein distance
5. `neetcode_interleaving_string` - 2D string matching
6. `neetcode_stock_cooldown` - State machine DP
7. `neetcode_burst_balloons` - Interval DP

### Learning Path:

Start with grid path problems, learn string DP patterns (LCS, edit distance), understand state machine DP, then tackle interval DP and complex matching problems.

---

## 17. Bit Manipulation

**Prerequisites:** Arrays & Hashing

**Exercise Count:** 6

### All Exercises (Easy → Hard):

1. `neetcode_single_number` ⭐ Easy
2. `neetcode_hamming_weight` ⭐ Easy
3. `neetcode_counting_bits` ⭐ Easy
4. `neetcode_reverse_bits` ⭐ Easy
5. `neetcode_missing_number` ⭐ Easy (also in Arrays)
6. `neetcode_sum_two_integers` ⭐⭐ Medium

### Core Workbook Exercises (6):

All 6 exercises are core - focused set of bit manipulation patterns.

1. `neetcode_single_number` - XOR properties
2. `neetcode_hamming_weight` - Bit counting
3. `neetcode_counting_bits` - DP + bit manipulation
4. `neetcode_reverse_bits` - Bit reversal
5. `neetcode_missing_number` - XOR for missing element
6. `neetcode_sum_two_integers` - Bitwise addition

### Learning Path:

Master XOR properties first, learn bit counting techniques, understand bit reversal, then tackle arithmetic operations using bitwise operators.

---

## 18. Math & Geometry

**Prerequisites:** Arrays & Hashing

**Exercise Count:** 9

### All Exercises (Easy → Hard):

1. `neetcode_plus_one` ⭐ Easy (also in Arrays)
2. `neetcode_happy_number` ⭐ Easy (also in Arrays)
3. `neetcode_reverse_integer` ⭐⭐ Medium
4. `neetcode_pow` ⭐⭐ Medium
5. `neetcode_multiply_strings` ⭐⭐ Medium
6. `neetcode_rotate_image` ⭐⭐ Medium (also in Arrays)
7. `neetcode_spiral_matrix` ⭐⭐ Medium (also in Arrays)
8. `neetcode_set_matrix_zeroes` ⭐⭐ Medium (also in Arrays)
9. `neetcode_detect_squares` ⭐⭐ Medium (also in Arrays)

### Core Workbook Exercises (6):

1. `neetcode_happy_number` - Cycle detection in sequences
2. `neetcode_pow` - Fast exponentiation
3. `neetcode_multiply_strings` - String arithmetic
4. `neetcode_rotate_image` - Matrix transformation
5. `neetcode_spiral_matrix` - Boundary traversal
6. `neetcode_detect_squares` - Geometry with hashing

### Learning Path:

Learn fast exponentiation and modular arithmetic, master matrix transformations, understand boundary traversal patterns, then tackle geometry problems.

---

## Recommended Learning Sequence

### Phase 1: Foundations (Weeks 1-4)
1. Arrays & Hashing (17 exercises) - 2 weeks
2. Two Pointers (5 exercises) - 3 days
3. Stack (7 exercises) - 4 days
4. Binary Search (7 exercises) - 4 days

### Phase 2: Data Structures (Weeks 5-8)
5. Sliding Window (6 exercises) - 3 days
6. Linked List (11 exercises) - 1 week
7. Trees (15 exercises) - 1.5 weeks
8. Tries (3 exercises) - 2 days
9. Heap/Priority Queue (8 exercises) - 4 days

### Phase 3: Algorithms & Patterns (Weeks 9-12)
10. Intervals (6 exercises) - 3 days
11. Greedy (8 exercises) - 4 days
12. Backtracking (10 exercises) - 5 days
13. Graphs (13 exercises) - 1 week

### Phase 4: Advanced Topics (Weeks 13-16)
14. 1-D DP (12 exercises) - 1 week
15. 2-D DP (11 exercises) - 1 week
16. Advanced Graphs (6 exercises) - 4 days
17. Bit Manipulation (6 exercises) - 3 days
18. Math & Geometry (9 exercises) - 4 days

---

## Topic Dependency Graph

```
Arrays & Hashing (1)
├── Two Pointers (2)
│   └── Sliding Window (5)
├── Stack (3)
├── Binary Search (4)
├── Linked List (6)
├── Trees (7)
│   └── Tries (8)
├── Heap/Priority Queue (9)
│   └── Advanced Graphs (12)
│       [requires Graphs (14)]
├── Intervals (10)
├── Greedy (11)
├── Backtracking (13)
├── Graphs (14)
│   └── Advanced Graphs (12)
├── 1-D DP (15)
│   └── 2-D DP (16)
├── Bit Manipulation (17)
└── Math & Geometry (18)
```

---

## Workbook Creation Guidelines

When creating thematic workbooks from this mapping:

### Beginner Workbooks (4-6 exercises each):
- **Arrays & Hashing Fundamentals:** contains_duplicate, valid_anagram, two_sum, group_anagrams
- **Two Pointers Basics:** valid_palindrome, two_sum_ii, 3sum
- **Stack Essentials:** valid_parentheses, min_stack, daily_temperatures
- **Tree Fundamentals:** invert_tree, max_depth, same_tree, diameter_tree

### Intermediate Workbooks (6-8 exercises each):
- **Sliding Window Mastery:** All 6 sliding window exercises
- **Linked List Patterns:** reverse_linked_list, reorder_list, copy_random_list, lru_cache
- **Binary Search Variations:** binary_search, find_min_rotated, search_rotated, koko_bananas
- **Graph Traversal:** num_islands, pacific_atlantic, rotting_oranges, course_schedule

### Advanced Workbooks (6-8 exercises each):
- **Dynamic Programming 1D:** house_robber, coin_change, word_break, longest_increasing_subseq
- **Dynamic Programming 2D:** unique_paths, lcs, edit_distance, burst_balloons
- **Backtracking Patterns:** subsets, permutations, combination_sum, n_queens
- **Advanced Graphs:** network_delay_time, min_cost_connect_points, alien_dictionary

---

## Notes

- Some exercises appear in multiple topics (e.g., `neetcode_missing_number` in both Arrays & Bit Manipulation)
- Difficulty ratings are based on typical NeetCode/LeetCode classifications
- Core workbook exercises represent the essential patterns that unlock the entire topic
- Prerequisites are recommendations, not strict requirements
- Total practice time estimate: 12-16 weeks at ~10 exercises per week

---

*Last updated: 2025-11-21*
*Total NeetCode exercises: 151*
