"""Tests for dynamic programming kata."""



try:
    from user_kata import fibonacci_memo
    from user_kata import fibonacci_tabulation
    from user_kata import knapsack
    from user_kata import longest_common_subsequence
    from user_kata import coin_change
except ImportError:
    from .reference import fibonacci_memo
    from .reference import fibonacci_tabulation
    from .reference import knapsack
    from .reference import longest_common_subsequence
    from .reference import coin_change


def test_fibonacci_memo():

    assert fibonacci_memo(0) == 0
    assert fibonacci_memo(1) == 1
    assert fibonacci_memo(5) == 5
    assert fibonacci_memo(10) == 55
    assert fibonacci_memo(20) == 6765


def test_fibonacci_tabulation():

    assert fibonacci_tabulation(0) == 0
    assert fibonacci_tabulation(1) == 1
    assert fibonacci_tabulation(5) == 5
    assert fibonacci_tabulation(10) == 55
    assert fibonacci_tabulation(20) == 6765


def test_knapsack_basic():

    weights = [2, 3, 4, 5]
    values = [3, 4, 5, 6]
    capacity = 5

    result = knapsack(weights, values, capacity)
    assert result == 7  # items 0 and 1


def test_knapsack_zero_capacity():

    weights = [1, 2, 3]
    values = [10, 20, 30]
    capacity = 0

    assert knapsack(weights, values, capacity) == 0


def test_longest_common_subsequence():

    assert longest_common_subsequence("ABCDE", "ACE") == 3
    assert longest_common_subsequence("AGGTAB", "GXTXAYB") == 4
    assert longest_common_subsequence("", "ABC") == 0
    assert longest_common_subsequence("ABC", "DEF") == 0


def test_coin_change_basic():

    coins = [1, 2, 5]
    assert coin_change(coins, 11) == 3  # 5+5+1
    assert coin_change(coins, 3) == 2  # 2+1


def test_coin_change_impossible():

    coins = [2]
    assert coin_change(coins, 3) == -1


def test_coin_change_zero():

    coins = [1, 2, 5]
    assert coin_change(coins, 0) == 0
