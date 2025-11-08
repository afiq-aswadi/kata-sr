"""Tests for dynamic programming kata."""



def test_fibonacci_memo():
    from template import fibonacci_memo

    assert fibonacci_memo(0) == 0
    assert fibonacci_memo(1) == 1
    assert fibonacci_memo(5) == 5
    assert fibonacci_memo(10) == 55
    assert fibonacci_memo(20) == 6765


def test_fibonacci_tabulation():
    from template import fibonacci_tabulation

    assert fibonacci_tabulation(0) == 0
    assert fibonacci_tabulation(1) == 1
    assert fibonacci_tabulation(5) == 5
    assert fibonacci_tabulation(10) == 55
    assert fibonacci_tabulation(20) == 6765


def test_knapsack_basic():
    from template import knapsack

    weights = [2, 3, 4, 5]
    values = [3, 4, 5, 6]
    capacity = 5

    result = knapsack(weights, values, capacity)
    assert result == 7  # items 0 and 1


def test_knapsack_zero_capacity():
    from template import knapsack

    weights = [1, 2, 3]
    values = [10, 20, 30]
    capacity = 0

    assert knapsack(weights, values, capacity) == 0


def test_longest_common_subsequence():
    from template import longest_common_subsequence

    assert longest_common_subsequence("ABCDE", "ACE") == 3
    assert longest_common_subsequence("AGGTAB", "GXTXAYB") == 4
    assert longest_common_subsequence("", "ABC") == 0
    assert longest_common_subsequence("ABC", "DEF") == 0


def test_coin_change_basic():
    from template import coin_change

    coins = [1, 2, 5]
    assert coin_change(coins, 11) == 3  # 5+5+1
    assert coin_change(coins, 3) == 2  # 2+1


def test_coin_change_impossible():
    from template import coin_change

    coins = [2]
    assert coin_change(coins, 3) == -1


def test_coin_change_zero():
    from template import coin_change

    coins = [1, 2, 5]
    assert coin_change(coins, 0) == 0
