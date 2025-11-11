"""Tests for Cheapest Flights Within K Stops kata."""

try:
    from user_kata import find_cheapest_price
except ImportError:
    from .reference import find_cheapest_price


def test_find_cheapest_price_example1():
    n = 4
    flights = [[0,1,100],[1,2,100],[2,0,100],[1,3,600],[2,3,200]]
    assert find_cheapest_price(n, flights, 0, 3, 1) == 700

def test_find_cheapest_price_example2():
    n = 3
    flights = [[0,1,100],[1,2,100],[0,2,500]]
    assert find_cheapest_price(n, flights, 0, 2, 1) == 200

def test_find_cheapest_price_example3():
    n = 3
    flights = [[0,1,100],[1,2,100],[0,2,500]]
    assert find_cheapest_price(n, flights, 0, 2, 0) == 500

def test_find_cheapest_price_no_route():
    n = 3
    flights = [[0,1,100]]
    assert find_cheapest_price(n, flights, 0, 2, 1) == -1

def test_find_cheapest_price_direct():
    n = 2
    flights = [[0,1,100]]
    assert find_cheapest_price(n, flights, 0, 1, 0) == 100
