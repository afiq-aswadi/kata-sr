"""Tests for Cheapest Flights Within K Stops kata."""

def test_find_cheapest_price_example1():
    from template import find_cheapest_price
    n = 4
    flights = [[0,1,100],[1,2,100],[2,0,100],[1,3,600],[2,3,200]]
    assert find_cheapest_price(n, flights, 0, 3, 1) == 700

def test_find_cheapest_price_example2():
    from template import find_cheapest_price
    n = 3
    flights = [[0,1,100],[1,2,100],[0,2,500]]
    assert find_cheapest_price(n, flights, 0, 2, 1) == 200

def test_find_cheapest_price_example3():
    from template import find_cheapest_price
    n = 3
    flights = [[0,1,100],[1,2,100],[0,2,500]]
    assert find_cheapest_price(n, flights, 0, 2, 0) == 500

def test_find_cheapest_price_no_route():
    from template import find_cheapest_price
    n = 3
    flights = [[0,1,100]]
    assert find_cheapest_price(n, flights, 0, 2, 1) == -1

def test_find_cheapest_price_direct():
    from template import find_cheapest_price
    n = 2
    flights = [[0,1,100]]
    assert find_cheapest_price(n, flights, 0, 1, 0) == 100
