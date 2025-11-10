"""Tests for Min Cost Climbing Stairs kata."""

def test_min_cost_example1():
    from template import min_cost_climbing_stairs
    assert min_cost_climbing_stairs([10,15,20]) == 15

def test_min_cost_example2():
    from template import min_cost_climbing_stairs
    assert min_cost_climbing_stairs([1,100,1,1,1,100,1,1,100,1]) == 6

def test_min_cost_two_steps():
    from template import min_cost_climbing_stairs
    assert min_cost_climbing_stairs([10, 15]) == 10

def test_min_cost_equal():
    from template import min_cost_climbing_stairs
    assert min_cost_climbing_stairs([1,1,1,1]) == 2
