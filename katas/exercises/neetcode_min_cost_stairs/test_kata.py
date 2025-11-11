"""Tests for Min Cost Climbing Stairs kata."""

try:
    from user_kata import min_cost_climbing_stairs
except ImportError:
    from .reference import min_cost_climbing_stairs


def test_min_cost_example1():
    assert min_cost_climbing_stairs([10,15,20]) == 15

def test_min_cost_example2():
    assert min_cost_climbing_stairs([1,100,1,1,1,100,1,1,100,1]) == 6

def test_min_cost_two_steps():
    assert min_cost_climbing_stairs([10, 15]) == 10

def test_min_cost_equal():
    assert min_cost_climbing_stairs([1,1,1,1]) == 2
