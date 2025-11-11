"""Tests for Network Delay Time kata."""

try:
    from user_kata import network_delay_time
except ImportError:
    from .reference import network_delay_time


def test_network_delay_time_example1():
    times = [[2,1,1],[2,3,1],[3,4,1]]
    assert network_delay_time(times, 4, 2) == 2

def test_network_delay_time_example2():
    times = [[1,2,1]]
    assert network_delay_time(times, 2, 1) == 1

def test_network_delay_time_example3():
    times = [[1,2,1]]
    assert network_delay_time(times, 2, 2) == -1

def test_network_delay_time_single():
    times = []
    assert network_delay_time(times, 1, 1) == 0

def test_network_delay_time_complex():
    times = [[1,2,1],[2,3,2],[1,3,4]]
    assert network_delay_time(times, 3, 1) == 3
