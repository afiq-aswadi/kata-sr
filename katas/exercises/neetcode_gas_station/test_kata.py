"""Tests for Gas Station kata."""

try:
    from user_kata import can_complete_circuit
except ImportError:
    from .reference import can_complete_circuit


def test_can_complete_circuit_example1():
    assert can_complete_circuit([1,2,3,4,5], [3,4,5,1,2]) == 3

def test_can_complete_circuit_example2():
    assert can_complete_circuit([2,3,4], [3,4,3]) == -1

def test_can_complete_circuit_single():
    assert can_complete_circuit([5], [4]) == 0

def test_can_complete_circuit_start_zero():
    assert can_complete_circuit([3,1,1], [1,2,2]) == 0

def test_can_complete_circuit_impossible():
    assert can_complete_circuit([1,2], [2,3]) == -1
