"""Tests for Car Fleet kata."""

try:
    from user_kata import car_fleet
except ImportError:
    from .reference import car_fleet


def test_car_fleet_example1():
    assert car_fleet(12, [10,8,0,5,3], [2,4,1,1,3]) == 3

def test_car_fleet_single():
    assert car_fleet(10, [0], [5]) == 1
