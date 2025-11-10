"""Tests for Car Fleet kata."""

def test_car_fleet_example1():
    from template import car_fleet
    assert car_fleet(12, [10,8,0,5,3], [2,4,1,1,3]) == 3

def test_car_fleet_single():
    from template import car_fleet
    assert car_fleet(10, [0], [5]) == 1
