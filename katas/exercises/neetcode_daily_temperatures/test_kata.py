"""Tests for Daily Temperatures kata."""

def test_daily_temperatures_example1():
    from template import daily_temperatures
    assert daily_temperatures([73,74,75,71,69,72,76,73]) == [1,1,4,2,1,1,0,0]

def test_daily_temperatures_example2():
    from template import daily_temperatures
    assert daily_temperatures([30,40,50,60]) == [1,1,1,0]

def test_daily_temperatures_decreasing():
    from template import daily_temperatures
    assert daily_temperatures([60,50,40,30]) == [0,0,0,0]
