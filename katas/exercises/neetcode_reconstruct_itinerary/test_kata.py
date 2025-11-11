"""Tests for Reconstruct Itinerary kata."""

try:
    from user_kata import find_itinerary
except ImportError:
    from .reference import find_itinerary


def test_find_itinerary_example1():
    tickets = [["MUC","LHR"],["JFK","MUC"],["SFO","SJC"],["LHR","SFO"]]
    assert find_itinerary(tickets) == ["JFK","MUC","LHR","SFO","SJC"]

def test_find_itinerary_example2():
    tickets = [["JFK","SFO"],["JFK","ATL"],["SFO","ATL"],["ATL","JFK"],["ATL","SFO"]]
    assert find_itinerary(tickets) == ["JFK","ATL","JFK","SFO","ATL","SFO"]

def test_find_itinerary_simple():
    tickets = [["JFK","ATL"],["ATL","JFK"]]
    assert find_itinerary(tickets) == ["JFK","ATL","JFK"]

def test_find_itinerary_linear():
    tickets = [["JFK","KUL"],["KUL","NRT"]]
    assert find_itinerary(tickets) == ["JFK","KUL","NRT"]
