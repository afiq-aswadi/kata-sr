"""Tests for Reconstruct Itinerary kata."""

def test_find_itinerary_example1():
    from template import find_itinerary
    tickets = [["MUC","LHR"],["JFK","MUC"],["SFO","SJC"],["LHR","SFO"]]
    assert find_itinerary(tickets) == ["JFK","MUC","LHR","SFO","SJC"]

def test_find_itinerary_example2():
    from template import find_itinerary
    tickets = [["JFK","SFO"],["JFK","ATL"],["SFO","ATL"],["ATL","JFK"],["ATL","SFO"]]
    assert find_itinerary(tickets) == ["JFK","ATL","JFK","SFO","ATL","SFO"]

def test_find_itinerary_simple():
    from template import find_itinerary
    tickets = [["JFK","ATL"],["ATL","JFK"]]
    assert find_itinerary(tickets) == ["JFK","ATL","JFK"]

def test_find_itinerary_linear():
    from template import find_itinerary
    tickets = [["JFK","KUL"],["KUL","NRT"]]
    assert find_itinerary(tickets) == ["JFK","KUL","NRT"]
