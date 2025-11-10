"""Tests for Time Based Key-Value Store kata."""

def test_timemap_example():
    from template import TimeMap
    tm = TimeMap()
    tm.set("foo", "bar", 1)
    assert tm.get("foo", 1) == "bar"
    assert tm.get("foo", 3) == "bar"
    tm.set("foo", "bar2", 4)
    assert tm.get("foo", 4) == "bar2"
    assert tm.get("foo", 5) == "bar2"

def test_timemap_multiple_keys():
    from template import TimeMap
    tm = TimeMap()
    tm.set("love", "high", 10)
    tm.set("love", "low", 20)
    assert tm.get("love", 5) == ""
    assert tm.get("love", 10) == "high"
    assert tm.get("love", 15) == "high"
    assert tm.get("love", 20) == "low"
    assert tm.get("love", 25) == "low"

def test_timemap_nonexistent_key():
    from template import TimeMap
    tm = TimeMap()
    assert tm.get("nonexistent", 1) == ""

def test_timemap_exact_timestamps():
    from template import TimeMap
    tm = TimeMap()
    tm.set("a", "val1", 1)
    tm.set("a", "val2", 5)
    tm.set("a", "val3", 10)
    assert tm.get("a", 1) == "val1"
    assert tm.get("a", 5) == "val2"
    assert tm.get("a", 10) == "val3"
