"""Tests for LRU Cache kata."""

try:
    from user_kata import LRUCache
except ImportError:
    from .reference import LRUCache


def test_lru_cache_basic():

    cache = LRUCache(2)
    cache.put(1, 1)
    cache.put(2, 2)
    assert cache.get(1) == 1
    cache.put(3, 3)  # evicts key 2
    assert cache.get(2) == -1
    cache.put(4, 4)  # evicts key 1
    assert cache.get(1) == -1
    assert cache.get(3) == 3
    assert cache.get(4) == 4

def test_lru_cache_update():

    cache = LRUCache(2)
    cache.put(1, 1)
    cache.put(2, 2)
    cache.put(1, 10)  # update key 1
    assert cache.get(1) == 10
    cache.put(3, 3)  # evicts key 2
    assert cache.get(2) == -1
    assert cache.get(1) == 10
    assert cache.get(3) == 3

def test_lru_cache_single_capacity():

    cache = LRUCache(1)
    cache.put(1, 1)
    assert cache.get(1) == 1
    cache.put(2, 2)  # evicts key 1
    assert cache.get(1) == -1
    assert cache.get(2) == 2

def test_lru_cache_get_updates_recency():

    cache = LRUCache(2)
    cache.put(1, 1)
    cache.put(2, 2)
    cache.get(1)  # makes 1 recently used
    cache.put(3, 3)  # should evict key 2, not 1
    assert cache.get(1) == 1
    assert cache.get(2) == -1
    assert cache.get(3) == 3
