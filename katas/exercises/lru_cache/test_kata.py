"""Tests for LRU Cache kata."""

import pytest


def test_lru_basic():
    from template import LRUCache

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


def test_lru_update():
    from template import LRUCache

    cache = LRUCache(2)
    cache.put(1, 1)
    cache.put(2, 2)
    cache.put(1, 10)  # update value
    assert cache.get(1) == 10


def test_lru_access_order():
    from template import LRUCache

    cache = LRUCache(2)
    cache.put(1, 1)
    cache.put(2, 2)
    cache.get(1)  # access 1, making 2 the LRU
    cache.put(3, 3)  # should evict 2
    assert cache.get(2) == -1
    assert cache.get(1) == 1
    assert cache.get(3) == 3


def test_lru_capacity_one():
    from template import LRUCache

    cache = LRUCache(1)
    cache.put(1, 1)
    assert cache.get(1) == 1
    cache.put(2, 2)
    assert cache.get(1) == -1
    assert cache.get(2) == 2


def test_lru_not_found():
    from template import LRUCache

    cache = LRUCache(2)
    assert cache.get(1) == -1


def test_lru_multiple_operations():
    from template import LRUCache

    cache = LRUCache(3)
    cache.put(1, 1)
    cache.put(2, 2)
    cache.put(3, 3)
    cache.get(1)
    cache.get(2)
    cache.put(4, 4)  # evicts 3
    assert cache.get(3) == -1
    assert cache.get(1) == 1
    assert cache.get(2) == 2
    assert cache.get(4) == 4
