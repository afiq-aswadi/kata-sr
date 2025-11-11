"""Tests for create_class_with_init kata."""

import pytest


try:
    from user_kata import create_class_with_init
except ImportError:
    from .reference import create_class_with_init


def test_positional_arguments():
    """Test class with __init__ using positional arguments."""

    Person = create_class_with_init('Person', ['name', 'age'])

    p = Person('Alice', 30)
    assert p.name == 'Alice'
    assert p.age == 30


def test_keyword_arguments():
    """Test class with __init__ using keyword arguments."""

    Person = create_class_with_init('Person', ['name', 'age'])

    p = Person(name='Bob', age=25)
    assert p.name == 'Bob'
    assert p.age == 25


def test_mixed_arguments():
    """Test class with __init__ using mixed arguments."""

    Person = create_class_with_init('Person', ['name', 'age', 'city'])

    p = Person('Charlie', age=35, city='NYC')
    assert p.name == 'Charlie'
    assert p.age == 35
    assert p.city == 'NYC'


def test_multiple_instances():
    """Test that multiple instances maintain separate attributes."""

    Point = create_class_with_init('Point', ['x', 'y'])

    p1 = Point(1, 2)
    p2 = Point(3, 4)

    assert p1.x == 1
    assert p1.y == 2
    assert p2.x == 3
    assert p2.y == 4


def test_single_parameter():
    """Test class with single init parameter."""

    Counter = create_class_with_init('Counter', ['value'])

    c = Counter(42)
    assert c.value == 42


def test_many_parameters():
    """Test class with many init parameters."""

    Config = create_class_with_init('Config', ['host', 'port', 'db', 'user'])

    cfg = Config('localhost', 5432, 'mydb', 'admin')
    assert cfg.host == 'localhost'
    assert cfg.port == 5432
    assert cfg.db == 'mydb'
    assert cfg.user == 'admin'


def test_string_attributes():
    """Test init with string values."""

    User = create_class_with_init('User', ['username', 'email'])

    u = User('alice', 'alice@example.com')
    assert u.username == 'alice'
    assert u.email == 'alice@example.com'
