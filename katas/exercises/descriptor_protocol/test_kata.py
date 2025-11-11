"""Tests for descriptor protocol kata."""

import pytest


try:
    from user_kata import ValidatedAttribute
    from user_kata import CachedProperty
    from user_kata import TypedAttribute
except ImportError:
    from .reference import ValidatedAttribute
    from .reference import CachedProperty
    from .reference import TypedAttribute


def test_validated_attribute():

    class Product:
        price = ValidatedAttribute(min_value=0, max_value=10000)

        def __init__(self, price):
            self.price = price

    p = Product(100)
    assert p.price == 100

    p.price = 500
    assert p.price == 500


def test_validated_attribute_min():

    class Product:
        price = ValidatedAttribute(min_value=0)

    p = Product()
    with pytest.raises(ValueError):
        p.price = -10


def test_validated_attribute_max():

    class Product:
        quantity = ValidatedAttribute(max_value=100)

    p = Product()
    with pytest.raises(ValueError):
        p.quantity = 200


def test_cached_property():

    call_count = 0

    class DataLoader:
        @CachedProperty
        def expensive_data(self):
            nonlocal call_count
            call_count += 1
            return "computed data"

    loader = DataLoader()
    assert call_count == 0

    data1 = loader.expensive_data
    assert call_count == 1
    assert data1 == "computed data"

    data2 = loader.expensive_data
    assert call_count == 1  # Should not recompute
    assert data2 == "computed data"


def test_typed_attribute():

    class Person:
        name = TypedAttribute(str)
        age = TypedAttribute(int)

    p = Person()
    p.name = "Alice"
    p.age = 30

    assert p.name == "Alice"
    assert p.age == 30


def test_typed_attribute_wrong_type():

    class Person:
        age = TypedAttribute(int)

    p = Person()
    with pytest.raises(TypeError):
        p.age = "thirty"


def test_multiple_instances():

    class Product:
        price = ValidatedAttribute(min_value=0)

    p1 = Product()
    p2 = Product()

    p1.price = 100
    p2.price = 200

    assert p1.price == 100
    assert p2.price == 200
