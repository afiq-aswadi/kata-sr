"""Tests for create_class_with_inheritance kata."""

import pytest


def test_basic_inheritance():
    """Test creating a class that inherits from a base class."""
    from template import create_class_with_inheritance

    class Animal:
        kingdom = "Animalia"

    Dog = create_class_with_inheritance('Dog', Animal, {'sound': 'woof'})

    assert Dog.__name__ == 'Dog'
    assert issubclass(Dog, Animal)
    assert Dog.sound == 'woof'
    assert Dog.kingdom == 'Animalia'


def test_inherits_methods():
    """Test that child class inherits parent methods."""
    from template import create_class_with_inheritance

    class Base:
        def greet(self):
            return "Hello"

    Child = create_class_with_inheritance('Child', Base, {'value': 42})

    obj = Child()
    assert obj.greet() == "Hello"
    assert obj.value == 42


def test_override_attribute():
    """Test that child class can override parent attributes."""
    from template import create_class_with_inheritance

    class Base:
        value = 10

    Derived = create_class_with_inheritance('Derived', Base, {'value': 20})

    assert Base.value == 10
    assert Derived.value == 20


def test_multiple_attributes():
    """Test child class with multiple new attributes."""
    from template import create_class_with_inheritance

    class Vehicle:
        wheels = 4

    Car = create_class_with_inheritance('Car', Vehicle, {
        'brand': 'Toyota',
        'model': 'Camry'
    })

    assert Car.wheels == 4
    assert Car.brand == 'Toyota'
    assert Car.model == 'Camry'


def test_isinstance_check():
    """Test isinstance works with dynamically created class."""
    from template import create_class_with_inheritance

    class Parent:
        pass

    Child = create_class_with_inheritance('Child', Parent, {})

    obj = Child()
    assert isinstance(obj, Child)
    assert isinstance(obj, Parent)


def test_empty_attributes():
    """Test creating child class with no new attributes."""
    from template import create_class_with_inheritance

    class Base:
        value = 100

    Child = create_class_with_inheritance('Child', Base, {})

    assert Child.value == 100
    obj = Child()
    assert obj.value == 100


def test_method_override():
    """Test overriding parent method."""
    from template import create_class_with_inheritance

    class Base:
        def say(self):
            return "base"

    def child_say(self):
        return "child"

    Child = create_class_with_inheritance('Child', Base, {'say': child_say})

    obj = Child()
    assert obj.say() == "child"
