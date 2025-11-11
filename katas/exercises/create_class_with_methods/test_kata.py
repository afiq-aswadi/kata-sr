"""Tests for create_class_with_methods kata."""

import pytest


try:
    from user_kata import create_class_with_methods
except ImportError:
    from .reference import create_class_with_methods


def test_single_method():
    """Test creating a class with a single method."""

    def greet(self):
        return "Hello!"

    MyClass = create_class_with_methods('MyClass', {'greet': greet})

    obj = MyClass()
    assert obj.greet() == "Hello!"


def test_multiple_methods():
    """Test creating a class with multiple methods."""

    def add(self, a, b):
        return a + b

    def multiply(self, a, b):
        return a * b

    Calculator = create_class_with_methods('Calculator', {
        'add': add,
        'multiply': multiply
    })

    calc = Calculator()
    assert calc.add(2, 3) == 5
    assert calc.multiply(4, 5) == 20


def test_method_with_self():
    """Test that methods receive self parameter."""

    def get_instance(self):
        return self

    MyClass = create_class_with_methods('MyClass', {'get_instance': get_instance})

    obj = MyClass()
    assert obj.get_instance() is obj


def test_methods_and_attributes():
    """Test class with both methods and attributes."""

    def get_full_name(self):
        return f"{self.first} {self.last}"

    Person = create_class_with_methods('Person', {
        'first': 'John',
        'last': 'Doe',
        'get_full_name': get_full_name
    })

    p = Person()
    assert p.get_full_name() == "John Doe"


def test_method_accessing_attributes():
    """Test method that accesses instance attributes."""

    def double_value(self):
        return self.value * 2

    MyClass = create_class_with_methods('MyClass', {
        'value': 10,
        'double_value': double_value
    })

    obj = MyClass()
    assert obj.double_value() == 20


def test_multiple_instances():
    """Test methods work independently across instances."""

    def set_value(self, val):
        self.data = val

    def get_value(self):
        return getattr(self, 'data', None)

    Container = create_class_with_methods('Container', {
        'set_value': set_value,
        'get_value': get_value
    })

    c1 = Container()
    c2 = Container()

    c1.set_value(100)
    c2.set_value(200)

    assert c1.get_value() == 100
    assert c2.get_value() == 200


def test_method_with_args_and_kwargs():
    """Test method that accepts various arguments."""

    def format_message(self, name, greeting="Hello"):
        return f"{greeting}, {name}!"

    Greeter = create_class_with_methods('Greeter', {
        'format_message': format_message
    })

    g = Greeter()
    assert g.format_message("Alice") == "Hello, Alice!"
    assert g.format_message("Bob", greeting="Hi") == "Hi, Bob!"
