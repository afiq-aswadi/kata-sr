"""Tests for dynamic class creation kata."""

import pytest


def test_create_simple_class_basic():
    """Test creating a simple class with attributes."""
    from template import create_simple_class

    MyClass = create_simple_class('MyClass', {'value': 42, 'name': 'test'})

    assert MyClass.__name__ == 'MyClass'
    assert MyClass.value == 42
    assert MyClass.name == 'test'


def test_create_simple_class_instantiation():
    """Test that dynamically created class can be instantiated."""
    from template import create_simple_class

    MyClass = create_simple_class('MyClass', {'default': 100})

    obj = MyClass()
    assert obj.default == 100


def test_create_class_with_base_inheritance():
    """Test creating a class that inherits from a base class."""
    from template import create_class_with_base

    class Animal:
        kingdom = "Animalia"

        def breathe(self):
            return "breathing"

    Dog = create_class_with_base('Dog', Animal, {'sound': 'woof', 'legs': 4})

    assert Dog.__name__ == 'Dog'
    assert issubclass(Dog, Animal)
    assert Dog.sound == 'woof'
    assert Dog.legs == 4
    assert Dog.kingdom == 'Animalia'

    dog = Dog()
    assert dog.breathe() == "breathing"
    assert dog.sound == 'woof'


def test_create_class_with_base_override():
    """Test that child class can override parent attributes."""
    from template import create_class_with_base

    class Base:
        value = 10

    Derived = create_class_with_base('Derived', Base, {'value': 20})

    assert Base.value == 10
    assert Derived.value == 20


def test_create_class_with_methods_single():
    """Test creating a class with a single method."""
    from template import create_class_with_methods

    def greet(self):
        return "Hello!"

    MyClass = create_class_with_methods('MyClass', {'greet': greet})

    obj = MyClass()
    assert obj.greet() == "Hello!"


def test_create_class_with_methods_multiple():
    """Test creating a class with multiple methods."""
    from template import create_class_with_methods

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


def test_create_class_with_methods_and_attributes():
    """Test creating a class with both methods and attributes."""
    from template import create_class_with_methods

    def get_full_name(self):
        return f"{self.first} {self.last}"

    Person = create_class_with_methods('Person', {
        'first': 'John',
        'last': 'Doe',
        'get_full_name': get_full_name
    })

    p = Person()
    assert p.get_full_name() == "John Doe"


def test_create_class_with_init_positional():
    """Test class with __init__ using positional arguments."""
    from template import create_class_with_init

    Person = create_class_with_init('Person', ['name', 'age'])

    p = Person('Alice', 30)
    assert p.name == 'Alice'
    assert p.age == 30


def test_create_class_with_init_keyword():
    """Test class with __init__ using keyword arguments."""
    from template import create_class_with_init

    Person = create_class_with_init('Person', ['name', 'age'])

    p = Person(name='Bob', age=25)
    assert p.name == 'Bob'
    assert p.age == 25


def test_create_class_with_init_mixed():
    """Test class with __init__ using mixed arguments."""
    from template import create_class_with_init

    Person = create_class_with_init('Person', ['name', 'age', 'city'])

    p = Person('Charlie', age=35, city='NYC')
    assert p.name == 'Charlie'
    assert p.age == 35
    assert p.city == 'NYC'


def test_create_class_with_init_multiple_instances():
    """Test that multiple instances maintain separate attributes."""
    from template import create_class_with_init

    Point = create_class_with_init('Point', ['x', 'y'])

    p1 = Point(1, 2)
    p2 = Point(3, 4)

    assert p1.x == 1
    assert p1.y == 2
    assert p2.x == 3
    assert p2.y == 4


def test_type_is_type():
    """Test that created classes are actually type objects."""
    from template import create_simple_class

    MyClass = create_simple_class('MyClass', {})

    assert isinstance(MyClass, type)
    assert type(MyClass) == type
