"""Tests for validated metaclass kata."""

import pytest


def test_validated_meta_success():
    """Test ValidatedMeta allows class with all required attributes."""
    from template import ValidatedMeta

    class User(metaclass=ValidatedMeta):
        required_attributes = ['name', 'email']
        name = "John"
        email = "john@example.com"

    u = User()
    assert u.name == "John"
    assert u.email == "john@example.com"


def test_validated_meta_missing_one():
    """Test ValidatedMeta raises error for missing attribute."""
    from template import ValidatedMeta

    with pytest.raises(TypeError, match="missing required attribute: email"):
        class InvalidUser(metaclass=ValidatedMeta):
            required_attributes = ['name', 'email']
            name = "John"
            # email is missing


def test_validated_meta_missing_multiple():
    """Test ValidatedMeta detects missing attributes."""
    from template import ValidatedMeta

    with pytest.raises(TypeError, match="missing required attribute"):
        class Config(metaclass=ValidatedMeta):
            required_attributes = ['host', 'port', 'database']
            host = "localhost"
            # port and database are missing


def test_validated_meta_no_requirements():
    """Test ValidatedMeta works without required_attributes."""
    from template import ValidatedMeta

    class Product(metaclass=ValidatedMeta):
        price = 100

    p = Product()
    assert p.price == 100


def test_validated_meta_empty_requirements():
    """Test ValidatedMeta works with empty required_attributes."""
    from template import ValidatedMeta

    class Service(metaclass=ValidatedMeta):
        required_attributes = []
        value = 42

    s = Service()
    assert s.value == 42


def test_validated_meta_all_required_present():
    """Test ValidatedMeta accepts class with many required attributes."""
    from template import ValidatedMeta

    class Model(metaclass=ValidatedMeta):
        required_attributes = ['a', 'b', 'c', 'd']
        a = 1
        b = 2
        c = 3
        d = 4

    m = Model()
    assert m.a == 1
    assert m.d == 4


def test_validated_meta_with_methods():
    """Test ValidatedMeta works with methods."""
    from template import ValidatedMeta

    class Calculator(metaclass=ValidatedMeta):
        required_attributes = ['name']
        name = "BasicCalc"

        def add(self, a, b):
            return a + b

    calc = Calculator()
    assert calc.add(2, 3) == 5


def test_validated_meta_none_value():
    """Test ValidatedMeta accepts None as a valid value."""
    from template import ValidatedMeta

    class Config(metaclass=ValidatedMeta):
        required_attributes = ['setting']
        setting = None

    c = Config()
    assert c.setting is None


def test_typed_meta_success():
    """Test TypedMeta allows correct types."""
    from template import TypedMeta

    class Person(metaclass=TypedMeta):
        attribute_types = {'name': str, 'age': int}
        name = "Alice"
        age = 30

    p = Person()
    assert p.name == "Alice"
    assert p.age == 30


def test_typed_meta_wrong_type():
    """Test TypedMeta raises error for wrong type."""
    from template import TypedMeta

    with pytest.raises(TypeError, match="must be int, got str"):
        class Person(metaclass=TypedMeta):
            attribute_types = {'age': int}
            age = "thirty"


def test_typed_meta_multiple_wrong():
    """Test TypedMeta detects type errors."""
    from template import TypedMeta

    with pytest.raises(TypeError, match="must be"):
        class Config(metaclass=TypedMeta):
            attribute_types = {'host': str, 'port': int}
            host = "localhost"
            port = "8080"  # Should be int


def test_typed_meta_no_validation():
    """Test TypedMeta works without attribute_types."""
    from template import TypedMeta

    class Simple(metaclass=TypedMeta):
        value = 42

    s = Simple()
    assert s.value == 42


def test_typed_meta_partial_validation():
    """Test TypedMeta only validates specified attributes."""
    from template import TypedMeta

    class Mixed(metaclass=TypedMeta):
        attribute_types = {'name': str}
        name = "Alice"
        age = 30  # Not validated
        count = "many"  # Not validated

    m = Mixed()
    assert m.name == "Alice"
    assert m.age == 30


def test_typed_meta_bool_type():
    """Test TypedMeta with bool type."""
    from template import TypedMeta

    class Flags(metaclass=TypedMeta):
        attribute_types = {'active': bool, 'debug': bool}
        active = True
        debug = False

    f = Flags()
    assert f.active is True
    assert f.debug is False


def test_typed_meta_list_type():
    """Test TypedMeta with list type."""
    from template import TypedMeta

    class Container(metaclass=TypedMeta):
        attribute_types = {'items': list}
        items = [1, 2, 3]

    c = Container()
    assert c.items == [1, 2, 3]


def test_typed_meta_custom_class():
    """Test TypedMeta with custom class type."""
    from template import TypedMeta

    class Address:
        pass

    class Person(metaclass=TypedMeta):
        attribute_types = {'address': Address}
        address = Address()

    p = Person()
    assert isinstance(p.address, Address)


def test_both_metaclasses_combined_pattern():
    """Test pattern combining validation concepts."""
    from template import ValidatedMeta

    class StrictModel(metaclass=ValidatedMeta):
        required_attributes = ['id', 'name']
        id = 1
        name = "Model"

        def __init__(self):
            self.data = []

    model = StrictModel()
    assert model.id == 1
    assert model.name == "Model"
    assert model.data == []
