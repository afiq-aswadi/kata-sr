"""Tests for singleton metaclass kata."""

import pytest


try:
    from user_kata import SingletonMeta
except ImportError:
    from .reference import SingletonMeta


def test_singleton_basic():
    """Test singleton metaclass creates only one instance."""

    class Database(metaclass=SingletonMeta):
        def __init__(self):
            self.connection = "connected"

    db1 = Database()
    db2 = Database()

    assert db1 is db2
    assert db1.connection == "connected"


def test_singleton_multiple_instantiations():
    """Test multiple calls return same instance."""

    class Service(metaclass=SingletonMeta):
        pass

    instances = [Service() for _ in range(10)]

    # All instances should be the same object
    for instance in instances:
        assert instance is instances[0]


def test_singleton_different_classes():
    """Test singleton works independently for different classes."""

    class ServiceA(metaclass=SingletonMeta):
        pass

    class ServiceB(metaclass=SingletonMeta):
        pass

    a1 = ServiceA()
    a2 = ServiceA()
    b1 = ServiceB()
    b2 = ServiceB()

    # Same class instances should be identical
    assert a1 is a2
    assert b1 is b2

    # Different class instances should not be identical
    assert a1 is not b1


def test_singleton_with_constructor_args():
    """Test singleton handles constructor arguments."""

    class Config(metaclass=SingletonMeta):
        def __init__(self, value=None):
            # Only initialize on first creation
            if not hasattr(self, 'initialized'):
                self.value = value
                self.initialized = True

    c1 = Config(value=42)
    c2 = Config(value=99)  # This should return c1, not create new

    assert c1 is c2
    assert c1.value == 42  # Should keep first value


def test_singleton_state_shared():
    """Test that singleton instances share state."""

    class Counter(metaclass=SingletonMeta):
        def __init__(self):
            if not hasattr(self, 'count'):
                self.count = 0

        def increment(self):
            self.count += 1

    c1 = Counter()
    c2 = Counter()

    c1.increment()
    assert c2.count == 1  # c2 sees c1's changes

    c2.increment()
    assert c1.count == 2  # c1 sees c2's changes


def test_singleton_with_inheritance():
    """Test singleton works with class inheritance."""

    class BaseService(metaclass=SingletonMeta):
        pass

    class EmailService(BaseService):
        pass

    class SMSService(BaseService):
        pass

    e1 = EmailService()
    e2 = EmailService()
    s1 = SMSService()
    s2 = SMSService()

    # Each subclass should have its own singleton instance
    assert e1 is e2
    assert s1 is s2
    assert e1 is not s1


def test_singleton_preserves_class_name():
    """Test that singleton preserves class metadata."""

    class MyService(metaclass=SingletonMeta):
        """Service documentation."""
        pass

    s = MyService()

    assert MyService.__name__ == 'MyService'
    assert MyService.__doc__ == "Service documentation."
    assert type(s).__name__ == 'MyService'


def test_singleton_isinstance_check():
    """Test that singleton instances pass isinstance checks."""

    class AppConfig(metaclass=SingletonMeta):
        pass

    config = AppConfig()

    assert isinstance(config, AppConfig)
    assert type(config) == AppConfig


def test_singleton_with_methods():
    """Test singleton class with methods."""

    class Logger(metaclass=SingletonMeta):
        def __init__(self):
            if not hasattr(self, 'logs'):
                self.logs = []

        def log(self, message):
            self.logs.append(message)

    log1 = Logger()
    log2 = Logger()

    log1.log("First message")
    log2.log("Second message")

    assert len(log1.logs) == 2
    assert len(log2.logs) == 2
    assert log1.logs == log2.logs
