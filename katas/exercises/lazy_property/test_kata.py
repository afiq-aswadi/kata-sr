"""
Tests for Lazy Property Kata
"""

import pytest
import time


def test_lazy_property_evaluates_once(solution):
    """Test that lazy_property only evaluates once."""
    lazy_property = solution.lazy_property

    eval_count = 0

    class MyClass:
        @lazy_property
        def expensive_property(self):
            nonlocal eval_count
            eval_count += 1
            time.sleep(0.01)
            return "computed"

    obj = MyClass()

    # First access - should compute
    result1 = obj.expensive_property
    assert result1 == "computed"
    assert eval_count == 1

    # Second access - should use cached value
    result2 = obj.expensive_property
    assert result2 == "computed"
    assert eval_count == 1  # Should not increment


def test_lazy_property_separate_instances(solution):
    """Test that lazy_property maintains separate cache per instance."""
    lazy_property = solution.lazy_property

    class Counter:
        def __init__(self, value):
            self.value = value

        @lazy_property
        def doubled(self):
            return self.value * 2

    obj1 = Counter(5)
    obj2 = Counter(10)

    assert obj1.doubled == 10
    assert obj2.doubled == 20
    assert obj1.doubled == 10  # Still cached separately


def test_lazy_property_class_access(solution):
    """Test that accessing lazy_property on class returns the descriptor."""
    lazy_property = solution.lazy_property

    class MyClass:
        @lazy_property
        def prop(self):
            return "value"

    # Accessing via class should return the descriptor itself
    descriptor = MyClass.prop
    assert isinstance(descriptor, lazy_property)


def test_lazy_property_preserves_metadata(solution):
    """Test that lazy_property preserves function metadata."""
    lazy_property = solution.lazy_property

    class MyClass:
        @lazy_property
        def documented_property(self):
            """This is a documented property."""
            return 42

    # Check that metadata is preserved on the descriptor
    descriptor = MyClass.documented_property
    assert descriptor.__name__ == "documented_property"
    assert "documented property" in descriptor.__doc__


def test_lazy_property_with_computation(solution):
    """Test lazy_property with actual computation."""
    lazy_property = solution.lazy_property

    class DataProcessor:
        def __init__(self, data):
            self.data = data
            self.computation_count = 0

        @lazy_property
        def processed_data(self):
            self.computation_count += 1
            return [x * 2 for x in self.data]

    processor = DataProcessor([1, 2, 3, 4, 5])

    # First access
    result1 = processor.processed_data
    assert result1 == [2, 4, 6, 8, 10]
    assert processor.computation_count == 1

    # Second access
    result2 = processor.processed_data
    assert result2 == [2, 4, 6, 8, 10]
    assert processor.computation_count == 1  # No recomputation


def test_lazy_property_caches_in_instance_dict(solution):
    """Test that lazy_property stores cached value in instance __dict__."""
    lazy_property = solution.lazy_property

    class MyClass:
        @lazy_property
        def prop(self):
            return "cached_value"

    obj = MyClass()

    # Before first access, should not be in __dict__
    assert "prop" not in obj.__dict__

    # Access the property
    result = obj.prop
    assert result == "cached_value"

    # After access, should be in __dict__
    assert "prop" in obj.__dict__
    assert obj.__dict__["prop"] == "cached_value"


def test_lazy_property_with_expensive_computation(solution):
    """Test lazy_property actually improves performance by caching."""
    lazy_property = solution.lazy_property

    class ExpensiveClass:
        @lazy_property
        def slow_computation(self):
            time.sleep(0.05)
            return "result"

    obj = ExpensiveClass()

    # First access - should be slow
    start1 = time.perf_counter()
    result1 = obj.slow_computation
    time1 = time.perf_counter() - start1

    # Second access - should be fast (cached)
    start2 = time.perf_counter()
    result2 = obj.slow_computation
    time2 = time.perf_counter() - start2

    assert result1 == result2 == "result"
    assert time1 > 0.04  # First call should take time
    assert time2 < 0.01  # Second call should be instant (cached)


def test_lazy_property_multiple_properties(solution):
    """Test multiple lazy properties on the same class."""
    lazy_property = solution.lazy_property

    class MultiProp:
        def __init__(self):
            self.count_a = 0
            self.count_b = 0

        @lazy_property
        def prop_a(self):
            self.count_a += 1
            return "a"

        @lazy_property
        def prop_b(self):
            self.count_b += 1
            return "b"

    obj = MultiProp()

    # Access prop_a
    assert obj.prop_a == "a"
    assert obj.count_a == 1
    assert obj.count_b == 0

    # Access prop_b
    assert obj.prop_b == "b"
    assert obj.count_a == 1
    assert obj.count_b == 1

    # Access both again
    assert obj.prop_a == "a"
    assert obj.prop_b == "b"
    assert obj.count_a == 1  # No recomputation
    assert obj.count_b == 1  # No recomputation


def test_lazy_property_with_none_value(solution):
    """Test that lazy_property correctly caches None values."""
    lazy_property = solution.lazy_property

    eval_count = 0

    class MyClass:
        @lazy_property
        def returns_none(self):
            nonlocal eval_count
            eval_count += 1
            return None

    obj = MyClass()

    result1 = obj.returns_none
    assert result1 is None
    assert eval_count == 1

    result2 = obj.returns_none
    assert result2 is None
    assert eval_count == 1  # Should still be 1 (cached)


def test_lazy_property_inherits_correctly(solution):
    """Test that lazy_property works with inheritance."""
    lazy_property = solution.lazy_property

    class Base:
        @lazy_property
        def prop(self):
            return "base"

    class Derived(Base):
        pass

    obj = Derived()
    assert obj.prop == "base"

    # Check caching works
    assert "prop" in obj.__dict__
