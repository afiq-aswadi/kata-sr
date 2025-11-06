"""Tests for operator overloading kata."""

import pytest


def test_vector_add():
    from template import Vector

    v1 = Vector(1, 2)
    v2 = Vector(3, 4)
    result = v1 + v2

    assert result.x == 4
    assert result.y == 6


def test_vector_sub():
    from template import Vector

    v1 = Vector(5, 7)
    v2 = Vector(2, 3)
    result = v1 - v2

    assert result.x == 3
    assert result.y == 4


def test_vector_mul():
    from template import Vector

    v = Vector(2, 3)
    result = v * 3

    assert result.x == 6
    assert result.y == 9


def test_vector_eq():
    from template import Vector

    v1 = Vector(1, 2)
    v2 = Vector(1, 2)
    v3 = Vector(2, 3)

    assert v1 == v2
    assert not (v1 == v3)


def test_money_add():
    from template import Money

    m1 = Money(10.50, "USD")
    m2 = Money(5.25, "USD")
    result = m1 + m2

    assert result.amount == 15.75
    assert result.currency == "USD"


def test_money_add_different_currency():
    from template import Money

    m1 = Money(10, "USD")
    m2 = Money(10, "EUR")

    with pytest.raises(ValueError):
        m1 + m2


def test_money_comparison():
    from template import Money

    m1 = Money(10, "USD")
    m2 = Money(20, "USD")

    assert m1 < m2
    assert not (m2 < m1)


def test_money_eq():
    from template import Money

    m1 = Money(10.50, "USD")
    m2 = Money(10.50, "USD")
    m3 = Money(10.50, "EUR")

    assert m1 == m2
    assert not (m1 == m3)


def test_matrix_getitem():
    from template import Matrix

    m = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    assert m[0, 0] == 1
    assert m[1, 2] == 6
    assert m[2, 1] == 8


def test_matrix_setitem():
    from template import Matrix

    m = Matrix([[1, 2], [3, 4]])
    m[0, 1] = 10

    assert m[0, 1] == 10


def test_matrix_len():
    from template import Matrix

    m = Matrix([[1, 2], [3, 4], [5, 6]])
    assert len(m) == 3
