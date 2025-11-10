"""Tests for Reverse Integer kata."""

def test_reverse_example1():
    from template import reverse
    assert reverse(123) == 321

def test_reverse_example2():
    from template import reverse
    assert reverse(-123) == -321

def test_reverse_example3():
    from template import reverse
    assert reverse(120) == 21

def test_reverse_zero():
    from template import reverse
    assert reverse(0) == 0

def test_reverse_overflow():
    from template import reverse
    assert reverse(1534236469) == 0
