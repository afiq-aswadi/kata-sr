"""Tests for context manager kata."""

import os
import tempfile
import time

import pytest


def test_timer():
    from template import Timer

    with Timer() as timer:
        time.sleep(0.1)

    assert timer.elapsed >= 0.1
    assert timer.elapsed < 0.2


def test_temporary_value():
    from template import TemporaryValue

    class Config:
        debug = False

    config = Config()
    assert config.debug is False

    with TemporaryValue(config, "debug", True):
        assert config.debug is True

    assert config.debug is False


def test_temporary_value_with_exception():
    from template import TemporaryValue

    class Config:
        value = 10

    config = Config()

    try:
        with TemporaryValue(config, "value", 20):
            assert config.value == 20
            raise ValueError("test error")
    except ValueError:
        pass

    # Should restore even after exception
    assert config.value == 10


def test_suppress_exception():
    from template import SuppressException

    with SuppressException(ValueError):
        raise ValueError("this should be suppressed")

    # Should reach here without exception


def test_suppress_exception_wrong_type():
    from template import SuppressException

    with pytest.raises(TypeError):
        with SuppressException(ValueError):
            raise TypeError("this should NOT be suppressed")


def test_suppress_multiple_exceptions():
    from template import SuppressException

    with SuppressException(ValueError, KeyError):
        raise KeyError("suppressed")

    # Should reach here


def test_file_writer():
    from template import file_writer

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "test.txt")

        with file_writer(filepath) as f:
            f.write("hello world")

        with open(filepath) as f:
            content = f.read()

        assert content == "hello world"
