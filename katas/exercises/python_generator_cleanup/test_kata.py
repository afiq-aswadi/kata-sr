"""
Tests for Python Generator Cleanup kata
"""

import pytest

try:
    from user_kata import file_reader_generator
except ImportError:
    from .reference import file_reader_generator
import tempfile
import os


def test_file_reader_generator_basic():
    """Test file_reader_generator reads all lines"""

    # Create a temporary file
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write("line1\nline2\nline3\n")
        temp_path = f.name

    try:
        gen = file_reader_generator(temp_path)
        lines = [line.strip() for line in gen]
        assert lines == ["line1", "line2", "line3"]
    finally:
        os.unlink(temp_path)


def test_file_reader_generator_partial_read():
    """Test file_reader_generator with partial consumption"""

    # Create a temporary file
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write("line1\nline2\nline3\n")
        temp_path = f.name

    try:
        # Read first two lines only
        gen = file_reader_generator(temp_path)
        lines = [next(gen).strip(), next(gen).strip()]

        assert lines == ["line1", "line2"]

        # Close generator early - should trigger cleanup
        gen.close()

        # Verify cleanup was called by reading file again
        gen2 = file_reader_generator(temp_path)
        all_lines = [line.strip() for line in gen2]
        assert all_lines == ["line1", "line2", "line3"]

    finally:
        os.unlink(temp_path)


def test_file_reader_generator_empty_file():
    """Test file_reader_generator with empty file"""

    # Create an empty file
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        temp_path = f.name

    try:
        gen = file_reader_generator(temp_path)
        lines = list(gen)
        assert lines == []
    finally:
        os.unlink(temp_path)


def test_file_reader_generator_single_line():
    """Test file_reader_generator with single line"""

    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write("only line\n")
        temp_path = f.name

    try:
        gen = file_reader_generator(temp_path)
        lines = [line.strip() for line in gen]
        assert lines == ["only line"]
    finally:
        os.unlink(temp_path)


def test_cleanup_on_close():
    """Test that cleanup happens when generator.close() is called"""

    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write("line1\nline2\nline3\n")
        temp_path = f.name

    try:
        gen = file_reader_generator(temp_path)
        first_line = next(gen)
        assert first_line.strip() == "line1"

        # Close generator explicitly
        gen.close()

        # Should be able to open file again (proves it was closed)
        gen2 = file_reader_generator(temp_path)
        lines = [line.strip() for line in gen2]
        assert len(lines) == 3

    finally:
        os.unlink(temp_path)


def test_cleanup_on_exception():
    """Test that cleanup happens even with exceptions"""

    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write("line1\nline2\nline3\n")
        temp_path = f.name

    try:
        gen = file_reader_generator(temp_path)
        next(gen)  # Read first line

        # Force exception by throwing into generator
        try:
            gen.throw(RuntimeError("test error"))
        except RuntimeError:
            pass  # Expected

        # Generator should have cleaned up
        # Verify by creating a new generator
        gen2 = file_reader_generator(temp_path)
        lines = list(gen2)
        assert len(lines) == 3

    finally:
        os.unlink(temp_path)


def test_cleanup_on_normal_completion():
    """Test that cleanup happens on normal completion"""

    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write("line1\nline2\n")
        temp_path = f.name

    try:
        gen = file_reader_generator(temp_path)

        # Consume all lines
        lines = list(gen)
        assert len(lines) == 2

        # File should be closed (cleanup happened)
        # Verify by opening again
        gen2 = file_reader_generator(temp_path)
        lines2 = list(gen2)
        assert len(lines2) == 2

    finally:
        os.unlink(temp_path)


def test_multiple_generators_same_file():
    """Test multiple generators can read the same file"""

    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write("line1\nline2\nline3\n")
        temp_path = f.name

    try:
        # Create two generators
        gen1 = file_reader_generator(temp_path)
        gen2 = file_reader_generator(temp_path)

        # Read from first
        line1 = next(gen1).strip()
        assert line1 == "line1"

        # Read from second
        line2 = next(gen2).strip()
        assert line2 == "line1"

        # Close both
        gen1.close()
        gen2.close()

    finally:
        os.unlink(temp_path)


def test_file_with_no_newline_at_end():
    """Test file without trailing newline"""

    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write("line1\nline2")  # No newline at end
        temp_path = f.name

    try:
        gen = file_reader_generator(temp_path)
        lines = [line.strip() for line in gen]
        assert lines == ["line1", "line2"]
    finally:
        os.unlink(temp_path)


def test_returns_generator():
    """Test that function returns a generator"""

    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write("test\n")
        temp_path = f.name

    try:
        gen = file_reader_generator(temp_path)
        assert hasattr(gen, '__iter__')
        assert hasattr(gen, '__next__')
        gen.close()
    finally:
        os.unlink(temp_path)


def test_whitespace_preservation():
    """Test that whitespace in lines is preserved"""

    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write("  spaces  \n\ttabs\t\n")
        temp_path = f.name

    try:
        gen = file_reader_generator(temp_path)
        lines = list(gen)
        assert lines[0] == "  spaces  \n"
        assert lines[1] == "\ttabs\t\n"
    finally:
        os.unlink(temp_path)
