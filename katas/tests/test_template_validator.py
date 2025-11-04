"""Tests for template validator."""

from pathlib import Path

from template_validator import count_todos, extract_blanks, is_template_filled


def test_extract_blanks_single_region(tmp_path: Path):
    """Test extracting a single BLANK region."""
    template_content = """
def foo():
    # BLANK_START
    return 42
    # BLANK_END
"""
    template_path = tmp_path / "template.py"
    template_path.write_text(template_content)

    blanks = extract_blanks(template_path)

    assert len(blanks) == 1
    assert blanks[0].start_line == 3
    assert blanks[0].end_line == 5
    assert blanks[0].content.strip() == "return 42"


def test_extract_blanks_multiple_regions(tmp_path: Path):
    """Test extracting multiple BLANK regions."""
    template_content = """
def foo():
    # BLANK_START
    x = 1
    # BLANK_END
    return x

def bar():
    # BLANK_START
    y = 2
    z = 3
    # BLANK_END
    return y + z
"""
    template_path = tmp_path / "template.py"
    template_path.write_text(template_content)

    blanks = extract_blanks(template_path)

    assert len(blanks) == 2
    assert blanks[0].start_line == 3
    assert blanks[0].end_line == 5
    assert "x = 1" in blanks[0].content
    assert blanks[1].start_line == 9
    assert blanks[1].end_line == 12
    assert "y = 2" in blanks[1].content
    assert "z = 3" in blanks[1].content


def test_extract_blanks_todo_marker(tmp_path: Path):
    """Test extracting TODO markers as blanks."""
    template_content = """
def foo():
    # TODO: implement this
    pass
"""
    template_path = tmp_path / "template.py"
    template_path.write_text(template_content)

    blanks = extract_blanks(template_path)

    # TODO markers without BLANK_END don't create complete regions
    assert len(blanks) == 0


def test_extract_blanks_empty_template(tmp_path: Path):
    """Test extracting blanks from empty template."""
    template_path = tmp_path / "template.py"
    template_path.write_text("")

    blanks = extract_blanks(template_path)

    assert len(blanks) == 0


def test_count_todos_multiple(tmp_path: Path):
    """Test counting multiple TODO markers."""
    template_content = """
def foo():
    # TODO: implement this
    pass

def bar():
    # TODO implement that
    pass

# TODO: another one
"""
    template_path = tmp_path / "template.py"
    template_path.write_text(template_content)

    count = count_todos(template_path)

    assert count == 3


def test_count_todos_none(tmp_path: Path):
    """Test counting TODOs when there are none."""
    template_content = """
def foo():
    return 42
"""
    template_path = tmp_path / "template.py"
    template_path.write_text(template_content)

    count = count_todos(template_path)

    assert count == 0


def test_count_todos_case_sensitive(tmp_path: Path):
    """Test that TODO counting handles spacing."""
    template_content = """
# TODO: with colon
#TODO without space
# TODO without colon
"""
    template_path = tmp_path / "template.py"
    template_path.write_text(template_content)

    count = count_todos(template_path)

    assert count == 3


def test_is_template_filled_complete(tmp_path: Path):
    """Test filled template detection for complete template."""
    template_content = """
def foo():
    # BLANK_START
    return 42
    # BLANK_END
"""
    template_path = tmp_path / "template.py"
    template_path.write_text(template_content)

    assert is_template_filled(template_path) is True


def test_is_template_filled_has_todos(tmp_path: Path):
    """Test filled template detection when TODOs remain."""
    template_content = """
def foo():
    # TODO: implement this
    pass
"""
    template_path = tmp_path / "template.py"
    template_path.write_text(template_content)

    assert is_template_filled(template_path) is False


def test_is_template_filled_blank_with_pass(tmp_path: Path):
    """Test filled template detection when blank only has pass."""
    template_content = """
def foo():
    # BLANK_START
    pass
    # BLANK_END
"""
    template_path = tmp_path / "template.py"
    template_path.write_text(template_content)

    assert is_template_filled(template_path) is False


def test_is_template_filled_empty_blank(tmp_path: Path):
    """Test filled template detection when blank is empty."""
    template_content = """
def foo():
    # BLANK_START
    # BLANK_END
"""
    template_path = tmp_path / "template.py"
    template_path.write_text(template_content)

    assert is_template_filled(template_path) is False


def test_is_template_filled_no_markers(tmp_path: Path):
    """Test filled template detection when there are no markers."""
    template_content = """
def foo():
    return 42
"""
    template_path = tmp_path / "template.py"
    template_path.write_text(template_content)

    # no TODOs and no blank regions means filled
    assert is_template_filled(template_path) is True
