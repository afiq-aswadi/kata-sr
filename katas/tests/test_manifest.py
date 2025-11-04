"""Tests for manifest parser."""

from pathlib import Path

import pytest

from manifest_parser import KataManifest, validate_kata_directory


def test_parse_valid_manifest(tmp_path: Path):
    """Test parsing a valid manifest."""
    manifest_content = """
[kata]
name = "test_kata"
category = "transformers"
base_difficulty = 4
description = "A test kata"
dependencies = ["prerequisite_kata"]

[[variations]]
name = "test_variation"
description = "A test variation"

[variations.params]
param1 = "value1"
param2 = 42
param3 = 3.14
param4 = [1, 2, 3]

[variations.params.param5]
inner = "x"
"""
    manifest_path = tmp_path / "manifest.toml"
    manifest_path.write_text(manifest_content)

    manifest = KataManifest.from_file(manifest_path)

    assert manifest.name == "test_kata"
    assert manifest.category == "transformers"
    assert manifest.base_difficulty == 4
    assert manifest.description == "A test kata"
    assert manifest.dependencies == ["prerequisite_kata"]
    assert len(manifest.variations) == 1
    assert manifest.variations[0].name == "test_variation"
    assert manifest.variations[0].description == "A test variation"
    assert manifest.variations[0].params == {
        "param1": "value1",
        "param2": 42,
        "param3": 3.14,
        "param4": [1, 2, 3],
        "param5": {"inner": "x"},
    }


def test_parse_minimal_manifest(tmp_path: Path):
    """Test parsing a minimal manifest without optional fields."""
    manifest_content = """
[kata]
name = "minimal_kata"
category = "basics"
base_difficulty = 1
description = "Minimal kata"
"""
    manifest_path = tmp_path / "manifest.toml"
    manifest_path.write_text(manifest_content)

    manifest = KataManifest.from_file(manifest_path)

    assert manifest.name == "minimal_kata"
    assert manifest.category == "basics"
    assert manifest.base_difficulty == 1
    assert manifest.description == "Minimal kata"
    assert manifest.dependencies == []
    assert manifest.variations == []


def test_parse_missing_kata_section(tmp_path: Path):
    """Test parsing manifest without [kata] section."""
    manifest_content = """
[other_section]
name = "test"
"""
    manifest_path = tmp_path / "manifest.toml"
    manifest_path.write_text(manifest_content)

    with pytest.raises(ValueError, match="Missing \\[kata\\] section"):
        KataManifest.from_file(manifest_path)


def test_parse_missing_required_field(tmp_path: Path):
    """Test parsing manifest with missing required field."""
    manifest_content = """
[kata]
name = "test_kata"
category = "transformers"
description = "Missing difficulty"
"""
    manifest_path = tmp_path / "manifest.toml"
    manifest_path.write_text(manifest_content)

    with pytest.raises(ValueError, match="Missing required field: base_difficulty"):
        KataManifest.from_file(manifest_path)


def test_parse_invalid_difficulty_type(tmp_path: Path):
    """Test parsing manifest with invalid difficulty type."""
    manifest_content = """
[kata]
name = "test_kata"
category = "transformers"
base_difficulty = "hard"
description = "Invalid difficulty type"
"""
    manifest_path = tmp_path / "manifest.toml"
    manifest_path.write_text(manifest_content)

    with pytest.raises(ValueError, match="base_difficulty must be an integer"):
        KataManifest.from_file(manifest_path)


def test_parse_nonexistent_file(tmp_path: Path):
    """Test parsing a nonexistent manifest file."""
    manifest_path = tmp_path / "nonexistent.toml"

    with pytest.raises(FileNotFoundError):
        KataManifest.from_file(manifest_path)


def test_variation_missing_name(tmp_path: Path):
    """Test parsing manifest with variation missing name."""
    manifest_content = """
[kata]
name = "test_kata"
category = "transformers"
base_difficulty = 4
description = "Test kata"

[[variations]]
description = "Missing name"
"""
    manifest_path = tmp_path / "manifest.toml"
    manifest_path.write_text(manifest_content)

    with pytest.raises(ValueError, match="Variation missing required field: name"):
        KataManifest.from_file(manifest_path)


def test_validate_kata_directory_valid(tmp_path: Path):
    """Test validating a complete kata directory."""
    # create manifest
    manifest_content = """
[kata]
name = "test_kata"
category = "transformers"
base_difficulty = 4
description = "A test kata"
"""
    (tmp_path / "manifest.toml").write_text(manifest_content)

    # create required files
    (tmp_path / "template.py").write_text("# template")
    (tmp_path / "test_kata.py").write_text("# tests")

    # should not raise
    manifest = validate_kata_directory(tmp_path)
    assert manifest.name == "test_kata"


def test_validate_kata_directory_missing_template(tmp_path: Path):
    """Test validating kata directory with missing template."""
    manifest_content = """
[kata]
name = "test_kata"
category = "transformers"
base_difficulty = 4
description = "A test kata"
"""
    (tmp_path / "manifest.toml").write_text(manifest_content)
    (tmp_path / "test_kata.py").write_text("# tests")
    # template.py is missing

    with pytest.raises(FileNotFoundError, match="Missing required file: template.py"):
        validate_kata_directory(tmp_path)


def test_validate_kata_directory_missing_tests(tmp_path: Path):
    """Test validating kata directory with missing test file."""
    manifest_content = """
[kata]
name = "test_kata"
category = "transformers"
base_difficulty = 4
description = "A test kata"
"""
    (tmp_path / "manifest.toml").write_text(manifest_content)
    (tmp_path / "template.py").write_text("# template")
    # test_kata.py is missing

    with pytest.raises(FileNotFoundError, match="Missing required file: test_kata.py"):
        validate_kata_directory(tmp_path)


def test_validate_kata_directory_missing_manifest(tmp_path: Path):
    """Test validating kata directory with missing manifest."""
    with pytest.raises(FileNotFoundError, match="No manifest.toml found"):
        validate_kata_directory(tmp_path)
