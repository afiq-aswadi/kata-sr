"""Integration tests for pytest runner."""

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

from runner import KataTestError, run_kata_tests

ROOT_DIR = Path(__file__).resolve().parent.parent


def run_kata_tests_in_subprocess(kata_dir: Path, template_path: Path) -> dict[str, Any]:
    """Invoke run_kata_tests in a fresh Python process to avoid nested pytest state."""
    command = [
        sys.executable,
        "-c",
        (
            "import json, pathlib, runner, sys\n"
            "kata_dir = pathlib.Path(sys.argv[1])\n"
            "template = pathlib.Path(sys.argv[2])\n"
            "result = runner.run_kata_tests(kata_dir, template)\n"
            "json.dump(result, sys.stdout)"
        ),
        str(kata_dir),
        str(template_path),
    ]
    completed = subprocess.run(
        command,
        cwd=str(ROOT_DIR),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr or completed.stdout)
    return cast(dict[str, Any], json.loads(completed.stdout))


def test_run_kata_tests_all_passing(tmp_path: Path):
    """Test running kata tests when all tests pass."""
    # create kata directory
    kata_dir = tmp_path / "test_kata"
    kata_dir.mkdir()

    # create test file
    test_content = """
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent.parent))

def test_addition():
    from user_kata import add
    assert add(1, 2) == 3

def test_subtraction():
    from user_kata import subtract
    assert subtract(5, 3) == 2
"""
    (kata_dir / "test_kata.py").write_text(test_content)

    # create user template
    template_path = tmp_path / "user_template.py"
    template_content = """
def add(a, b):
    return a + b

def subtract(a, b):
    return a - b
"""
    template_path.write_text(template_content)

    # run tests
    results = run_kata_tests_in_subprocess(kata_dir, template_path)

    assert results["passed"] is True, results
    assert results["num_passed"] == 2, results
    assert results["num_failed"] == 0, results
    assert results["num_skipped"] == 0
    assert results["duration_ms"] > 0
    assert len(results["results"]) == 2
    assert all(r["status"] == "passed" for r in results["results"])


def test_run_kata_tests_some_failing(tmp_path: Path):
    """Test running kata tests when some tests fail."""
    kata_dir = tmp_path / "test_kata"
    kata_dir.mkdir()

    test_content = """
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent.parent))

def test_passing():
    from user_kata import add
    assert add(1, 2) == 3

def test_failing():
    from user_kata import add
    assert add(1, 2) == 4  # intentionally wrong
"""
    (kata_dir / "test_kata.py").write_text(test_content)

    template_path = tmp_path / "user_template.py"
    template_content = """
def add(a, b):
    return a + b
"""
    template_path.write_text(template_content)

    results = run_kata_tests_in_subprocess(kata_dir, template_path)

    assert results["passed"] is False
    assert results["results"], results
    assert results["num_passed"] == 1, results
    assert results["num_failed"] == 1
    assert results["num_skipped"] == 0
    assert len(results["results"]) == 2

    # check that failure has output
    failing_test = next(r for r in results["results"] if r["status"] == "failed")
    assert "test_failing" in failing_test["test_name"]
    assert failing_test["output"] != ""


def test_run_kata_tests_import_error(tmp_path: Path):
    """Test running kata tests when user template has import error."""
    kata_dir = tmp_path / "test_kata"
    kata_dir.mkdir()

    test_content = """
def test_something():
    assert True
"""
    (kata_dir / "test_kata.py").write_text(test_content)

    template_path = tmp_path / "user_template.py"
    template_content = """
import nonexistent_module  # this will fail
"""
    template_path.write_text(template_content)

    results = run_kata_tests_in_subprocess(kata_dir, template_path)

    assert results["passed"] is False
    assert "error" in results
    error_results = cast(KataTestError, results)
    assert "Failed to import template" in error_results["error"]


def test_run_kata_tests_missing_test_file(tmp_path: Path):
    """Test running kata tests when test file is missing."""
    kata_dir = tmp_path / "test_kata"
    kata_dir.mkdir()
    # no test_kata.py created

    template_path = tmp_path / "user_template.py"
    template_path.write_text("def add(a, b): return a + b")

    results = run_kata_tests_in_subprocess(kata_dir, template_path)

    assert results["passed"] is False
    assert "error" in results
    error_results = cast(KataTestError, results)
    assert "No test files found" in error_results["error"]


def test_run_kata_tests_setup_failure(tmp_path: Path):
    """Test that setup failures surface in the JSON output."""
    kata_dir = tmp_path / "test_kata"
    kata_dir.mkdir()

    test_content = """
import pytest

@pytest.fixture(autouse=True)
def failing_fixture():
    raise RuntimeError("boom")

def test_something():
    assert True
"""
    (kata_dir / "test_setup.py").write_text(test_content)

    template_path = tmp_path / "user_template.py"
    template_path.write_text("pass")

    results = run_kata_tests(kata_dir, template_path)

    assert results["passed"] is False
    assert results["num_failed"] == 1
    assert results["num_skipped"] == 0
    assert len(results["results"]) == 1
    entry = results["results"][0]
    assert entry["status"] == "failed"
    assert "setup" in entry["test_name"]
    assert "boom" in entry["output"]


def test_run_kata_tests_syntax_error_in_template(tmp_path: Path):
    """Test running kata tests when template has syntax error."""
    kata_dir = tmp_path / "test_kata"
    kata_dir.mkdir()

    test_content = """
def test_something():
    assert True
"""
    (kata_dir / "test_kata.py").write_text(test_content)

    template_path = tmp_path / "user_template.py"
    template_content = """
def add(a, b)  # missing colon
    return a + b
"""
    template_path.write_text(template_content)

    results = run_kata_tests(kata_dir, template_path)

    assert results["passed"] is False
    assert "error" in results


def test_run_kata_tests_with_torch(tmp_path: Path):
    """Test running kata tests with torch operations."""
    kata_dir = tmp_path / "test_kata"
    kata_dir.mkdir()

    test_content = """
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent.parent))
import torch

def test_tensor_shape():
    from user_kata import create_tensor
    t = create_tensor(3, 4)
    assert t.shape == (3, 4)

def test_tensor_sum():
    from user_kata import tensor_sum
    t = torch.tensor([1.0, 2.0, 3.0])
    assert tensor_sum(t) == 6.0
"""
    (kata_dir / "test_kata.py").write_text(test_content)

    template_path = tmp_path / "user_template.py"
    template_content = """
import torch

def create_tensor(rows, cols):
    return torch.zeros(rows, cols)

def tensor_sum(t):
    return t.sum().item()
"""
    template_path.write_text(template_content)

    results = run_kata_tests(kata_dir, template_path)

    assert results["results"], results
    assert results["passed"] is True, results
    assert results["num_passed"] == 2, results
    assert results["num_failed"] == 0, results
