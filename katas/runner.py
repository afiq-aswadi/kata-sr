"""Pytest runner that outputs structured JSON for Rust to parse.

Usage:
    python -m runner <kata_id> <template_path>

Output (JSON to stdout):
{
    "passed": true,
    "num_passed": 5,
    "num_failed": 0,
    "num_skipped": 0,
    "duration_ms": 234,
    "results": [
        {
            "test_name": "test_output_shape",
            "status": "passed",
            "output": ""
        },
        {
            "test_name": "test_attention_normalized",
            "status": "failed",
            "output": "AssertionError: weights don't sum to 1.0..."
        }
    ]
}
"""

import importlib.util
import json
import sys
import time
from io import StringIO
from pathlib import Path
from typing import TypedDict

import pytest


class TestResult(TypedDict):
    """Individual test result."""

    test_name: str
    status: str
    output: str


class KataTestResults(TypedDict):
    """Results from running kata tests."""

    passed: bool
    num_passed: int
    num_failed: int
    num_skipped: int
    duration_ms: int
    results: list[TestResult]


class KataTestError(TypedDict):
    """Error result when tests cannot run."""

    passed: bool
    error: str
    num_passed: int
    num_failed: int
    num_skipped: int
    duration_ms: int
    results: list[TestResult]


class JSONReporter:
    """Pytest plugin to collect results in memory."""

    def __init__(self) -> None:
        self.results: list[TestResult] = []
        self._node_results: dict[str, TestResult] = {}
        self.start_time: float | None = None

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        """Collect test results during test execution.

        Args:
            report: pytest test report object
        """
        test_name = report.nodeid.split("::")[-1]
        display_name = test_name if report.when == "call" else f"{test_name} ({report.when})"

        output = ""
        if hasattr(report, "longreprtext") and report.longreprtext:
            output = report.longreprtext
        elif hasattr(report, "longrepr") and report.longrepr:
            output = str(report.longrepr)

        outcome = report.outcome

        if report.when == "call":
            self._node_results[report.nodeid] = {
                "test_name": display_name,
                "status": outcome,
                "output": output,
            }
        elif outcome in {"failed", "skipped"}:
            # record setup/teardown errors and skips so they surface in JSON
            self._node_results[report.nodeid] = {
                "test_name": display_name,
                "status": outcome,
                "output": output,
            }

        self.results = list(self._node_results.values())

    def pytest_sessionstart(self, session: pytest.Session) -> None:
        """Record start time when pytest session begins.

        Args:
            session: pytest session object
        """
        self.start_time = time.time()


def run_kata_tests(kata_module_path: Path, template_path: Path) -> KataTestResults | KataTestError:
    """Run tests for a kata exercise.

    Args:
        kata_module_path: path to kata directory (e.g., exercises/multihead_attention/)
        template_path: path to user's filled-in template (e.g., /tmp/kata_123.py)

    Returns:
        JSON-serializable dict with test results
    """
    # import the user's template as a module
    try:
        spec = importlib.util.spec_from_file_location("user_kata", template_path)
        if spec is None or spec.loader is None:
            return {
                "passed": False,
                "error": f"Could not load template: {template_path}",
                "num_passed": 0,
                "num_failed": 0,
                "num_skipped": 0,
                "duration_ms": 0,
                "results": [],
            }

        user_module = importlib.util.module_from_spec(spec)
        sys.modules["user_kata"] = user_module
        spec.loader.exec_module(user_module)
    except Exception as e:
        return {
            "passed": False,
            "error": f"Failed to import template: {e}",
            "num_passed": 0,
            "num_failed": 0,
            "num_skipped": 0,
            "duration_ms": 0,
            "results": [],
        }

    # ensure there are tests to run
    test_files = sorted(p for p in kata_module_path.rglob("test_*.py") if p.is_file())
    if not test_files:
        return {
            "passed": False,
            "error": f"No test files found in {kata_module_path}",
            "num_passed": 0,
            "num_failed": 0,
            "num_skipped": 0,
            "duration_ms": 0,
            "results": [],
        }

    # run pytest with custom reporter
    reporter = JSONReporter()
    start = time.time()

    # capture stdout/stderr to prevent pytest output from mixing with JSON
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = StringIO()
    sys.stderr = StringIO()

    try:
        exit_code = pytest.main(
            [*(str(path) for path in test_files), "-v", "--tb=short"],
            plugins=[reporter],
        )
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr

    duration_ms = int((time.time() - start) * 1000)

    results = reporter.results

    # aggregate results
    num_passed = sum(1 for r in results if r["status"] == "passed")
    num_failed = sum(1 for r in results if r["status"] == "failed")
    num_skipped = sum(1 for r in results if r["status"] == "skipped")

    return {
        "passed": exit_code == 0,
        "num_passed": num_passed,
        "num_failed": num_failed,
        "num_skipped": num_skipped,
        "duration_ms": duration_ms,
        "results": results,
    }


def main() -> None:
    """CLI entry point for runner."""
    if len(sys.argv) != 3:
        print(json.dumps({"error": "Usage: python -m runner <kata_id> <template_path>"}))
        sys.exit(1)

    kata_id = sys.argv[1]
    template_path = Path(sys.argv[2])

    # resolve kata module path from kata_id
    # assume kata_id is the directory name under exercises/
    kata_module_path = Path(__file__).parent / "exercises" / kata_id

    if not kata_module_path.exists():
        print(json.dumps({"error": f"Kata not found: {kata_id}"}))
        sys.exit(1)

    if not template_path.exists():
        print(json.dumps({"error": f"Template not found: {template_path}"}))
        sys.exit(1)

    # run tests and output JSON
    results = run_kata_tests(kata_module_path, template_path)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
