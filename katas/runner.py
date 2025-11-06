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
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
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

    def pytest_runtest_logreport(self, report: "pytest.TestReport") -> None:
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

    def pytest_sessionstart(self, session: "pytest.Session") -> None:
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
    added_sys_paths: list[str] = []

    def _ensure_sys_path(path: Path) -> None:
        path_str = str(path.resolve())
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
            added_sys_paths.append(path_str)

    def _cleanup_sys_path() -> None:
        while added_sys_paths:
            path_str = added_sys_paths.pop()
            try:
                sys.path.remove(path_str)
            except ValueError:
                continue

    def _ensure_local_site_packages() -> None:
        candidate_dirs = list(Path(__file__).resolve().parents)
        for base in candidate_dirs:
            venv_dir = base / ".venv"
            if not venv_dir.exists():
                continue

            lib_dir = venv_dir / "lib"
            if lib_dir.exists():
                for site_dir in lib_dir.glob("python*/site-packages"):
                    if site_dir.is_dir():
                        _ensure_sys_path(site_dir)

            windows_site = venv_dir / "Lib" / "site-packages"
            if windows_site.exists():
                _ensure_sys_path(windows_site)

            # stop after first venv we find
            break

    _ensure_sys_path(Path(__file__).parent)
    _ensure_local_site_packages()

    try:
        import pytest
    except ModuleNotFoundError:
        _cleanup_sys_path()
        return {
            "passed": False,
            "error": (
                "pytest is required to run kata tests. "
                "Run `uv sync` inside the `katas/` directory (or activate its virtualenv) "
                "and try again."
            ),
            "num_passed": 0,
            "num_failed": 0,
            "num_skipped": 0,
            "duration_ms": 0,
            "results": [],
        }

    _ensure_sys_path(template_path.parent)

    previous_module = sys.modules.get("user_kata")

    try:
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
        except Exception as exc:
            return {
                "passed": False,
                "error": f"Failed to import template: {exc}",
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

        reporter = JSONReporter()
        start = time.time()

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
    finally:
        if previous_module is not None:
            sys.modules["user_kata"] = previous_module
        else:
            sys.modules.pop("user_kata", None)
        _cleanup_sys_path()


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
