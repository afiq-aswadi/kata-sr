# Agent 2: Python Kata Framework

## Mission

Build the Python infrastructure for kata execution: pytest runner with JSON output, manifest parser, template validation utilities, and base framework for kata exercises. This enables kata authors (Agent 4) to create exercises easily.

## Dependencies

None. You can start immediately in parallel with Agent 1.

## What You're Building

### 1. Pytest Runner (JSON Output)
Command-line tool that runs kata tests and outputs structured JSON results

### 2. Manifest Parser
Parse TOML kata manifests into structured data (for Rust to consume)

### 3. Template Validation
Utilities to extract and validate TODO/BLANK markers in templates

### 4. Base Framework
Optional helper utilities for kata authors (no required framework - keep it minimal)

## Detailed Specifications

### Project Setup

```toml
# katas/pyproject.toml
[project]
name = "kata-framework"
version = "0.1.0"
requires-python = ">=3.13"
dependencies = [
    "pytest>=8.0.0",
    "torch>=2.0.0",
    "einops>=0.7.0",
    "toml>=0.10.2",
]

[tool.uv]
dev-dependencies = [
    "ruff>=0.3.0",
]

[tool.pytest.ini_options]
testpaths = ["exercises"]
python_files = ["test_*.py"]
```

### Pytest Runner with JSON Output

```python
# katas/runner.py

"""
Pytest runner that outputs structured JSON for Rust to parse.

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
            "output": "AssertionError: weights don't sum to 1.0\n  Expected: 1.0\n  Got: 0.98"
        }
    ]
}
"""

import sys
import json
import time
from pathlib import Path
import importlib.util
import pytest
from io import StringIO


class JSONReporter:
    """Pytest plugin to collect results in memory"""

    def __init__(self):
        self.results = []
        self.start_time = None

    def pytest_runtest_logreport(self, report):
        if report.when == "call":
            self.results.append({
                "test_name": report.nodeid.split("::")[-1],
                "status": report.outcome,  # passed, failed, skipped
                "output": report.longreprtext if hasattr(report, "longreprtext") else "",
            })

    def pytest_sessionstart(self, session):
        self.start_time = time.time()


def run_kata_tests(kata_module_path: Path, template_path: Path) -> dict:
    """
    Run tests for a kata exercise.

    Args:
        kata_module_path: Path to kata directory (e.g., exercises/multihead_attention/)
        template_path: Path to user's filled-in template (e.g., /tmp/kata_123.py)

    Returns:
        JSON-serializable dict with test results
    """
    # Import the user's template as a module
    spec = importlib.util.spec_from_file_location("user_kata", template_path)
    user_module = importlib.util.module_from_spec(spec)
    sys.modules["user_kata"] = user_module
    spec.loader.exec_module(user_module)

    # Find test file
    test_file = kata_module_path / "test_kata.py"
    if not test_file.exists():
        return {
            "passed": False,
            "error": f"Test file not found: {test_file}",
            "results": [],
        }

    # Run pytest with custom reporter
    reporter = JSONReporter()
    start = time.time()

    # Capture stdout/stderr
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = StringIO()
    sys.stderr = StringIO()

    try:
        exit_code = pytest.main(
            [str(test_file), "-v", "--tb=short"],
            plugins=[reporter],
        )
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr

    duration_ms = int((time.time() - start) * 1000)

    # Aggregate results
    num_passed = sum(1 for r in reporter.results if r["status"] == "passed")
    num_failed = sum(1 for r in reporter.results if r["status"] == "failed")
    num_skipped = sum(1 for r in reporter.results if r["status"] == "skipped")

    return {
        "passed": exit_code == 0,
        "num_passed": num_passed,
        "num_failed": num_failed,
        "num_skipped": num_skipped,
        "duration_ms": duration_ms,
        "results": reporter.results,
    }


def main():
    if len(sys.argv) != 3:
        print(json.dumps({"error": "Usage: python -m runner <kata_id> <template_path>"}))
        sys.exit(1)

    kata_id = sys.argv[1]
    template_path = Path(sys.argv[2])

    # Resolve kata module path from kata_id
    # Assume kata_id is the directory name under exercises/
    kata_module_path = Path(__file__).parent / "exercises" / kata_id

    if not kata_module_path.exists():
        print(json.dumps({"error": f"Kata not found: {kata_id}"}))
        sys.exit(1)

    if not template_path.exists():
        print(json.dumps({"error": f"Template not found: {template_path}"}))
        sys.exit(1)

    # Run tests and output JSON
    results = run_kata_tests(kata_module_path, template_path)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
```

### Manifest Parser

```python
# katas/manifest_parser.py

"""
Parse kata manifest.toml files into structured data.

This is mainly for validation. Rust will parse manifests directly,
but we provide this for Python-side tooling if needed.
"""

import toml
from pathlib import Path
from dataclasses import dataclass


@dataclass
class KataVariation:
    name: str
    description: str
    params: dict


@dataclass
class KataManifest:
    name: str
    category: str
    base_difficulty: int
    description: str
    dependencies: list[str]
    variations: list[KataVariation]

    @classmethod
    def from_file(cls, path: Path) -> "KataManifest":
        """Load and validate manifest.toml"""
        data = toml.load(path)

        kata = data.get("kata", {})
        if not kata:
            raise ValueError("Missing [kata] section in manifest")

        # Validate required fields
        required = ["name", "category", "base_difficulty", "description"]
        for field in required:
            if field not in kata:
                raise ValueError(f"Missing required field: {field}")

        # Parse variations
        variations = []
        for var_data in data.get("variations", []):
            variations.append(KataVariation(
                name=var_data["name"],
                description=var_data.get("description", ""),
                params=var_data.get("params", {}),
            ))

        return cls(
            name=kata["name"],
            category=kata["category"],
            base_difficulty=kata["base_difficulty"],
            description=kata["description"],
            dependencies=kata.get("dependencies", []),
            variations=variations,
        )

    def validate(self, kata_dir: Path):
        """Validate that required files exist"""
        required_files = ["template.py", "test_kata.py"]
        for filename in required_files:
            if not (kata_dir / filename).exists():
                raise FileNotFoundError(f"Missing required file: {filename}")


def validate_kata_directory(kata_dir: Path):
    """Validate a kata directory structure"""
    manifest_path = kata_dir / "manifest.toml"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No manifest.toml found in {kata_dir}")

    manifest = KataManifest.from_file(manifest_path)
    manifest.validate(kata_dir)

    return manifest


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python manifest_parser.py <kata_dir>")
        sys.exit(1)

    kata_dir = Path(sys.argv[1])
    try:
        manifest = validate_kata_directory(kata_dir)
        print(f"✓ Valid kata: {manifest.name}")
        print(f"  Category: {manifest.category}")
        print(f"  Difficulty: {manifest.base_difficulty}")
        print(f"  Dependencies: {manifest.dependencies}")
        if manifest.variations:
            print(f"  Variations: {len(manifest.variations)}")
    except Exception as e:
        print(f"✗ Invalid kata: {e}")
        sys.exit(1)
```

### Template Validation

```python
# katas/template_validator.py

"""
Extract and validate TODO/BLANK markers in kata templates.
"""

import re
from pathlib import Path
from dataclasses import dataclass


@dataclass
class BlankRegion:
    start_line: int
    end_line: int
    content: str


def extract_blanks(template_path: Path) -> list[BlankRegion]:
    """
    Extract BLANK_START/BLANK_END regions from template.

    Returns list of blank regions with line numbers.
    """
    lines = template_path.read_text().splitlines()
    blanks = []
    current_blank = None

    for i, line in enumerate(lines, start=1):
        if "# BLANK_START" in line or "# TODO" in line:
            if current_blank is None:
                current_blank = {"start": i, "lines": []}
        elif "# BLANK_END" in line:
            if current_blank is not None:
                blanks.append(BlankRegion(
                    start_line=current_blank["start"],
                    end_line=i,
                    content="\n".join(current_blank["lines"]),
                ))
                current_blank = None
        elif current_blank is not None:
            current_blank["lines"].append(line)

    return blanks


def count_todos(template_path: Path) -> int:
    """Count # TODO markers in template"""
    content = template_path.read_text()
    return len(re.findall(r"#\s*TODO", content))


def is_template_filled(template_path: Path) -> bool:
    """
    Check if template has been filled in by user.

    Heuristic: no remaining TODO markers and BLANK regions have content.
    """
    if count_todos(template_path) > 0:
        return False

    blanks = extract_blanks(template_path)
    for blank in blanks:
        # Check if blank region is not just "pass" or empty
        content = blank.content.strip()
        if not content or content == "pass":
            return False

    return True


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python template_validator.py <template.py>")
        sys.exit(1)

    template = Path(sys.argv[1])
    blanks = extract_blanks(template)
    todos = count_todos(template)

    print(f"Template: {template.name}")
    print(f"TODO markers: {todos}")
    print(f"BLANK regions: {len(blanks)}")
    for blank in blanks:
        print(f"  Lines {blank.start_line}-{blank.end_line}")

    if is_template_filled(template):
        print("✓ Template appears to be filled")
    else:
        print("✗ Template needs work")
```

### Base Framework (Optional Helpers)

```python
# katas/framework.py

"""
Optional utilities for kata authors.
Keep this minimal - katas should be standalone Python code.
"""

import torch


def assert_shape(tensor: torch.Tensor, expected_shape: tuple, name: str = "tensor"):
    """Helper for shape assertions in tests"""
    if tensor.shape != expected_shape:
        raise AssertionError(
            f"{name} has wrong shape.\n"
            f"  Expected: {expected_shape}\n"
            f"  Got: {tensor.shape}"
        )


def assert_close(a: torch.Tensor, b: torch.Tensor, rtol: float = 1e-5, name: str = "tensor"):
    """Helper for numerical comparison"""
    if not torch.allclose(a, b, rtol=rtol):
        max_diff = (a - b).abs().max().item()
        raise AssertionError(
            f"{name} values don't match.\n"
            f"  Max difference: {max_diff}\n"
            f"  Tolerance: {rtol}"
        )


# Don't add more - keep it lean!
```

## File Structure You'll Create

```
katas/
├── pyproject.toml
├── __init__.py
├── runner.py                 # Main pytest runner with JSON output
├── manifest_parser.py        # Parse/validate manifest.toml
├── template_validator.py     # Extract TODOs/blanks
├── framework.py              # Optional test helpers
└── exercises/
    └── __init__.py
```

## Testing Requirements

Create tests for your utilities:

```python
# katas/test_framework.py

import pytest
from pathlib import Path
from manifest_parser import KataManifest, validate_kata_directory
from template_validator import extract_blanks, count_todos


def test_manifest_parser():
    # Create temp manifest and test parsing
    pass


def test_blank_extraction():
    sample = """
def foo():
    # BLANK_START
    return 42
    # BLANK_END
    """
    # Test extraction logic
    pass
```

Run tests with: `uv run pytest`

## Output Format Examples

### Successful Run
```json
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
      "test_name": "test_attention_weights",
      "status": "passed",
      "output": ""
    }
  ]
}
```

### Failed Run
```json
{
  "passed": false,
  "num_passed": 3,
  "num_failed": 2,
  "num_skipped": 0,
  "duration_ms": 189,
  "results": [
    {
      "test_name": "test_output_shape",
      "status": "passed",
      "output": ""
    },
    {
      "test_name": "test_causal_mask",
      "status": "failed",
      "output": "AssertionError: Causal mask not applied correctly\n  Expected upper triangle to be masked\n  Got: some attention to future positions"
    }
  ]
}
```

## Acceptance Criteria

- [ ] `runner.py` executes pytest and outputs valid JSON
- [ ] Manifest parser validates all required fields
- [ ] Template validator extracts TODO/BLANK markers
- [ ] Framework helpers (assert_shape, assert_close) work correctly
- [ ] All utilities have unit tests
- [ ] Code passes `ruff check` and `ruff format`
- [ ] Documentation strings for all public functions

## Handoff to Other Agents

Once complete, your work provides:
- **To Agent 1 (Rust):** JSON protocol for test results
- **To Agent 4 (Example katas):** Framework and utilities for writing katas
- **To Agent 3 (TUI):** Command to run: `python -m runner <kata_id> <template_path>`

Document the JSON schema clearly so Agent 1 can implement the Rust parser.

## Notes

- Keep the framework minimal - don't build a complex abstraction layer
- Pytest plugin should not print anything except final JSON to stdout
- Use type hints everywhere (this is scaffold-quality code)
- No emojis, no fancy formatting - clean, simple Python
- Test runner should handle import errors gracefully (user code might be broken)
- Consider edge cases: empty tests, syntax errors in user code, missing imports
