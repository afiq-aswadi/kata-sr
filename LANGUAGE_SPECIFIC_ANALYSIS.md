# Language-Specific Functionality Analysis - Kata-SR

## Executive Summary

The codebase currently has **strong Python-specific coupling** throughout the system. Most language-specific logic is hardcoded for Python with `.py` file extensions, pytest as the test framework, and a Python environment manager (uv/venv). To support multiple languages, these coupling points need to be abstracted into a language-agnostic system.

---

## 1. Files Containing Language-Specific Logic

### Rust Files

#### 1.1 Python Environment Setup
- **File**: `/home/user/kata-sr/src/python_env.rs` (136 lines)
- **Language assumptions**: 
  - Hardcoded `uv` package manager check
  - Hardcoded `katas/.venv` virtual environment path
  - Hardcoded `katas/.venv/bin/python` interpreter path
  - Environment variables: `KATA_SR_PYTHON`, `VIRTUAL_ENV`, `KATA_SR_KATAS_DIR`
- **Used by**: `main.rs` (startup initialization)

#### 1.2 Python Test Runner
- **File**: `/home/user/kata-sr/src/runner/python_runner.rs` (119 lines)
- **Language assumptions**:
  - Calls Python with `-m runner <kata_id> <template_path>` 
  - Expects JSON output from pytest runner
  - Hardcoded interpreter path via `KATA_SR_PYTHON` env var
  - Expects katas directory via `KATA_SR_KATAS_DIR`
- **Key function**: `run_python_tests(kata_id: &str, template_path: &Path) -> TestResults`
- **Interface**: JSON JSON serialization with:
  ```rust
  pub struct TestResults {
      pub passed: bool,
      pub num_passed: i32,
      pub num_failed: i32,
      pub num_skipped: i32,
      pub duration_ms: i64,
      pub results: Vec<TestResult>,
  }
  ```

#### 1.3 Kata Template Generation
- **File**: `/home/user/kata-sr/src/core/kata_generator.rs` (250+ lines)
- **Language assumptions**:
  - Generates `.py` files only: `template.py`, `test_kata.py`, `reference.py`
  - Python-specific function signature: `def kata_{slug}():`
  - Hardcoded `import pytest` in test files
  - Hardcoded `from template import kata_{slug}` in tests
  - Manifest generation is language-agnostic, but file generation is not

#### 1.4 Practice Screen
- **File**: `/home/user/kata-sr/src/tui/practice.rs` (267 lines)
- **Language assumptions**:
  - Hardcoded `/tmp/kata_{id}.py` template path
  - Assumes all templates are in `katas/exercises/{kata_name}/template.py`
  - Calls `run_python_tests()` directly
  - No support for alternative file extensions

#### 1.5 Main Entry Point
- **File**: `/home/user/kata-sr/src/main.rs` (159 lines)
- **Language assumptions**:
  - Requires Python environment setup before TUI
  - Configures Python-specific env vars (KATA_SR_PYTHON, VIRTUAL_ENV, PATH)
  - Environment setup is mandatory for TUI (optional for debug commands)

#### 1.6 Configuration
- **File**: `/home/user/kata-sr/src/config/mod.rs` (100+ lines)
- **Current state**: Editor configuration is language-agnostic
- **Missing**: No language selection in config; assumes Python globally

#### 1.7 CLI Debug Operations
- **File**: `/home/user/kata-sr/src/cli/debug.rs`
- **Assumption**: Debug commands may reference Python paths/runners

### Python Files

#### 2.1 Test Runner Module
- **File**: `/home/user/kata-sr/katas/runner.py` (308 lines)
- **Key responsibility**: 
  - Takes kata_id and template_path as arguments
  - Imports user template as `user_kata` module
  - Runs pytest on test files in kata directory
  - Returns JSON with results
- **Language assumptions**:
  - Hardcoded to Python 3.13+
  - Uses pytest for test execution
  - Assumes tests are in `test_*.py` files (pytest convention)
  - Module import assumes `.py` extension
- **Interface contract**: CLI: `python -m runner <kata_id> <template_path>`

#### 2.2 Manifest Parser
- **File**: `/home/user/kata-sr/katas/manifest_parser.py` (149 lines)
- **Language assumptions**:
  - Validates presence of `template.py` and `test_kata.py`
  - No language field in validation
- **Currently unused** by Rust side (reads manifests directly)

#### 2.3 Kata Framework
- **File**: `/home/user/kata-sr/katas/framework.py`
- **Purpose**: Test utilities and base classes for katas
- **Language assumptions**: Python-specific test helpers

### Configuration Files

#### 3.1 Python Project Configuration
- **File**: `/home/user/kata-sr/katas/pyproject.toml`
- **Language assumptions**:
  - Hardcoded `requires-python = ">=3.13"`
  - Dependencies: torch, numpy, einops, pytest, etc. (ML/Python ecosystem)
  - Pytest configuration: `testpaths = ["tests", "exercises"]`
  - No mechanism for multiple language environments

#### 3.2 Kata Manifest TOML
- **Example**: `/home/user/kata-sr/katas/exercises/string_reverse/manifest.toml`
- **Current fields**:
  ```toml
  [kata]
  name = "string_reverse"
  category = "strings"
  base_difficulty = 1
  description = "..."
  dependencies = []
  ```
- **Missing**: No `language` field
- **Language-agnostic**: Yes, but assumes Python in file generation

---

## 2. Current Contracts & Interfaces Between Components

### 2.1 Python Environment Contract

```
PythonEnv::setup()
  ├─> Checks: `which uv`
  ├─> Ensures: `katas/.venv/` exists (via `uv sync`)
  ├─> Returns: Path to `katas/.venv/bin/python`
  └─> Sets env vars:
      - KATA_SR_PYTHON = interpreter path
      - VIRTUAL_ENV = venv directory
      - KATA_SR_KATAS_DIR = katas/ directory
      - PATH += venv/bin or venv/Scripts
```

### 2.2 Test Runner Contract

```
run_python_tests(kata_id: &str, template_path: &Path) -> TestResults
  │
  ├─> Spawns: {KATA_SR_PYTHON} -m runner {kata_id} {template_path}
  ├─> Working dir: {KATA_SR_KATAS_DIR}
  │
  └─> Expects JSON on stdout:
      {
        "passed": bool,
        "num_passed": i32,
        "num_failed": i32,
        "num_skipped": i32,
        "duration_ms": i64,
        "results": [
          {
            "test_name": string,
            "status": string,
            "output": string,
            "duration_ms": i64
          }
        ]
      }
```

### 2.3 File System Contract

```
katas/
├── exercises/
│   ├── <kata_slug>/
│   │   ├── manifest.toml        (kata metadata - language-agnostic)
│   │   ├── template.py          (user edits this - language-specific)
│   │   ├── test_kata.py         (tests - language-specific)
│   │   └── reference.py         (solution - language-specific)
│   └── ...
├── runner.py                    (test runner entry point)
├── manifest_parser.py           (manifest parsing - Python-only)
├── pyproject.toml               (Python dependencies only)
└── .venv/                       (Python venv only)
```

### 2.4 Template Generation Contract

```
generate_kata_files(form_data, exercises_dir) -> Result<String>
  │
  ├─> Creates: {exercises_dir}/{slug}/
  ├─> Generates:
  │   ├── manifest.toml          (language-agnostic metadata)
  │   ├── template.py            (HARDCODED .py extension)
  │   ├── test_kata.py           (HARDCODED .py extension)
  │   └── reference.py           (HARDCODED .py extension)
  │
  └─> Returns: slug (kata directory name)
```

### 2.5 Database Contract

```sql
katas table (CURRENT STATE):
├── id, name, category, description, base_difficulty, ...
├── MISSING: language FIELD
└── Assumes all katas are Python
   (no language metadata to drive behavior)

sessions table (CURRENT STATE):
├── id, kata_id, started_at, completed_at
├── test_results_json          (stores TestResults JSON)
├── code_attempt               (stores user's solution)
└── Assumes all test results follow Python JSON schema
```

---

## 3. Key Extension Points for Multi-Language Support

### 3.1 Language Registry System (NEW)

**Purpose**: Abstract away language-specific behavior

```
Language trait/config:
├── name: String (e.g., "python", "javascript", "rust")
├── file_extensions: [String] (e.g., [".py", ".pyc"])
├── test_framework: String (e.g., "pytest", "jest", "cargo test")
├── environment_setup: fn() -> Result<EnvConfig>
├── template_generator: fn(kata_data) -> Result<TemplateFiles>
├── test_runner: fn(kata_id, template_path) -> Result<TestResults>
├── dependencies_manager: String (e.g., "pip", "npm", "cargo")
└── default_template_content: fn(slug) -> String
```

### 3.2 Database Schema Extensions

```sql
ALTER TABLE katas ADD COLUMN language TEXT DEFAULT 'python';
ALTER TABLE katas ADD COLUMN language_version TEXT;

-- Example: 
INSERT INTO katas (name, language, language_version, ...)
VALUES ('my_kata', 'python', '3.13', ...);
```

### 3.3 Manifest Enhancement

```toml
[kata]
name = "my_kata"
category = "strings"
base_difficulty = 1
description = "..."
language = "python"          # NEW FIELD
language_version = ">=3.13"  # NEW FIELD
dependencies = []
```

### 3.4 Directory Structure Refactoring

```
katas/
├── manifests/               (language-agnostic metadata only)
│   ├── <kata_slug>.toml
│   └── ...
├── languages/               (language-specific runners & setup)
│   ├── python/
│   │   ├── runner.py
│   │   ├── environment.py
│   │   └── pyproject.toml
│   ├── javascript/
│   │   ├── runner.js
│   │   ├── environment.js
│   │   └── package.json
│   └── ...
├── exercises/
│   ├── <kata_slug>/
│   │   ├── template.py      (or .js, .rs, etc.)
│   │   ├── test_kata.py     (language-specific)
│   │   └── reference.py     (language-specific)
│   └── ...
└── .venv/                   (could support multiple envs)
```

---

## 4. Hardcoded Python Assumptions

### Critical Coupling Points

| Location | Code | Impact |
|----------|------|--------|
| `python_env.rs` | `katas/.venv/bin/python` | Environment setup |
| `python_runner.rs` | Hardcoded Python interpreter path | Test execution |
| `kata_generator.rs` | `template.py`, `test_kata.py`, `reference.py` | Kata creation |
| `practice.rs` | `/tmp/kata_{id}.py` | File paths |
| `practice.rs` | Direct call to `run_python_tests()` | No abstraction layer |
| `main.rs` | Mandatory Python env setup | Startup flow |
| `pyproject.toml` | Python 3.13+, pytest, torch, etc. | Dependencies |

### Less Critical (More Flexible)

| Location | Code | Impact |
|----------|------|--------|
| `runner.rs` | `pub mod python_runner` | Can add other runners |
| `config/mod.rs` | Editor config is generic | Already language-agnostic |
| `manifest_parser.py` | Validates `.py` files exist | Can be extended |
| `kata_loader.rs` | Loads manifests | Language-agnostic already |

---

## 5. Database Schema Considerations

### Current Limitations

1. **No language column**: Can't determine kata language from database
2. **No language_version column**: Can't track version requirements
3. **Test results JSON schema is Python-specific**: Could be extended
4. **Session code_attempt is untyped**: Stores raw text without language context

### Proposed Schema Changes

```sql
-- Add language metadata
ALTER TABLE katas ADD COLUMN language TEXT DEFAULT 'python' NOT NULL;
ALTER TABLE katas ADD COLUMN language_version TEXT;  -- e.g., "3.13", "16.0.0"
ALTER TABLE katas ADD COLUMN template_extension TEXT;  -- e.g., ".py", ".js"

-- Extend sessions for language context
ALTER TABLE sessions ADD COLUMN language TEXT;
ALTER TABLE sessions ADD COLUMN language_version TEXT;

-- New table for language configuration
CREATE TABLE IF NOT EXISTS language_configs (
    id INTEGER PRIMARY KEY,
    language TEXT NOT NULL UNIQUE,
    runner_script TEXT,           -- e.g., "runner.py", "runner.js"
    dependencies_file TEXT,       -- e.g., "pyproject.toml", "package.json"
    test_framework TEXT,          -- e.g., "pytest", "jest"
    default_version TEXT,         -- e.g., "3.13"
    environment_type TEXT         -- e.g., "venv", "node_modules"
);
```

---

## 6. TUI Component Considerations

### Affected Components

#### 6.1 Practice Screen (`practice.rs`)
- **Issue**: Hardcoded `.py` file generation
- **Needed**: Language parameter to determine file extension
- **Impact**: File path construction, editor launch

#### 6.2 Create Kata Screen (`create_kata.rs`)
- **Current fields**: Name, Category, Description, Difficulty, Dependencies
- **Needed**: Language selection dropdown
- **Impact**: Form validation, template generation

#### 6.3 Results Screen (`results.rs`)
- **Current**: Displays pytest-format test results
- **Needed**: Language-aware result formatting
- **Impact**: Test result parsing and display

#### 6.4 Details Screen (`details.rs`)
- **Current**: Shows kata info
- **Needed**: Display language and language_version
- **Impact**: UI layout minimal

#### 6.5 Editor Config (`config/mod.rs`)
- **Current**: Editor command and args
- **Enhancement**: Per-language editor overrides? (Optional)
- **Impact**: Startup config only

---

## 7. Test Framework Abstraction

### Current: Python/pytest specific

```rust
// In python_runner.rs
pub struct TestResults {
    pub passed: bool,
    pub num_passed: i32,
    pub num_failed: i32,
    pub num_skipped: i32,
    pub duration_ms: i64,
    pub results: Vec<TestResult>,
}
```

### Proposed: Language-agnostic

```rust
pub struct TestResults {
    pub passed: bool,
    pub total_tests: i32,
    pub passed_tests: i32,
    pub failed_tests: i32,
    pub skipped_tests: i32,
    pub duration_ms: i64,
    pub language: String,           // NEW
    pub results: Vec<TestResult>,
}

pub struct TestResult {
    pub test_name: String,
    pub status: TestStatus,         // ENUM instead of String
    pub output: String,
    pub duration_ms: i64,
    pub language_specific: Option<LanguageTestData>,  // For language-specific fields
}

pub enum TestStatus {
    Passed,
    Failed,
    Skipped,
    Errored,
}
```

---

## 8. Summary of Extension Points

| Extension Point | Current State | Needed for Multi-Language |
|-----------------|---------------|--------------------------|
| Environment setup | Python-only (uv/venv) | Language-specific manager |
| Test runner | Python-specific runner.py | Language-specific runners |
| Template generation | Hardcoded .py files | Language-aware factory |
| File paths | /tmp/kata_{id}.py | Dynamic extensions |
| Database schema | No language column | Language + version fields |
| Test result format | Python/pytest JSON | Language-agnostic schema |
| Configuration | Editor config only | Language selection + setup |
| TUI forms | No language field | Language selector dropdown |
| Manifest format | Language-agnostic TOML | Language field |

---

## 9. Implementation Roadmap

### Phase 1: Foundation (Database & Config)
1. Add `language` column to `katas` table (default: "python")
2. Add `language_version` column to `katas` table
3. Add language selection to manifest TOML format
4. Create `language_configs` table for driver metadata

### Phase 2: Abstraction Layer
1. Create `LanguageDriver` trait/enum to abstract language behavior
2. Implement `PythonDriver` wrapping current python_env.rs code
3. Move test runner dispatch to language driver
4. Move template generation to language-aware factory

### Phase 3: TUI Integration
1. Add language selector to create_kata form
2. Update practice screen to use language from database
3. Update template generation to use language-specific factory
4. Update results parsing to be language-aware

### Phase 4: Multi-Language Support (Actual Languages)
1. Implement `JavaScriptDriver` (Node.js + Jest)
2. Implement `RustDriver` (Cargo + tests)
3. Test interoperability and database migrations

---

## 10. Files Summary Table

| File | Purpose | Language-Specific? | Coupling Level |
|------|---------|-------------------|-----------------|
| `python_env.rs` | Python environment setup | YES | CRITICAL |
| `python_runner.rs` | Test execution | YES | CRITICAL |
| `kata_generator.rs` | Template file generation | YES (partially) | HIGH |
| `practice.rs` | User editing interface | YES (file paths) | MEDIUM |
| `main.rs` | Entry point | YES (env setup) | MEDIUM |
| `config/mod.rs` | Application config | NO | LOW |
| `runner.py` | Python test runner | YES | CRITICAL |
| `manifest_parser.py` | Manifest validation | YES (validates .py) | MEDIUM |
| `kata_loader.rs` | Load manifests from disk | NO | LOW |
| `db/schema.rs` | Database migrations | NO | LOW |
| `db/repo.rs` | Database queries | NO | LOW |

