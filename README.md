# Kata Spaced Repetition

> **Note:** Personal software, not intended for general use. 

A terminal-based spaced repetition system for mastering coding patterns and algorithms. Practice PyTorch implementations, algorithm patterns, and ML techniques with scientifically-optimized review scheduling using the FSRS-5 algorithm.

## Features

### FSRS-5 Spaced Repetition Algorithm

- Modern memory model that tracks **stability**, **difficulty**, and **retrievability**
- Optimized from millions of reviews for maximum retention efficiency
- 4-point rating scale: Again / Hard / Good / Easy
- Personalized parameter optimization based on your review history

### Intelligent Kata Management

- **Multi-tag organization**: Categorize katas with multiple tags (e.g., "transformers", "attention", "advanced")
- **Library browser**: Search, filter, and sort available katas
- **Dependency graphs**: Unlock advanced katas by mastering prerequisites
- **Adaptive difficulty**: System tracks your performance and recommends appropriate challenges
- **Kata variations**: Practice different versions with varying constraints

### Progress Analytics

- **Streak tracking**: Monitor consecutive days of practice
- **GitHub-style heatmap**: Visual calendar of your activity
- **Success rate trends**: Track improvement over time
- **Category breakdown**: See which areas you're focusing on
- All visualizations are text-based for terminal environments

### Interactive TUI

- **Dashboard**: See katas due today with priority sorting
- **Library tabs**: Browse "My Deck" vs "All Katas"
- **Live test feedback**: Run pytest tests with real-time results
- **Editor integration**: Edit code in your preferred editor (vim/nvim/neovim)
- **Keyboard-driven**: Fast navigation with intuitive keybindings

## Quick Start

### Prerequisites

1. **Rust** (1.70+)
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   ```

2. **uv** (Python package manager)
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. **Editor**: vim, nvim, or neovim (configurable)

### Installation

```bash
# Clone repository
git clone https://github.com/afiq-aswadi/kata-sr.git
cd kata-sr

# Build and install
cargo install --path .

# Run for the first time (initializes database and Python environment)
kata-sr
```

On first run, the tool will:
1. Create virtual environment at `katas/.venv/`
2. Install Python dependencies (pytest, torch, einops)
3. Initialize database at `~/.local/share/kata-sr/kata.db`
4. Import available katas from `katas/exercises/`

## Usage

### Starting the TUI

```bash
kata-sr
```

### Navigation

**Global Keybindings:**
- `?` - Help screen with all keybindings
- `q` - Quit application
- `Esc` - Return to previous screen
- `l` - Open Library
- `Tab` - Switch between Library tabs (My Deck / All Katas)

**Dashboard:**
- `↑/↓` or `j/k` - Navigate katas
- `Enter` - Practice selected kata
- `d` - Delete kata from deck
- `r` - Refresh view

**Library:**
- `↑/↓` or `j/k` - Navigate available katas
- `Enter` or `a` - Add kata to practice deck
- `n` - Create new kata (form-based)
- `/` - Search by name or tags
- `f` - Filter by tags
- `s` - Sort (name, difficulty, date added)

**Practice Flow:**
1. Select kata from dashboard
2. Read description and requirements
3. Press `Enter` to open editor
4. Implement the solution
5. Save and exit editor
6. Tests run automatically
7. Review results
8. Rate difficulty (1-4)
9. Return to dashboard

### Creating Custom Katas

#### Method 1: TUI Form (Recommended)

1. Run `kata-sr` and press `l` for Library
2. Press `n` to create new kata
3. Fill in the form:
   - **Name**: Display name (e.g., "Multi-Head Attention")
   - **Category**: Primary category (e.g., "transformers")
   - **Tags**: Additional tags, comma-separated (e.g., "attention, advanced, pytorch")
   - **Description**: What to implement (multiline supported)
   - **Difficulty**: 1-5 stars
   - **Dependencies**: Prerequisites (optional)
4. Confirm directory name → files generated automatically
5. Find kata in Library → press `a` to add to deck

Generated files in `katas/exercises/<kata_name>/`:
- `manifest.toml` - Metadata and configuration
- `template.py` - Starter code with TODOs
- `test_kata.py` - Test suite (edit these!)
- `reference.py` - Your reference solution

#### Method 2: Manual Creation

Create directory structure:

```bash
mkdir -p katas/exercises/my_kata
cd katas/exercises/my_kata
```

Create `manifest.toml`:

```toml
[kata]
name = "my_kata"
category = "algorithms"
tags = ["sorting", "intermediate"]
base_difficulty = 3
description = """
Implement quicksort with in-place partitioning.
"""
dependencies = ["bubble_sort"]  # Optional
```

Create `template.py` with function signatures:

```python
def quicksort(arr: list[int]) -> list[int]:
    """Sort array using quicksort algorithm."""
    raise NotImplementedError("TODO: Implement quicksort")
```

Create `test_kata.py` with pytest tests:

```python
from template import quicksort

def test_empty_array():
    assert quicksort([]) == []

def test_sorted_array():
    assert quicksort([1, 2, 3]) == [1, 2, 3]

def test_reverse_sorted():
    assert quicksort([3, 2, 1]) == [1, 2, 3]
```

Then in TUI, the kata will automatically appear in Library.

### Best Practices

- **Descriptions**: Explain *what* to implement, not *how*
- **Tests**: Verify correctness, not implementation details
- **Difficulty**: Start conservative; FSRS will adapt
- **Tags**: Use consistent naming (lowercase, underscores)
- **Dependencies**: Use sparingly to avoid blocking progress

## Architecture

### Tech Stack

**Rust** (Backend + TUI):
- `ratatui` - Terminal UI framework
- `rusqlite` - SQLite database
- `clap` - CLI argument parsing
- `serde_json` - JSON serialization
- `chrono` - Date/time handling

**Python** (Kata Execution):
- `pytest` - Test runner
- `torch` - PyTorch for ML katas
- `einops` - Tensor operations
- `uv` - Fast package management

### Workflow

```
1. User runs: kata-sr
2. Dashboard shows katas due today (sorted by priority)
3. User selects kata → reads description
4. Rust writes template to /tmp/kata_<session_id>.py
5. Rust spawns editor (vim/nvim/neovim)
6. User implements solution, saves, exits
7. Rust spawns pytest in background thread
8. TUI shows test results (pass/fail with details)
9. User rates difficulty (1-4)
10. FSRS-5 algorithm schedules next review
11. Back to dashboard with updated stats
```

### File Structure

```
kata-sr/
├── src/
│   ├── main.rs                 # CLI entry point
│   ├── lib.rs                  # Public API
│   ├── python_env.rs           # Python environment bootstrap
│   ├── core/
│   │   ├── fsrs.rs            # FSRS-5 algorithm
│   │   ├── fsrs_optimizer.rs  # Parameter optimization
│   │   ├── difficulty.rs      # Adaptive difficulty
│   │   ├── dependencies.rs    # Dependency graph
│   │   └── analytics.rs       # Statistics computation
│   ├── db/
│   │   ├── schema.rs          # SQLite migrations
│   │   └── repo.rs            # Repository layer
│   └── tui/
│       ├── app.rs             # Main event loop
│       ├── dashboard.rs       # Dashboard view
│       ├── library.rs         # Library browser
│       ├── practice.rs        # Practice session
│       ├── results.rs         # Test results view
│       └── heatmap.rs         # Activity visualization
├── katas/
│   ├── pyproject.toml         # Python dependencies
│   ├── runner.py              # Pytest wrapper
│   └── exercises/
│       ├── multihead_attention/
│       ├── dfs_bfs/
│       └── .../
└── CLAUDE.md                   # Project overview
```

## FSRS-5 Algorithm

### What is FSRS?

FSRS-5 (Free Spaced Repetition Scheduler) uses a memory model that accurately predicts when you'll forget information based on research from millions of reviews.

**Key Concepts:**
- **Stability**: Half-life of memory (how long until 50% retrieval probability)
- **Difficulty**: Inherent complexity of the material (1-10 scale)
- **Retrievability**: Current probability of successful recall

### Rating System

- **Again (1)**: Complete failure → reset to 1 day, mark as lapsed
- **Hard (2)**: Struggled but passed → minimal interval growth
- **Good (3)**: Normal difficulty → standard progression
- **Easy (4)**: Too easy → accelerated interval growth

### Parameter Optimization

FSRS-5 uses 19 parameters that can be personalized to your review history:

```bash
# Optimize parameters based on your review history (50+ reviews recommended)
kata-sr optimize-fsrs

# View current parameters
kata-sr fsrs-stats
```

The system starts with default parameters (optimized from millions of reviews) and can be refined as you accumulate more practice data.

## Database

### Location

```
~/.local/share/kata-sr/kata.db
```

### Schema

**katas** - Kata metadata and FSRS state
- Basic info: `name`, `category`, `tags`, `description`
- FSRS state: `fsrs_stability`, `fsrs_difficulty`, `fsrs_state`
- Legacy SM-2: `current_ease_factor`, `current_interval_days`
- Scheduling: `next_review_at`, `last_reviewed_at`

**sessions** - Practice history
- Test results: `num_passed`, `num_failed`, `duration_ms`
- Rating: `quality_rating` (1-4)
- Full output: `test_results_json`

**daily_stats** - Aggregated analytics
- Metrics: `total_reviews`, `success_rate`, `streak_days`
- Category breakdown: `categories_json`

**fsrs_params** - Optimized parameters
- 19 weights stored as JSON
- Timestamp for versioning

## CLI Reference

### Main Commands

```bash
# Start TUI
kata-sr

# Use custom database
kata-sr --db-path /path/to/kata.db

# Show help
kata-sr --help
```

### Debug Commands

Useful for development and testing:

```bash
# Reset all katas to initial state (makes everything due immediately)
kata-sr debug reset-all

# Reset specific kata
kata-sr debug reset <kata_name>

# Make kata due now (preserves FSRS state)
kata-sr debug force-due <kata_name>

# Reimport katas from manifests (updates metadata, preserves review state)
kata-sr debug reimport

# Reimport and delete katas not in exercises/ directory
kata-sr debug reimport --prune

# Show database statistics
kata-sr debug stats
kata-sr debug stats --json

# List all katas
kata-sr debug list

# List only due katas
kata-sr debug list --due

# Clear session history
kata-sr debug clear-sessions

# Delete specific kata
kata-sr debug delete <kata_name>
```

### Common Workflows

**Quick testing setup:**
```bash
kata-sr debug reset-all
kata-sr
```

**Iterate on kata content:**
```bash
# Edit template.py or test_kata.py
kata-sr debug reimport  # Reload changes
```

**Fresh start:**
```bash
kata-sr debug clear-sessions
kata-sr debug clear-stats
kata-sr debug reset-all
```

**Isolated testing:**
```bash
kata-sr --db-path /tmp/test.db debug stats
```

## Development

### Building

```bash
# Debug build
cargo build

# Release build
cargo build --release

# Install locally
cargo install --path .
```

### Testing

```bash
# Run all tests
cargo test

# Run with output
cargo test -- --nocapture

# Run specific test file
cargo test --test integration_full

# Run specific module
cargo test core::fsrs::tests

# Run FSRS-specific tests
cargo test fsrs
```

### Code Quality

```bash
# Format code
cargo fmt

# Lint (zero warnings enforced)
cargo clippy -- -D warnings

# Generate documentation
cargo doc --open
```

## Example Katas

The repository includes several example katas to get started:

- **multihead_attention**: Implement scaled dot-product multi-head attention
- **dfs_bfs**: Depth-first and breadth-first search algorithms
- **mlp**: Multi-layer perceptron with PyTorch
- **einsum_basics**: Practice Einstein summation notation

Each kata includes:
- Clear description and requirements
- Template with function signatures
- Comprehensive test suite
- Reference solution

## Troubleshooting

### Python environment issues

```bash
# Manually recreate Python environment
cd katas
uv sync
```

### Database corruption

```bash
# Reset database (WARNING: loses all progress)
rm ~/.local/share/kata-sr/kata.db
kata-sr  # Reinitializes
```

### Tests not running

```bash
# Verify Python environment
source katas/.venv/bin/activate
python -m pytest katas/exercises/<kata_name>/

# Check test file syntax
python -m py_compile katas/exercises/<kata_name>/test_kata.py
```

### Kata not appearing in Library

```bash
# Reimport manifests
kata-sr debug reimport

# Check manifest.toml is valid
cat katas/exercises/<kata_name>/manifest.toml
```

## References

- [FSRS-5 Algorithm](https://github.com/open-spaced-repetition/fsrs4anki/wiki/The-Algorithm)
- [FSRS Research Paper](https://www.nature.com/articles/s41562-024-01962-9)
- [Spaced Repetition Wiki](https://www.gwern.net/Spaced-repetition)

## License

MIT

## Acknowledgments

- FSRS algorithm by [Jarrett Ye](https://github.com/open-spaced-repetition/fsrs-rs)
- Ratatui framework for terminal UI

