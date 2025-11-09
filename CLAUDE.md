# Kata Spaced Repetition - Project Overview

## Vision

A personal TUI tool for practicing coding patterns (multi-head attention, DFS/BFS, MLP, etc.) using spaced repetition. Think LeetCode meets Anki - you implement algorithms from templates, tests verify correctness, and an SM-2 scheduler ensures you retain knowledge long-term.

## Architecture

**Hybrid Rust + Python system:**

- **Rust** handles CLI, TUI (ratatui), scheduling, database, and coordination
- **Python** handles kata execution (pytest), since exercises involve PyTorch/ML code
- Communication via JSON over stdio

**Tech Stack:**

- Rust: ratatui, rusqlite, clap, serde_json, chrono
- Python: pytest, torch, einops (managed by uv)

## Core Workflow

```
1. User runs: kata-sr
2. TUI dashboard shows katas due today + stats
3. User selects kata
4. Rust writes template to /tmp/kata_<id>.py
5. Rust spawns nvim with template
6. User fills in TODOs/blanks, saves, exits
7. Rust spawns pytest in background thread
8. TUI shows test results (pass/fail)
9. User rates difficulty (0-3: Again/Hard/Good/Easy)
10. SM-2 algorithm schedules next review
11. Back to dashboard
```

## Key Features

### 1. SM-2 Spaced Repetition

- Standard Anki algorithm with 4-point rating scale (Again/Hard/Good/Easy)
- State stored per-kata: next_review_at, ease_factor, interval_days, repetition_count
- Dashboard queries `WHERE next_review_at <= now()` for fast "due today" view

### 2. Adaptive Difficulty Tracking

- Independent from SM-2 (doesn't affect scheduling)
- Tracks success rate over recent reviews
- current_difficulty increases if too easy (>90% success), decreases if struggling (<50%)
- Used for UI recommendations: "Try katas at your level"

### 3. Kata Dependencies (Prerequisites)

- Directed graph: some katas require others to be unlocked first
- Must pass prerequisite N times before dependent kata becomes available
- Dashboard shows locked katas with 🔒 + explanation
- Enables structured learning paths

### 4. Kata Variations

- Base kata can have variations with different parameters/constraints
- Example: "attention" → "attention_causal", "attention_cross"
- Each variation scheduled independently
- Linked in UI for easy discovery

### 5. Progress Analytics

- Current streak (consecutive days with reviews)
- Weekly heatmap (ASCII visualization)
- Success rate trends
- Category breakdown
- All text-based, no plots

## File Structure

kata-sr/
├── Cargo.toml                    # Rust project
├── src/
│   ├── main.rs                   # CLI entry, Python env bootstrap
│   ├── tui/
│   │   ├── app.rs               # Main event loop
│   │   ├── dashboard.rs         # Due katas, stats, analytics
│   │   ├── practice.rs          # Kata description, editor spawn
│   │   └── results.rs           # Test output, rating input
│   ├── core/
│   │   ├── scheduler.rs         # SM-2 implementation
│   │   ├── difficulty.rs        # Adaptive difficulty tracker
│   │   └── kata_loader.rs       # Import manifests into DB
│   ├── db/
│   │   ├── schema.rs            # SQLite tables + migrations
│   │   └── repo.rs              # Database queries
│   └── runner/
│       └── python_runner.rs     # Spawn pytest, parse JSON
├── katas/
│   ├── pyproject.toml           # Python dependencies (uv)
│   ├── runner.py                # Pytest wrapper (JSON output)
│   └── exercises/
│       ├── multihead_attention/
│       │   ├── manifest.toml    # Metadata (name, deps, difficulty)
│       │   ├── template.py      # Starter code with TODOs
│       │   ├── reference.py     # Solution
│       │   └── test_kata.py     # Pytest tests
│       └── .../
├── CLAUDE.md                     # This file
├── AGENT_1.md                    # Rust core & database
├── AGENT_2.md                    # Python framework
├── AGENT_3.md                    # TUI application
├── AGENT_4.md                    # Example katas
└── AGENT_5.md                    # Analytics & integration

## Database Schema

**katas:** kata metadata + current SM-2 state

- id, name, category, description
- base_difficulty, current_difficulty
- parent_kata_id, variation_params (for variations)
- next_review_at (indexed!), last_reviewed_at
- current_ease_factor, current_interval_days, current_repetition_count

**kata_dependencies:** prerequisite graph

- kata_id, depends_on_kata_id, required_success_count

**sessions:** full practice history

- id, kata_id, started_at, completed_at
- test_results_json (pytest output)
- num_passed, num_failed, num_skipped, duration_ms
- quality_rating (0-3)

**daily_stats:** aggregated analytics

- date, total_reviews, success_rate, streak_days, categories_json

## Agent Coordination

### Phase 1: Foundation (Parallel)

- **Agent 1:** Rust core & database (schema, SM-2, difficulty, deps)
- **Agent 2:** Python kata framework (runner, manifest parser)

**Agents 1 and 2 can work completely in parallel.**

### Phase 2: Applications (Parallel, depends on Phase 1)

- **Agent 3:** Rust TUI application (depends on Agent 1 for schema/repo)
- **Agent 4:** Example katas (depends on Agent 2 for framework)

**Agents 3 and 4 can work in parallel after Phase 1 completes.**

### Phase 3: Integration (depends on all previous)

- **Agent 5:** Analytics & integration (final testing, polish)

## Key Design Decisions

### Rating Scale: 0-3 (unified across stack)

```rust
enum QualityRating {
    Again = 0,  // Reset to day 1, ease -= 0.2
    Hard = 1,   // interval *= 1.2, ease -= 0.15
    Good = 2,   // interval *= ease_factor
    Easy = 3,   // interval *= ease_factor * 1.3, ease += 0.15
}
```

### Python Environment Bootstrap

On startup, `main.rs`:

1. Check `which uv` → error if missing
2. Check `katas/.venv/` exists
3. If not: run `uv sync --directory katas/` with spinner
4. Resolve interpreter path: `katas/.venv/bin/python`
5. Cache in AppState for all runner calls

### TUI Responsiveness

- Test execution runs on background thread
- Main event loop handles `AppEvent::Input` and `AppEvent::TestComplete`
- UI shows spinner while tests run, disable input temporarily
- No blocking on main thread

### Kata Manifest Format (TOML)

```toml
[kata]
name = "multihead_attention"
category = "transformers"
base_difficulty = 4
description = "Implement scaled dot-product multi-head attention"
dependencies = ["mlp"]

[[variations]]
name = "attention_causal"
description = "Add causal masking"
params = { mask_type = "causal" }
```

Metadata lives in SQLite (single source of truth). Manifests are imported via `kata-sr add`.

### Python Runner Protocol

```bash
# Rust calls:
katas/.venv/bin/python -m runner <kata_id> <template_path>

# Python returns JSON to stdout:
{
  "passed": true,
  "num_passed": 5,
  "num_failed": 0,
  "num_skipped": 0,
  "duration_ms": 234,
  "results": [
    {"test_name": "test_output_shape", "status": "passed", "output": ""},
    {"test_name": "test_attention_normalized", "status": "failed", "output": "..."}
  ]
}
```

## Acceptance Criteria

The project is complete when:

1. `cargo install --path .` creates working `kata-sr` binary
2. Running `kata-sr` launches TUI with dashboard
3. User can select kata, edit in nvim, run tests, rate difficulty
4. SM-2 scheduling works correctly (katas appear when due)
5. Dependencies work (locked katas can't be practiced)
6. Analytics dashboard shows streak, success rate, heatmap
7. At least 3 example katas exist with full tests
8. README documents installation and usage

## Getting Started for Agents

1. Read this file (CLAUDE.md) for overall context
2. Read your specific AGENT_X.md file for detailed instructions
3. Check dependencies: if your agent depends on another, coordinate or wait
4. Implement according to spec, write tests as you go
5. When done, notify in main thread and help with integration

## Communication Between Agents

- Use database schema from Agent 1 as contract
- Python runner protocol (JSON) is contract between Agent 1 and Agent 2
- Repo interface in Agent 1 is contract for Agent 3
- Don't make breaking changes without coordinating

## Notes

- User edits katas in external nvim, not embedded editor
- Database lives at `~/.local/share/kata-sr/kata.db`
- Templates written to `/tmp/kata_<session_id>.py` during practice
