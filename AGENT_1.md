# Agent 1: Rust Core & Database

## Mission

Build the foundational Rust infrastructure: database schema, repository layer, SM-2 scheduler, adaptive difficulty tracking, and dependency graph resolution. This is the backbone that all other agents depend on.

## Dependencies

None. You can start immediately in parallel with Agent 2.

## What You're Building

### 1. Database Schema & Migrations
SQLite database with proper migrations at `~/.local/share/kata-sr/kata.db`

### 2. Repository Layer
Clean interface for all database operations with proper error handling

### 3. SM-2 Scheduler
Implement Anki's SM-2 algorithm with 0-3 rating scale

### 4. Adaptive Difficulty Tracker
Track user performance and adjust kata difficulty independently of SM-2

### 5. Dependency Graph Resolver
Manage kata prerequisites and unlock logic

### 6. Python Environment Bootstrap
Detect uv, setup Python virtualenv, cache interpreter path

## Detailed Specifications

### Database Schema

Create these tables with migrations using rusqlite:

```rust
// katas table
CREATE TABLE katas (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    category TEXT NOT NULL,
    description TEXT NOT NULL,
    base_difficulty INTEGER NOT NULL CHECK(base_difficulty >= 1 AND base_difficulty <= 5),
    current_difficulty REAL NOT NULL,
    parent_kata_id INTEGER,
    variation_params TEXT,  // JSON blob or NULL
    next_review_at INTEGER,  // Unix timestamp, NULL for never reviewed
    last_reviewed_at INTEGER,  // Unix timestamp, NULL for never reviewed
    current_ease_factor REAL NOT NULL DEFAULT 2.5,
    current_interval_days INTEGER NOT NULL DEFAULT 1,
    current_repetition_count INTEGER NOT NULL DEFAULT 0,
    created_at INTEGER NOT NULL,
    FOREIGN KEY (parent_kata_id) REFERENCES katas(id)
);

CREATE INDEX idx_next_review ON katas(next_review_at);
CREATE INDEX idx_category ON katas(category);

// kata_dependencies table
CREATE TABLE kata_dependencies (
    kata_id INTEGER NOT NULL,
    depends_on_kata_id INTEGER NOT NULL,
    required_success_count INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (kata_id, depends_on_kata_id),
    FOREIGN KEY (kata_id) REFERENCES katas(id),
    FOREIGN KEY (depends_on_kata_id) REFERENCES katas(id)
);

// sessions table
CREATE TABLE sessions (
    id INTEGER PRIMARY KEY,
    kata_id INTEGER NOT NULL,
    started_at INTEGER NOT NULL,
    completed_at INTEGER,
    test_results_json TEXT,
    num_passed INTEGER,
    num_failed INTEGER,
    num_skipped INTEGER,
    duration_ms INTEGER,
    quality_rating INTEGER CHECK(quality_rating >= 0 AND quality_rating <= 3),
    FOREIGN KEY (kata_id) REFERENCES katas(id)
);

CREATE INDEX idx_sessions_kata ON sessions(kata_id);
CREATE INDEX idx_sessions_completed ON sessions(completed_at);

// daily_stats table
CREATE TABLE daily_stats (
    date TEXT PRIMARY KEY,  // YYYY-MM-DD format
    total_reviews INTEGER NOT NULL DEFAULT 0,
    total_successes INTEGER NOT NULL DEFAULT 0,
    success_rate REAL,
    streak_days INTEGER NOT NULL DEFAULT 0,
    categories_json TEXT  // JSON: {"transformers": 5, "graphs": 3, ...}
);
```

### SM-2 Scheduler Implementation

```rust
// src/core/scheduler.rs

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QualityRating {
    Again = 0,  // Complete failure
    Hard = 1,   // Struggled but passed
    Good = 2,   // Normal difficulty
    Easy = 3,   // Too easy
}

pub struct SM2State {
    pub ease_factor: f64,
    pub interval_days: i64,
    pub repetition_count: i64,
}

impl SM2State {
    pub fn new() -> Self {
        Self {
            ease_factor: 2.5,
            interval_days: 1,
            repetition_count: 0,
        }
    }

    pub fn update(&mut self, quality: QualityRating) -> i64 {
        // Returns next interval in days
        match quality {
            QualityRating::Again => {
                // Reset
                self.interval_days = 1;
                self.repetition_count = 0;
                self.ease_factor = (self.ease_factor - 0.2).max(1.3);
            }
            QualityRating::Hard => {
                // Minimal growth
                self.ease_factor = (self.ease_factor - 0.15).max(1.3);
                self.interval_days = (self.interval_days as f64 * 1.2).round() as i64;
                self.repetition_count += 1;
            }
            QualityRating::Good => {
                // Standard SM-2
                if self.repetition_count == 0 {
                    self.interval_days = 1;
                } else if self.repetition_count == 1 {
                    self.interval_days = 6;
                } else {
                    self.interval_days = (self.interval_days as f64 * self.ease_factor).round() as i64;
                }
                self.repetition_count += 1;
            }
            QualityRating::Easy => {
                // Accelerated growth
                self.ease_factor = (self.ease_factor + 0.15).min(2.5);
                if self.repetition_count == 0 {
                    self.interval_days = 1;
                } else if self.repetition_count == 1 {
                    self.interval_days = 6;
                } else {
                    self.interval_days = (self.interval_days as f64 * self.ease_factor * 1.3).round() as i64;
                }
                self.repetition_count += 1;
            }
        }

        self.interval_days
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sm2_good_progression() {
        let mut state = SM2State::new();

        // First review: Good -> 1 day
        assert_eq!(state.update(QualityRating::Good), 1);
        assert_eq!(state.repetition_count, 1);

        // Second review: Good -> 6 days
        assert_eq!(state.update(QualityRating::Good), 6);
        assert_eq!(state.repetition_count, 2);

        // Third review: Good -> 6 * 2.5 = 15 days
        assert_eq!(state.update(QualityRating::Good), 15);
        assert_eq!(state.repetition_count, 3);
    }

    #[test]
    fn test_sm2_reset_on_again() {
        let mut state = SM2State::new();
        state.update(QualityRating::Good);
        state.update(QualityRating::Good);
        state.update(QualityRating::Good);

        // Should be at 15 days, repetition 3
        assert_eq!(state.interval_days, 15);

        // Rating Again should reset
        state.update(QualityRating::Again);
        assert_eq!(state.interval_days, 1);
        assert_eq!(state.repetition_count, 0);
    }

    // Add more tests for Hard, Easy, edge cases
}
```

### Adaptive Difficulty Tracker

```rust
// src/core/difficulty.rs

pub struct DifficultyTracker {
    window_size: usize,  // How many recent sessions to consider
}

impl DifficultyTracker {
    pub fn new(window_size: usize) -> Self {
        Self { window_size }
    }

    pub fn compute_adjustment(&self, recent_sessions: &[bool]) -> f64 {
        // recent_sessions: vec of pass/fail for last N attempts
        if recent_sessions.len() < 3 {
            return 0.0;  // Not enough data
        }

        let success_rate = recent_sessions.iter().filter(|&&x| x).count() as f64
            / recent_sessions.len() as f64;

        if success_rate > 0.9 {
            0.2  // Increase difficulty
        } else if success_rate < 0.5 {
            -0.3  // Decrease difficulty
        } else {
            0.0  // No change
        }
    }

    pub fn apply_adjustment(current: f64, adjustment: f64) -> f64 {
        (current + adjustment).max(1.0).min(5.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_increase_on_high_success() {
        let tracker = DifficultyTracker::new(5);
        let sessions = vec![true, true, true, true, true];
        assert_eq!(tracker.compute_adjustment(&sessions), 0.2);
    }

    #[test]
    fn test_decrease_on_low_success() {
        let tracker = DifficultyTracker::new(5);
        let sessions = vec![false, false, true, false, false];
        assert_eq!(tracker.compute_adjustment(&sessions), -0.3);
    }
}
```

### Dependency Graph Resolver

```rust
// src/core/dependencies.rs

use std::collections::{HashMap, HashSet};

pub struct DependencyGraph {
    // kata_id -> list of (depends_on_id, required_count)
    dependencies: HashMap<i64, Vec<(i64, i64)>>,
}

impl DependencyGraph {
    pub fn new() -> Self {
        Self {
            dependencies: HashMap::new(),
        }
    }

    pub fn add_dependency(&mut self, kata_id: i64, depends_on: i64, required_count: i64) {
        self.dependencies
            .entry(kata_id)
            .or_insert_with(Vec::new)
            .push((depends_on, required_count));
    }

    pub fn is_unlocked(&self, kata_id: i64, success_counts: &HashMap<i64, i64>) -> bool {
        if let Some(deps) = self.dependencies.get(&kata_id) {
            for (dep_id, required) in deps {
                let count = success_counts.get(dep_id).unwrap_or(&0);
                if count < required {
                    return false;
                }
            }
        }
        true
    }

    pub fn get_blocking_dependencies(&self, kata_id: i64, success_counts: &HashMap<i64, i64>)
        -> Vec<(i64, i64, i64)> {
        // Returns (dep_id, required_count, current_count) for unsatisfied dependencies
        let mut blocking = Vec::new();

        if let Some(deps) = self.dependencies.get(&kata_id) {
            for (dep_id, required) in deps {
                let current = *success_counts.get(dep_id).unwrap_or(&0);
                if current < *required {
                    blocking.push((*dep_id, *required, current));
                }
            }
        }

        blocking
    }

    // Topological sort for suggested learning order
    pub fn topological_sort(&self) -> Result<Vec<i64>, String> {
        // Implement Kahn's algorithm
        // Return error if cycle detected
        unimplemented!("Implement topological sort")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_unlock() {
        let mut graph = DependencyGraph::new();
        graph.add_dependency(2, 1, 1);  // Kata 2 depends on Kata 1

        let mut counts = HashMap::new();
        assert_eq!(graph.is_unlocked(2, &counts), false);

        counts.insert(1, 1);
        assert_eq!(graph.is_unlocked(2, &counts), true);
    }
}
```

### Repository Layer

```rust
// src/db/repo.rs

use rusqlite::{Connection, Result, params};
use chrono::{DateTime, Utc};

pub struct KataRepository {
    conn: Connection,
}

impl KataRepository {
    pub fn new(db_path: &str) -> Result<Self> {
        let conn = Connection::open(db_path)?;
        Ok(Self { conn })
    }

    pub fn run_migrations(&self) -> Result<()> {
        // Run SQL migrations from schema.rs
        unimplemented!("Implement migrations")
    }

    pub fn get_katas_due(&self, before: DateTime<Utc>) -> Result<Vec<Kata>> {
        // Query katas where next_review_at <= before
        unimplemented!()
    }

    pub fn get_kata_by_id(&self, id: i64) -> Result<Option<Kata>> {
        unimplemented!()
    }

    pub fn create_kata(&self, kata: &NewKata) -> Result<i64> {
        // Insert and return id
        unimplemented!()
    }

    pub fn update_kata_after_review(&self, kata_id: i64, state: &SM2State, next_review: DateTime<Utc>) -> Result<()> {
        // Update next_review_at, ease_factor, interval_days, repetition_count
        unimplemented!()
    }

    pub fn create_session(&self, session: &Session) -> Result<i64> {
        unimplemented!()
    }

    pub fn get_recent_sessions(&self, kata_id: i64, limit: usize) -> Result<Vec<Session>> {
        unimplemented!()
    }

    pub fn get_success_counts(&self) -> Result<HashMap<i64, i64>> {
        // Count successful sessions per kata (for dependency checking)
        unimplemented!()
    }

    pub fn load_dependency_graph(&self) -> Result<DependencyGraph> {
        // Load from kata_dependencies table
        unimplemented!()
    }

    // Add more methods as needed
}

pub struct Kata {
    pub id: i64,
    pub name: String,
    pub category: String,
    pub description: String,
    pub base_difficulty: i32,
    pub current_difficulty: f64,
    pub parent_kata_id: Option<i64>,
    pub variation_params: Option<String>,
    pub next_review_at: Option<DateTime<Utc>>,
    pub last_reviewed_at: Option<DateTime<Utc>>,
    pub current_ease_factor: f64,
    pub current_interval_days: i64,
    pub current_repetition_count: i64,
}

pub struct NewKata {
    pub name: String,
    pub category: String,
    pub description: String,
    pub base_difficulty: i32,
    pub parent_kata_id: Option<i64>,
    pub variation_params: Option<String>,
}

pub struct Session {
    pub kata_id: i64,
    pub started_at: DateTime<Utc>,
    pub completed_at: Option<DateTime<Utc>>,
    pub test_results_json: Option<String>,
    pub num_passed: Option<i32>,
    pub num_failed: Option<i32>,
    pub num_skipped: Option<i32>,
    pub duration_ms: Option<i64>,
    pub quality_rating: Option<i32>,
}
```

### Python Environment Bootstrap

```rust
// src/main.rs or src/python_env.rs

use std::process::Command;
use std::path::{Path, PathBuf};

pub struct PythonEnv {
    interpreter_path: PathBuf,
}

impl PythonEnv {
    pub fn setup() -> Result<Self, String> {
        // 1. Check uv is installed
        if !Command::new("which")
            .arg("uv")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false) {
            return Err("uv not found. Install with: curl -LsSf https://astral.sh/uv/install.sh | sh".to_string());
        }

        // 2. Check if venv exists
        let venv_path = Path::new("katas/.venv");
        if !venv_path.exists() {
            println!("Setting up Python environment...");
            let status = Command::new("uv")
                .args(&["sync", "--directory", "katas/"])
                .status()
                .map_err(|e| format!("Failed to run uv sync: {}", e))?;

            if !status.success() {
                return Err("uv sync failed".to_string());
            }
        }

        // 3. Resolve interpreter path
        let interpreter = venv_path.join("bin/python");
        if !interpreter.exists() {
            return Err("Python interpreter not found in venv".to_string());
        }

        Ok(Self { interpreter_path: interpreter })
    }

    pub fn interpreter_path(&self) -> &Path {
        &self.interpreter_path
    }
}
```

## File Structure You'll Create

```
src/
├── main.rs                 # CLI entry, setup Python env, basic arg parsing
├── core/
│   ├── mod.rs
│   ├── scheduler.rs        # SM2State + QualityRating
│   ├── difficulty.rs       # DifficultyTracker
│   └── dependencies.rs     # DependencyGraph
├── db/
│   ├── mod.rs
│   ├── schema.rs           # SQL migrations as string constants
│   └── repo.rs             # KataRepository + models
└── lib.rs                  # Export public interfaces
```

## Cargo.toml Dependencies

```toml
[package]
name = "kata-sr"
version = "0.1.0"
edition = "2021"

[dependencies]
rusqlite = { version = "0.31", features = ["bundled"] }
chrono = "0.4"
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
clap = { version = "4.5", features = ["derive"] }
anyhow = "1.0"
thiserror = "1.0"
```

Don't add ratatui yet - that's for Agent 3.

## Testing Requirements

Write unit tests for:
- SM-2 state transitions (all quality ratings)
- Edge cases (ease factor bounds, interval calculations)
- Difficulty tracker adjustments
- Dependency graph (simple unlock, multiple deps, cycles)
- Repository CRUD operations

Run tests with: `cargo test`

## Acceptance Criteria

- [ ] Database schema created with migrations
- [ ] Repository layer with all CRUD operations
- [ ] SM-2 scheduler passes all test cases
- [ ] Adaptive difficulty tracker implemented
- [ ] Dependency graph resolver with unlock logic
- [ ] Python environment bootstrap working
- [ ] All unit tests passing
- [ ] Code follows Rust best practices (no unwrap in production code, proper error handling)

## Handoff to Other Agents

Once complete, your work provides:
- **To Agent 3 (TUI):** KataRepository interface, SM2State, DependencyGraph
- **To Agent 5 (Analytics):** Database schema, repo methods for querying stats

Ensure your public API is well-documented and stable. Agent 3 will import your crates and use these interfaces extensively.

## Notes

- Use `anyhow::Result` for main.rs, `thiserror` for library errors
- Database path: `~/.local/share/kata-sr/kata.db`
- Create parent directories if they don't exist
- Keep SM-2 implementation pure (no side effects, easy to test)
- Write good error messages for Python env setup failures
