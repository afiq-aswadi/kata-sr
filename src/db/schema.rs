//! Database schema definitions and migration management.
//!
//! This module provides SQL migrations for the kata spaced repetition system.
//! The schema includes tables for:
//! - katas: kata metadata and SM-2 scheduling state
//! - kata_dependencies: prerequisite relationships between katas
//! - sessions: practice session history with test results
//! - daily_stats: aggregated daily statistics

use rusqlite::{Connection, Result};

/// SQL migration for creating the katas table.
///
/// Stores kata metadata and current SM-2 spaced repetition state.
/// The `next_review_at` field is indexed for efficient "due today" queries.
const MIGRATION_KATAS: &str = r#"
CREATE TABLE IF NOT EXISTS katas (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    category TEXT NOT NULL,
    description TEXT NOT NULL,
    base_difficulty INTEGER NOT NULL CHECK(base_difficulty >= 1 AND base_difficulty <= 5),
    current_difficulty REAL NOT NULL,
    parent_kata_id INTEGER,
    variation_params TEXT,
    next_review_at INTEGER,
    last_reviewed_at INTEGER,
    current_ease_factor REAL NOT NULL DEFAULT 2.5,
    current_interval_days INTEGER NOT NULL DEFAULT 1,
    current_repetition_count INTEGER NOT NULL DEFAULT 0,
    created_at INTEGER NOT NULL,
    FOREIGN KEY (parent_kata_id) REFERENCES katas(id)
);

CREATE INDEX IF NOT EXISTS idx_next_review ON katas(next_review_at);
CREATE INDEX IF NOT EXISTS idx_category ON katas(category);
"#;

/// SQL migration for creating the kata_dependencies table.
///
/// Manages prerequisite relationships where one kata depends on another.
/// A kata must be successfully completed `required_success_count` times
/// before dependent katas are unlocked.
const MIGRATION_KATA_DEPENDENCIES: &str = r#"
CREATE TABLE IF NOT EXISTS kata_dependencies (
    kata_id INTEGER NOT NULL,
    depends_on_kata_id INTEGER NOT NULL,
    required_success_count INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (kata_id, depends_on_kata_id),
    FOREIGN KEY (kata_id) REFERENCES katas(id),
    FOREIGN KEY (depends_on_kata_id) REFERENCES katas(id)
);
"#;

/// SQL migration for creating the sessions table.
///
/// Records all practice sessions including test results and quality ratings.
/// The `test_results_json` field stores detailed pytest output in JSON format.
const MIGRATION_SESSIONS: &str = r#"
CREATE TABLE IF NOT EXISTS sessions (
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

CREATE INDEX IF NOT EXISTS idx_sessions_kata ON sessions(kata_id);
CREATE INDEX IF NOT EXISTS idx_sessions_completed ON sessions(completed_at);
"#;

/// SQL migration for creating the daily_stats table.
///
/// Aggregates daily review statistics including success rates and streaks.
/// The `categories_json` field stores per-category review counts in JSON format.
const MIGRATION_DAILY_STATS: &str = r#"
CREATE TABLE IF NOT EXISTS daily_stats (
    date TEXT PRIMARY KEY,
    total_reviews INTEGER NOT NULL DEFAULT 0,
    total_successes INTEGER NOT NULL DEFAULT 0,
    success_rate REAL,
    streak_days INTEGER NOT NULL DEFAULT 0,
    categories_json TEXT
);
"#;

/// Runs all database migrations.
///
/// Creates all tables and indexes if they don't exist. Safe to call
/// multiple times as all migrations use `IF NOT EXISTS`.
///
/// # Arguments
///
/// * `conn` - Active database connection
///
/// # Examples
///
/// ```no_run
/// use rusqlite::Connection;
/// use kata_sr::db::schema::run_migrations;
///
/// let conn = Connection::open("kata.db")?;
/// run_migrations(&conn)?;
/// # Ok::<(), rusqlite::Error>(())
/// ```
pub fn run_migrations(conn: &Connection) -> Result<()> {
    conn.execute_batch(MIGRATION_KATAS)?;
    conn.execute_batch(MIGRATION_KATA_DEPENDENCIES)?;
    conn.execute_batch(MIGRATION_SESSIONS)?;
    conn.execute_batch(MIGRATION_DAILY_STATS)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rusqlite::Connection;

    #[test]
    fn test_migrations_run_successfully() {
        let conn = Connection::open_in_memory().unwrap();
        assert!(run_migrations(&conn).is_ok());
    }

    #[test]
    fn test_migrations_idempotent() {
        let conn = Connection::open_in_memory().unwrap();
        run_migrations(&conn).unwrap();
        // running again should not fail
        assert!(run_migrations(&conn).is_ok());
    }

    #[test]
    fn test_katas_table_created() {
        let conn = Connection::open_in_memory().unwrap();
        run_migrations(&conn).unwrap();

        let mut stmt = conn
            .prepare("SELECT name FROM sqlite_master WHERE type='table' AND name='katas'")
            .unwrap();
        let exists = stmt.exists([]).unwrap();
        assert!(exists);
    }

    #[test]
    fn test_indexes_created() {
        let conn = Connection::open_in_memory().unwrap();
        run_migrations(&conn).unwrap();

        let mut stmt = conn
            .prepare("SELECT name FROM sqlite_master WHERE type='index' AND name='idx_next_review'")
            .unwrap();
        let exists = stmt.exists([]).unwrap();
        assert!(exists);
    }
}
