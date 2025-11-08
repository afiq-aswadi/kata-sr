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

/// SQL migration for creating the fsrs_params table.
///
/// Stores optimized FSRS-5 parameters (19 weights).
/// Multiple parameter sets can be stored with timestamps for versioning.
const MIGRATION_FSRS_PARAMS: &str = r#"
CREATE TABLE IF NOT EXISTS fsrs_params (
    id INTEGER PRIMARY KEY,
    w0 REAL NOT NULL, w1 REAL NOT NULL, w2 REAL NOT NULL, w3 REAL NOT NULL,
    w4 REAL NOT NULL, w5 REAL NOT NULL, w6 REAL NOT NULL, w7 REAL NOT NULL,
    w8 REAL NOT NULL, w9 REAL NOT NULL, w10 REAL NOT NULL, w11 REAL NOT NULL,
    w12 REAL NOT NULL, w13 REAL NOT NULL, w14 REAL NOT NULL, w15 REAL NOT NULL,
    w16 REAL NOT NULL, w17 REAL NOT NULL, w18 REAL NOT NULL,
    created_at INTEGER NOT NULL
);
"#;

/// SQL migration for adding FSRS columns to katas table.
///
/// Adds FSRS-5 state tracking columns alongside existing SM-2 columns.
/// The `scheduler_type` column determines which algorithm is active ('SM2' or 'FSRS').
const MIGRATION_ADD_FSRS_COLUMNS: &str = r#"
-- Add FSRS state columns
ALTER TABLE katas ADD COLUMN fsrs_stability REAL DEFAULT 0.0;
ALTER TABLE katas ADD COLUMN fsrs_difficulty REAL DEFAULT 0.0;
ALTER TABLE katas ADD COLUMN fsrs_elapsed_days INTEGER DEFAULT 0;
ALTER TABLE katas ADD COLUMN fsrs_scheduled_days INTEGER DEFAULT 0;
ALTER TABLE katas ADD COLUMN fsrs_reps INTEGER DEFAULT 0;
ALTER TABLE katas ADD COLUMN fsrs_lapses INTEGER DEFAULT 0;
ALTER TABLE katas ADD COLUMN fsrs_state TEXT DEFAULT 'New';
ALTER TABLE katas ADD COLUMN scheduler_type TEXT DEFAULT 'SM2';
"#;

/// Runs all database migrations.
///
/// Creates all tables and indexes if they don't exist. Safe to call
/// multiple times as all migrations use `IF NOT EXISTS`.
///
/// Also adds FSRS columns if they don't exist yet (for upgrading existing databases).
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
    conn.execute_batch(MIGRATION_FSRS_PARAMS)?;

    // Add FSRS columns if they don't exist (for upgrading existing databases)
    add_fsrs_columns_if_needed(conn)?;

    Ok(())
}

/// Adds FSRS columns to katas table if they don't already exist.
///
/// This function checks if the FSRS columns exist before attempting to add them,
/// making it safe to call on both new and existing databases.
fn add_fsrs_columns_if_needed(conn: &Connection) -> Result<()> {
    // Check if fsrs_stability column exists
    let column_exists: bool = conn
        .prepare("SELECT COUNT(*) FROM pragma_table_info('katas') WHERE name='fsrs_stability'")?
        .query_row([], |row| {
            let count: i64 = row.get(0)?;
            Ok(count > 0)
        })?;

    if !column_exists {
        // Add all FSRS columns at once
        conn.execute_batch(MIGRATION_ADD_FSRS_COLUMNS)?;
    }

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
