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
    quality_rating INTEGER CHECK(quality_rating >= 1 AND quality_rating <= 4),
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

/// SQL migration for creating the kata_tags table.
///
/// Manages many-to-many relationships between katas and tags.
/// Replaces the single category field with flexible tagging.
const MIGRATION_KATA_TAGS: &str = r#"
CREATE TABLE IF NOT EXISTS kata_tags (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kata_id INTEGER NOT NULL,
    tag TEXT NOT NULL,
    FOREIGN KEY (kata_id) REFERENCES katas(id) ON DELETE CASCADE,
    UNIQUE(kata_id, tag)
);

CREATE INDEX IF NOT EXISTS idx_kata_tags_tag ON kata_tags(tag);
CREATE INDEX IF NOT EXISTS idx_kata_tags_kata_id ON kata_tags(kata_id);
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
ALTER TABLE katas ADD COLUMN scheduler_type TEXT DEFAULT 'FSRS';
"#;

/// SQL migration for adding problematic kata flags.
///
/// Adds columns to mark katas as problematic (buggy tests, broken templates, etc.)
/// allowing users to flag issues during practice and batch-fix them later.
const MIGRATION_ADD_PROBLEMATIC_FLAGS: &str = r#"
-- Add problematic kata tracking columns
ALTER TABLE katas ADD COLUMN is_problematic BOOLEAN DEFAULT FALSE;
ALTER TABLE katas ADD COLUMN problematic_notes TEXT;
ALTER TABLE katas ADD COLUMN flagged_at INTEGER;
"#;

/// SQL migration for adding code attempt storage to sessions.
///
/// Adds a column to store the user's code attempt for each session.
/// This enables viewing past solutions in session history.
const MIGRATION_ADD_CODE_ATTEMPT: &str = r#"
-- Add code attempt storage
ALTER TABLE sessions ADD COLUMN code_attempt TEXT;
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
    conn.execute_batch(MIGRATION_KATA_TAGS)?;
    conn.execute_batch(MIGRATION_FSRS_PARAMS)?;

    // Add FSRS columns if they don't exist (for upgrading existing databases)
    add_fsrs_columns_if_needed(conn)?;

    // Migrate existing categories to tags
    migrate_categories_to_tags(conn)?;

    // Migrate all katas to use FSRS-5 as the default scheduler
    migrate_to_fsrs(conn)?;

    // Update sessions table constraint for FSRS 1-4 rating scale
    update_sessions_rating_constraint(conn)?;

    // Add problematic kata tracking columns if they don't exist
    add_problematic_columns_if_needed(conn)?;

    // Add code attempt column if it doesn't exist
    add_code_attempt_column_if_needed(conn)?;

    Ok(())
}

/// Migrates existing category values to the kata_tags table.
///
/// This is a one-time migration that:
/// - Reads all katas with non-empty category values
/// - Creates corresponding tag entries in kata_tags
/// - Preserves the category field for backward compatibility
///
/// Safe to run multiple times (uses INSERT OR IGNORE).
pub fn migrate_categories_to_tags(conn: &Connection) -> Result<()> {
    let mut stmt =
        conn.prepare("SELECT id, category FROM katas WHERE category IS NOT NULL AND category != ''")?;

    let katas: Vec<(i64, String)> = stmt
        .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?
        .collect::<Result<Vec<_>>>()?;

    for (kata_id, category) in katas {
        // Add category as a tag (INSERT OR IGNORE handles duplicates)
        conn.execute(
            "INSERT OR IGNORE INTO kata_tags (kata_id, tag) VALUES (?, ?)",
            rusqlite::params![kata_id, category],
        )?;
    }

    Ok(())
}

/// Migrates all legacy SM-2 katas to FSRS-5.
///
/// This is an upgrade migration for existing databases. Since SM-2 compatibility
/// has been removed, all katas must use FSRS-5. This updates any remaining
/// SM-2 katas or katas with NULL scheduler_type to use FSRS.
///
/// Safe to run multiple times - only affects katas not already using FSRS.
pub fn migrate_to_fsrs(conn: &Connection) -> Result<()> {
    // Only update katas that aren't already using FSRS
    conn.execute(
        "UPDATE katas SET scheduler_type = 'FSRS'
         WHERE scheduler_type != 'FSRS' OR scheduler_type IS NULL",
        [],
    )?;
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

/// Adds problematic kata tracking columns to katas table if they don't already exist.
///
/// This function checks if the is_problematic column exists before attempting to add it,
/// making it safe to call on both new and existing databases.
fn add_problematic_columns_if_needed(conn: &Connection) -> Result<()> {
    // Check if is_problematic column exists
    let column_exists: bool = conn
        .prepare("SELECT COUNT(*) FROM pragma_table_info('katas') WHERE name='is_problematic'")?
        .query_row([], |row| {
            let count: i64 = row.get(0)?;
            Ok(count > 0)
        })?;

    if !column_exists {
        // Add all problematic tracking columns at once
        conn.execute_batch(MIGRATION_ADD_PROBLEMATIC_FLAGS)?;
    }

    Ok(())
}

/// Adds code_attempt column to sessions table if it doesn't already exist.
///
/// This function checks if the code_attempt column exists before attempting to add it,
/// making it safe to call on both new and existing databases.
fn add_code_attempt_column_if_needed(conn: &Connection) -> Result<()> {
    // Check if code_attempt column exists
    let column_exists: bool = conn
        .prepare("SELECT COUNT(*) FROM pragma_table_info('sessions') WHERE name='code_attempt'")?
        .query_row([], |row| {
            let count: i64 = row.get(0)?;
            Ok(count > 0)
        })?;

    if !column_exists {
        // Add code attempt column
        conn.execute_batch(MIGRATION_ADD_CODE_ATTEMPT)?;
    }

    Ok(())
}

/// Updates the sessions table constraint to support FSRS 1-4 rating scale.
///
/// SQLite doesn't allow modifying CHECK constraints, so we need to recreate
/// the table. This migration:
/// 1. Creates a new table with the updated constraint (1-4 instead of 0-3)
/// 2. Copies all existing data
/// 3. Drops the old table
/// 4. Renames the new table
///
/// Safe to run multiple times - checks if migration is needed first.
fn update_sessions_rating_constraint(conn: &Connection) -> Result<()> {
    // Check if the table already has the new constraint by looking at the schema
    let schema: String = conn.query_row(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='sessions'",
        [],
        |row| row.get(0),
    )?;

    // If the schema already contains "quality_rating >= 1", migration is not needed
    if schema.contains("quality_rating >= 1") {
        return Ok(());
    }

    // Recreate the table with the new constraint
    conn.execute_batch(
        r#"
        -- Create new table with updated constraint
        CREATE TABLE IF NOT EXISTS sessions_new (
            id INTEGER PRIMARY KEY,
            kata_id INTEGER NOT NULL,
            started_at INTEGER NOT NULL,
            completed_at INTEGER,
            test_results_json TEXT,
            num_passed INTEGER,
            num_failed INTEGER,
            num_skipped INTEGER,
            duration_ms INTEGER,
            quality_rating INTEGER CHECK(quality_rating >= 1 AND quality_rating <= 4),
            FOREIGN KEY (kata_id) REFERENCES katas(id)
        );

        -- Copy existing data
        INSERT INTO sessions_new
        SELECT id, kata_id, started_at, completed_at, test_results_json,
               num_passed, num_failed, num_skipped, duration_ms,
               CASE
                   WHEN quality_rating IS NULL THEN NULL
                   ELSE quality_rating + 1
               END as quality_rating
        FROM sessions;

        -- Drop old table
        DROP TABLE sessions;

        -- Rename new table
        ALTER TABLE sessions_new RENAME TO sessions;

        -- Recreate indexes
        CREATE INDEX IF NOT EXISTS idx_sessions_kata ON sessions(kata_id);
        CREATE INDEX IF NOT EXISTS idx_sessions_completed ON sessions(completed_at);
        "#,
    )?;

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
