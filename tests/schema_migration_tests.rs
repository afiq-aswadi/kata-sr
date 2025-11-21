//! Database schema migration tests.
//!
//! Tests backward compatibility, data preservation, and error handling during schema migrations.

use chrono::Utc;
use kata_sr::db::repo::{KataRepository, NewKata, NewSession};
use kata_sr::db::schema::{migrate_categories_to_tags, migrate_to_fsrs, run_migrations};
use rusqlite::Connection;

#[test]
fn test_migration_from_empty_database() {
    let conn = Connection::open_in_memory().unwrap();
    let result = run_migrations(&conn);
    assert!(result.is_ok());

    // Verify all tables exist
    let tables = vec!["katas", "kata_dependencies", "sessions", "daily_stats", "kata_tags", "fsrs_params"];
    for table in tables {
        let mut stmt = conn
            .prepare(&format!(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='{}'",
                table
            ))
            .unwrap();
        assert!(stmt.exists([]).unwrap(), "Table {} should exist", table);
    }
}

#[test]
fn test_migration_preserves_existing_data() {
    let repo = KataRepository::new_in_memory().unwrap();
    repo.run_migrations().unwrap();

    // Insert test data
    let new_kata = NewKata {
        name: "test_kata".to_string(),
        category: "algorithms".to_string(),
        description: "Test kata".to_string(),
        base_difficulty: 3,
        parent_kata_id: None,
        variation_params: None,
    };

    let kata_id = repo.create_kata(&new_kata, Utc::now()).unwrap();

    // Create a session
    let session = NewSession {
        kata_id,
        started_at: Utc::now(),
        completed_at: Some(Utc::now()),
        test_results_json: Some("{\"passed\": true}".to_string()),
        num_passed: Some(5),
        num_failed: Some(0),
        num_skipped: Some(0),
        duration_ms: Some(1000),
        quality_rating: Some(3),
        code_attempt: None,
    };

    repo.create_session(&session).unwrap();

    // Run migrations again (should be idempotent)
    repo.run_migrations().unwrap();

    // Verify data is preserved
    let kata = repo.get_kata_by_id(kata_id).unwrap().unwrap();
    assert_eq!(kata.name, "test_kata");
    assert_eq!(kata.category, "algorithms");

    let sessions = repo.get_recent_sessions(kata_id, 10).unwrap();
    assert_eq!(sessions.len(), 1);
}

#[test]
fn test_fsrs_columns_added_to_existing_database() {
    let conn = Connection::open_in_memory().unwrap();

    // Create old schema without FSRS columns
    conn.execute_batch(
        r#"
        CREATE TABLE katas (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL UNIQUE,
            category TEXT NOT NULL,
            description TEXT NOT NULL,
            base_difficulty INTEGER NOT NULL,
            current_difficulty REAL NOT NULL,
            parent_kata_id INTEGER,
            variation_params TEXT,
            next_review_at INTEGER,
            last_reviewed_at INTEGER,
            current_ease_factor REAL NOT NULL DEFAULT 2.5,
            current_interval_days INTEGER NOT NULL DEFAULT 1,
            current_repetition_count INTEGER NOT NULL DEFAULT 0,
            created_at INTEGER NOT NULL
        );
        "#,
    )
    .unwrap();

    // Run migrations (should add FSRS columns)
    run_migrations(&conn).unwrap();

    // Verify FSRS columns exist
    let columns = vec![
        "fsrs_stability",
        "fsrs_difficulty",
        "fsrs_elapsed_days",
        "fsrs_scheduled_days",
        "fsrs_reps",
        "fsrs_lapses",
        "fsrs_state",
        "scheduler_type",
    ];

    for col in columns {
        let count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM pragma_table_info('katas') WHERE name=?",
                [col],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(count, 1, "Column {} should exist", col);
    }
}

#[test]
fn test_categories_migrated_to_tags() {
    let conn = Connection::open_in_memory().unwrap();

    // Create tables
    conn.execute_batch(
        r#"
        CREATE TABLE katas (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL UNIQUE,
            category TEXT NOT NULL,
            description TEXT NOT NULL,
            base_difficulty INTEGER NOT NULL,
            current_difficulty REAL NOT NULL,
            parent_kata_id INTEGER,
            variation_params TEXT,
            next_review_at INTEGER,
            last_reviewed_at INTEGER,
            current_ease_factor REAL NOT NULL DEFAULT 2.5,
            current_interval_days INTEGER NOT NULL DEFAULT 1,
            current_repetition_count INTEGER NOT NULL DEFAULT 0,
            created_at INTEGER NOT NULL
        );

        CREATE TABLE kata_tags (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            kata_id INTEGER NOT NULL,
            tag TEXT NOT NULL,
            FOREIGN KEY (kata_id) REFERENCES katas(id) ON DELETE CASCADE,
            UNIQUE(kata_id, tag)
        );
        "#,
    )
    .unwrap();

    // Insert katas with categories
    conn.execute(
        "INSERT INTO katas (id, name, category, description, base_difficulty, current_difficulty, created_at)
         VALUES (1, 'kata1', 'algorithms', 'Test', 3, 3.0, 0)",
        [],
    )
    .unwrap();

    conn.execute(
        "INSERT INTO katas (id, name, category, description, base_difficulty, current_difficulty, created_at)
         VALUES (2, 'kata2', 'transformers', 'Test', 2, 2.0, 0)",
        [],
    )
    .unwrap();

    // Migrate categories to tags
    migrate_categories_to_tags(&conn).unwrap();

    // Verify tags were created
    let count: i64 = conn
        .query_row("SELECT COUNT(*) FROM kata_tags", [], |row| row.get(0))
        .unwrap();
    assert_eq!(count, 2);

    // Verify specific tags
    let tag1: String = conn
        .query_row(
            "SELECT tag FROM kata_tags WHERE kata_id = 1",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(tag1, "algorithms");

    let tag2: String = conn
        .query_row(
            "SELECT tag FROM kata_tags WHERE kata_id = 2",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(tag2, "transformers");
}

#[test]
fn test_categories_to_tags_migration_is_idempotent() {
    let conn = Connection::open_in_memory().unwrap();

    conn.execute_batch(
        r#"
        CREATE TABLE katas (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL UNIQUE,
            category TEXT NOT NULL,
            description TEXT NOT NULL,
            base_difficulty INTEGER NOT NULL,
            current_difficulty REAL NOT NULL,
            parent_kata_id INTEGER,
            variation_params TEXT,
            next_review_at INTEGER,
            last_reviewed_at INTEGER,
            current_ease_factor REAL NOT NULL DEFAULT 2.5,
            current_interval_days INTEGER NOT NULL DEFAULT 1,
            current_repetition_count INTEGER NOT NULL DEFAULT 0,
            created_at INTEGER NOT NULL
        );

        CREATE TABLE kata_tags (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            kata_id INTEGER NOT NULL,
            tag TEXT NOT NULL,
            FOREIGN KEY (kata_id) REFERENCES katas(id) ON DELETE CASCADE,
            UNIQUE(kata_id, tag)
        );
        "#,
    )
    .unwrap();

    conn.execute(
        "INSERT INTO katas (id, name, category, description, base_difficulty, current_difficulty, created_at)
         VALUES (1, 'kata1', 'algorithms', 'Test', 3, 3.0, 0)",
        [],
    )
    .unwrap();

    // Run migration twice
    migrate_categories_to_tags(&conn).unwrap();
    migrate_categories_to_tags(&conn).unwrap();

    // Should still have only one tag (no duplicates)
    let count: i64 = conn
        .query_row("SELECT COUNT(*) FROM kata_tags WHERE kata_id = 1", [], |row| {
            row.get(0)
        })
        .unwrap();
    assert_eq!(count, 1);
}

#[test]
fn test_sm2_to_fsrs_migration() {
    let conn = Connection::open_in_memory().unwrap();

    // Create table with scheduler_type column
    conn.execute_batch(
        r#"
        CREATE TABLE katas (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL UNIQUE,
            category TEXT NOT NULL,
            description TEXT NOT NULL,
            base_difficulty INTEGER NOT NULL,
            current_difficulty REAL NOT NULL,
            parent_kata_id INTEGER,
            variation_params TEXT,
            next_review_at INTEGER,
            last_reviewed_at INTEGER,
            current_ease_factor REAL NOT NULL DEFAULT 2.5,
            current_interval_days INTEGER NOT NULL DEFAULT 1,
            current_repetition_count INTEGER NOT NULL DEFAULT 0,
            created_at INTEGER NOT NULL,
            fsrs_stability REAL DEFAULT 0.0,
            fsrs_difficulty REAL DEFAULT 0.0,
            fsrs_elapsed_days INTEGER DEFAULT 0,
            fsrs_scheduled_days INTEGER DEFAULT 0,
            fsrs_reps INTEGER DEFAULT 0,
            fsrs_lapses INTEGER DEFAULT 0,
            fsrs_state TEXT DEFAULT 'New',
            scheduler_type TEXT
        );
        "#,
    )
    .unwrap();

    // Insert katas with SM2 and NULL scheduler types
    conn.execute(
        "INSERT INTO katas (id, name, category, description, base_difficulty, current_difficulty, created_at, scheduler_type)
         VALUES (1, 'sm2_kata', 'test', 'Test', 3, 3.0, 0, 'SM2')",
        [],
    )
    .unwrap();

    conn.execute(
        "INSERT INTO katas (id, name, category, description, base_difficulty, current_difficulty, created_at, scheduler_type)
         VALUES (2, 'null_kata', 'test', 'Test', 3, 3.0, 0, NULL)",
        [],
    )
    .unwrap();

    conn.execute(
        "INSERT INTO katas (id, name, category, description, base_difficulty, current_difficulty, created_at, scheduler_type)
         VALUES (3, 'fsrs_kata', 'test', 'Test', 3, 3.0, 0, 'FSRS')",
        [],
    )
    .unwrap();

    // Run migration
    migrate_to_fsrs(&conn).unwrap();

    // Verify all katas use FSRS
    let count: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM katas WHERE scheduler_type = 'FSRS'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(count, 3);
}

#[test]
fn test_sessions_rating_constraint_migration() -> Result<(), Box<dyn std::error::Error>> {
    let conn = Connection::open_in_memory().unwrap();

    // Create old sessions table with 0-3 rating scale
    conn.execute_batch(
        r#"
        CREATE TABLE IF NOT EXISTS katas (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL UNIQUE,
            category TEXT NOT NULL,
            description TEXT NOT NULL,
            base_difficulty INTEGER NOT NULL,
            current_difficulty REAL NOT NULL,
            created_at INTEGER NOT NULL
        );

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
        "#,
    )
    .unwrap();

    // Insert test kata
    conn.execute(
        "INSERT INTO katas (id, name, category, description, base_difficulty, current_difficulty, created_at)
         VALUES (1, 'test', 'test', 'Test', 3, 3.0, 0)",
        [],
    )
    .unwrap();

    // Insert sessions with old rating scale (0-3)
    for rating in 0..=3 {
        conn.execute(
            "INSERT INTO sessions (kata_id, started_at, quality_rating)
             VALUES (1, 0, ?)",
            [rating],
        )
        .unwrap();
    }

    // Run full migrations (includes constraint update)
    run_migrations(&conn).unwrap();

    // Verify ratings were converted (0->1, 1->2, 2->3, 3->4)
    let ratings: Vec<i32> = conn
        .prepare("SELECT quality_rating FROM sessions ORDER BY quality_rating")?
        .query_map([], |row| row.get(0))?
        .collect::<Result<Vec<_>, _>>()
        .unwrap();

    assert_eq!(ratings, vec![1, 2, 3, 4]);

    Ok(())
}

#[test]
fn test_problematic_columns_added() {
    let conn = Connection::open_in_memory().unwrap();

    // Create initial schema without problematic columns
    conn.execute_batch(
        r#"
        CREATE TABLE katas (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL UNIQUE,
            category TEXT NOT NULL,
            description TEXT NOT NULL,
            base_difficulty INTEGER NOT NULL,
            current_difficulty REAL NOT NULL,
            created_at INTEGER NOT NULL
        );
        "#,
    )
    .unwrap();

    // Run migrations (should add problematic columns)
    run_migrations(&conn).unwrap();

    // Verify problematic columns exist
    let columns = vec!["is_problematic", "problematic_notes", "flagged_at"];
    for col in columns {
        let count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM pragma_table_info('katas') WHERE name=?",
                [col],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(count, 1, "Column {} should exist", col);
    }
}

#[test]
fn test_code_attempt_column_added() {
    let conn = Connection::open_in_memory().unwrap();

    // Create initial schema without code_attempt column
    conn.execute_batch(
        r#"
        CREATE TABLE katas (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL UNIQUE,
            category TEXT NOT NULL,
            description TEXT NOT NULL,
            base_difficulty INTEGER NOT NULL,
            current_difficulty REAL NOT NULL,
            created_at INTEGER NOT NULL
        );

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
            quality_rating INTEGER,
            FOREIGN KEY (kata_id) REFERENCES katas(id)
        );
        "#,
    )
    .unwrap();

    // Run migrations (should add code_attempt column)
    run_migrations(&conn).unwrap();

    // Verify code_attempt column exists
    let count: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM pragma_table_info('sessions') WHERE name='code_attempt'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(count, 1);
}

#[test]
fn test_migration_handles_empty_categories() {
    let conn = Connection::open_in_memory().unwrap();

    conn.execute_batch(
        r#"
        CREATE TABLE katas (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL UNIQUE,
            category TEXT NOT NULL,
            description TEXT NOT NULL,
            base_difficulty INTEGER NOT NULL,
            current_difficulty REAL NOT NULL,
            created_at INTEGER NOT NULL
        );

        CREATE TABLE kata_tags (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            kata_id INTEGER NOT NULL,
            tag TEXT NOT NULL,
            FOREIGN KEY (kata_id) REFERENCES katas(id) ON DELETE CASCADE,
            UNIQUE(kata_id, tag)
        );
        "#,
    )
    .unwrap();

    // Insert kata with empty category
    conn.execute(
        "INSERT INTO katas (id, name, category, description, base_difficulty, current_difficulty, created_at)
         VALUES (1, 'kata1', '', 'Test', 3, 3.0, 0)",
        [],
    )
    .unwrap();

    // Migrate (should not create tag for empty category)
    migrate_categories_to_tags(&conn).unwrap();

    let count: i64 = conn
        .query_row("SELECT COUNT(*) FROM kata_tags", [], |row| row.get(0))
        .unwrap();
    assert_eq!(count, 0);
}

#[test]
fn test_migration_preserves_null_values() {
    let repo = KataRepository::new_in_memory().unwrap();
    repo.run_migrations().unwrap();

    let new_kata = NewKata {
        name: "test_kata".to_string(),
        category: "test".to_string(),
        description: "Test".to_string(),
        base_difficulty: 3,
        parent_kata_id: None,
        variation_params: None,
    };

    let kata_id = repo.create_kata(&new_kata, Utc::now()).unwrap();

    // Create session with NULL values
    let session = NewSession {
        kata_id,
        started_at: Utc::now(),
        completed_at: None,
        test_results_json: None,
        num_passed: None,
        num_failed: None,
        num_skipped: None,
        duration_ms: None,
        quality_rating: None,
        code_attempt: None,
    };

    repo.create_session(&session).unwrap();

    // Run migrations again
    repo.run_migrations().unwrap();

    // Verify NULL values are preserved
    let sessions = repo.get_recent_sessions(kata_id, 10).unwrap();
    assert_eq!(sessions.len(), 1);
    assert!(sessions[0].completed_at.is_none());
    assert!(sessions[0].quality_rating.is_none());
}

#[test]
fn test_foreign_key_constraints_maintained_after_migration() {
    let conn = Connection::open_in_memory().unwrap();
    conn.execute("PRAGMA foreign_keys = ON", []).unwrap();

    run_migrations(&conn).unwrap();

    // Create a kata
    conn.execute(
        "INSERT INTO katas (id, name, category, description, base_difficulty, current_difficulty, created_at)
         VALUES (1, 'test', 'test', 'Test', 3, 3.0, 0)",
        [],
    )
    .unwrap();

    // Try to insert session with non-existent kata_id (should fail)
    let result = conn.execute(
        "INSERT INTO sessions (kata_id, started_at) VALUES (999, 0)",
        [],
    );
    assert!(result.is_err());

    // Try to insert dependency with non-existent kata (should fail)
    let result = conn.execute(
        "INSERT INTO kata_dependencies (kata_id, depends_on_kata_id, required_success_count)
         VALUES (999, 1, 1)",
        [],
    );
    assert!(result.is_err());
}

#[test]
fn test_indexes_recreated_after_migration() {
    let conn = Connection::open_in_memory().unwrap();
    run_migrations(&conn).unwrap();

    // Verify critical indexes exist
    let indexes = vec![
        "idx_next_review",
        "idx_category",
        "idx_sessions_kata",
        "idx_sessions_completed",
        "idx_kata_tags_tag",
        "idx_kata_tags_kata_id",
    ];

    for index in indexes {
        let exists: bool = conn
            .prepare(&format!(
                "SELECT COUNT(*) > 0 FROM sqlite_master WHERE type='index' AND name='{}'",
                index
            ))
            .unwrap()
            .query_row([], |row| row.get(0))
            .unwrap();
        assert!(exists, "Index {} should exist", index);
    }
}
