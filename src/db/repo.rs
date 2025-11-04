//! Repository layer for database operations.
//!
//! This module provides a clean interface for all database CRUD operations,
//! including kata management, session tracking, and dependency relationships.
//! All database access should go through the `KataRepository` to ensure
//! consistent error handling and transaction management.

use crate::core::dependencies::DependencyGraph;
use crate::core::scheduler::SM2State;
use chrono::{DateTime, LocalResult, TimeZone, Utc};
use rusqlite::types::Type;
use rusqlite::{params, Connection, Result, Row};
use thiserror::Error;
use std::collections::HashMap;
use std::path::Path;

/// Main repository interface for database operations.
///
/// Provides methods for managing katas, sessions, dependencies, and stats.
/// All methods use the `?` operator for error propagation rather than unwrap.
///
/// # Examples
///
/// ```no_run
/// use kata_sr::db::repo::KataRepository;
///
/// let repo = KataRepository::new("kata.db")?;
/// repo.run_migrations()?;
///
/// let due_katas = repo.get_katas_due(chrono::Utc::now())?;
/// # Ok::<(), rusqlite::Error>(())
/// ```
pub struct KataRepository {
    conn: Connection,
}

impl KataRepository {
    /// Opens a database connection and creates a new repository.
    ///
    /// Creates parent directories if they don't exist.
    ///
    /// # Arguments
    ///
    /// * `db_path` - Path to the SQLite database file
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use kata_sr::db::repo::KataRepository;
    /// let repo = KataRepository::new("~/.local/share/kata-sr/kata.db")?;
    /// # Ok::<(), rusqlite::Error>(())
    /// ```
    pub fn new<P: AsRef<Path>>(db_path: P) -> Result<Self> {
        let path = db_path.as_ref();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                rusqlite::Error::InvalidPath(
                    format!("Failed to create parent directory: {}", e).into(),
                )
            })?;
        }

        let conn = Connection::open(path)?;
        configure_connection(&conn)?;
        Ok(Self { conn })
    }

    /// Creates an in-memory database for testing.
    ///
    /// # Examples
    ///
    /// ```
    /// # use kata_sr::db::repo::KataRepository;
    /// let repo = KataRepository::new_in_memory()?;
    /// repo.run_migrations()?;
    /// # Ok::<(), rusqlite::Error>(())
    /// ```
    pub fn new_in_memory() -> Result<Self> {
        let conn = Connection::open_in_memory()?;
        configure_connection(&conn)?;
        Ok(Self { conn })
    }

    /// Runs all database migrations.
    ///
    /// Safe to call multiple times. Creates all tables and indexes if they don't exist.
    pub fn run_migrations(&self) -> Result<()> {
        crate::db::schema::run_migrations(&self.conn)
    }

    /// Retrieves all katas that are due for review before the given timestamp.
    ///
    /// A kata is due if its `next_review_at` is NULL (never reviewed) or
    /// less than or equal to the given timestamp.
    ///
    /// # Arguments
    ///
    /// * `before` - Only return katas due at or before this time
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use kata_sr::db::repo::KataRepository;
    /// # use chrono::Utc;
    /// # let repo = KataRepository::new("kata.db")?;
    /// let due_katas = repo.get_katas_due(Utc::now())?;
    /// for kata in due_katas {
    ///     println!("Kata {} is due", kata.name);
    /// }
    /// # Ok::<(), rusqlite::Error>(())
    /// ```
    pub fn get_katas_due(&self, before: DateTime<Utc>) -> Result<Vec<Kata>> {
        let timestamp = before.timestamp();
        let mut stmt = self.conn.prepare(
            "SELECT id, name, category, description, base_difficulty, current_difficulty,
                    parent_kata_id, variation_params, next_review_at, last_reviewed_at,
                    current_ease_factor, current_interval_days, current_repetition_count, created_at
             FROM katas
             WHERE next_review_at IS NULL OR next_review_at <= ?1
             ORDER BY next_review_at ASC NULLS FIRST",
        )?;

        let katas = stmt
            .query_map([timestamp], row_to_kata)?
            .collect::<Result<Vec<_>>>()?;

        Ok(katas)
    }

    /// Retrieves all katas from the database.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use kata_sr::db::repo::KataRepository;
    /// # let repo = KataRepository::new("kata.db")?;
    /// let all_katas = repo.get_all_katas()?;
    /// # Ok::<(), rusqlite::Error>(())
    /// ```
    pub fn get_all_katas(&self) -> Result<Vec<Kata>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, name, category, description, base_difficulty, current_difficulty,
                    parent_kata_id, variation_params, next_review_at, last_reviewed_at,
                    current_ease_factor, current_interval_days, current_repetition_count, created_at
             FROM katas
             ORDER BY created_at DESC",
        )?;

        let katas = stmt
            .query_map([], row_to_kata)?
            .collect::<Result<Vec<_>>>()?;

        Ok(katas)
    }

    /// Retrieves a single kata by ID.
    ///
    /// Returns `None` if the kata doesn't exist.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use kata_sr::db::repo::KataRepository;
    /// # let repo = KataRepository::new("kata.db")?;
    /// if let Some(kata) = repo.get_kata_by_id(1)? {
    ///     println!("Found kata: {}", kata.name);
    /// }
    /// # Ok::<(), rusqlite::Error>(())
    /// ```
    pub fn get_kata_by_id(&self, id: i64) -> Result<Option<Kata>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, name, category, description, base_difficulty, current_difficulty,
                    parent_kata_id, variation_params, next_review_at, last_reviewed_at,
                    current_ease_factor, current_interval_days, current_repetition_count, created_at
             FROM katas
             WHERE id = ?1",
        )?;

        let mut rows = stmt.query([id])?;
        if let Some(row) = rows.next()? {
            Ok(Some(row_to_kata(row)?))
        } else {
            Ok(None)
        }
    }

    /// Creates a new kata in the database.
    ///
    /// Returns the ID of the newly created kata.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use kata_sr::db::repo::{KataRepository, NewKata};
    /// # use chrono::Utc;
    /// # let repo = KataRepository::new("kata.db")?;
    /// let new_kata = NewKata {
    ///     name: "multi_head_attention".to_string(),
    ///     category: "transformers".to_string(),
    ///     description: "Implement multi-head attention".to_string(),
    ///     base_difficulty: 4,
    ///     parent_kata_id: None,
    ///     variation_params: None,
    /// };
    ///
    /// let kata_id = repo.create_kata(&new_kata, Utc::now())?;
    /// # Ok::<(), rusqlite::Error>(())
    /// ```
    pub fn create_kata(&self, kata: &NewKata, created_at: DateTime<Utc>) -> Result<i64> {
        let timestamp = created_at.timestamp();
        self.conn.execute(
            "INSERT INTO katas (name, category, description, base_difficulty, current_difficulty,
                               parent_kata_id, variation_params, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![
                kata.name,
                kata.category,
                kata.description,
                kata.base_difficulty,
                kata.base_difficulty as f64,
                kata.parent_kata_id,
                kata.variation_params,
                timestamp,
            ],
        )?;

        Ok(self.conn.last_insert_rowid())
    }

    /// Updates a kata's scheduling state after a review.
    ///
    /// Updates the SM-2 state (ease factor, interval, repetition count),
    /// next review timestamp, and last reviewed timestamp.
    ///
    /// # Arguments
    ///
    /// * `kata_id` - ID of the kata to update
    /// * `state` - New SM-2 state
    /// * `next_review` - When the kata should be reviewed next
    /// * `reviewed_at` - When the review was completed
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use kata_sr::db::repo::KataRepository;
    /// # use kata_sr::core::scheduler::{SM2State, QualityRating};
    /// # use chrono::{Utc, Duration};
    /// # let repo = KataRepository::new("kata.db")?;
    /// let mut state = SM2State::new();
    /// let interval = state.update(QualityRating::Good);
    /// let next_review = Utc::now() + Duration::days(interval);
    ///
    /// repo.update_kata_after_review(1, &state, next_review, Utc::now())?;
    /// # Ok::<(), rusqlite::Error>(())
    /// ```
    pub fn update_kata_after_review(
        &self,
        kata_id: i64,
        state: &SM2State,
        next_review: DateTime<Utc>,
        reviewed_at: DateTime<Utc>,
    ) -> Result<()> {
        self.conn.execute(
            "UPDATE katas
             SET next_review_at = ?1,
                 last_reviewed_at = ?2,
                 current_ease_factor = ?3,
                 current_interval_days = ?4,
                 current_repetition_count = ?5
             WHERE id = ?6",
            params![
                next_review.timestamp(),
                reviewed_at.timestamp(),
                state.ease_factor,
                state.interval_days,
                state.repetition_count,
                kata_id,
            ],
        )?;

        Ok(())
    }

    /// Updates a kata's current difficulty.
    ///
    /// Used by the adaptive difficulty tracker to adjust kata difficulty
    /// based on user performance.
    pub fn update_kata_difficulty(&self, kata_id: i64, new_difficulty: f64) -> Result<()> {
        self.conn.execute(
            "UPDATE katas SET current_difficulty = ?1 WHERE id = ?2",
            params![new_difficulty, kata_id],
        )?;
        Ok(())
    }

    /// Creates a new practice session record.
    ///
    /// Returns the ID of the newly created session.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use kata_sr::db::repo::{KataRepository, NewSession};
    /// # use chrono::Utc;
    /// # let repo = KataRepository::new("kata.db")?;
    /// let session = NewSession {
    ///     kata_id: 1,
    ///     started_at: Utc::now(),
    ///     completed_at: Some(Utc::now()),
    ///     test_results_json: Some("{\"passed\": true}".to_string()),
    ///     num_passed: Some(5),
    ///     num_failed: Some(0),
    ///     num_skipped: Some(0),
    ///     duration_ms: Some(1234),
    ///     quality_rating: Some(2),
    /// };
    ///
    /// let session_id = repo.create_session(&session)?;
    /// # Ok::<(), rusqlite::Error>(())
    /// ```
    pub fn create_session(&self, session: &NewSession) -> Result<i64> {
        self.conn.execute(
            "INSERT INTO sessions (kata_id, started_at, completed_at, test_results_json,
                                  num_passed, num_failed, num_skipped, duration_ms, quality_rating)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            params![
                session.kata_id,
                session.started_at.timestamp(),
                session.completed_at.map(|dt| dt.timestamp()),
                session.test_results_json,
                session.num_passed,
                session.num_failed,
                session.num_skipped,
                session.duration_ms,
                session.quality_rating,
            ],
        )?;

        Ok(self.conn.last_insert_rowid())
    }

    /// Retrieves recent sessions for a kata, ordered by most recent first.
    ///
    /// # Arguments
    ///
    /// * `kata_id` - The kata to get sessions for
    /// * `limit` - Maximum number of sessions to return
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use kata_sr::db::repo::KataRepository;
    /// # let repo = KataRepository::new("kata.db")?;
    /// let recent = repo.get_recent_sessions(1, 5)?;
    /// # Ok::<(), rusqlite::Error>(())
    /// ```
    pub fn get_recent_sessions(&self, kata_id: i64, limit: usize) -> Result<Vec<Session>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, kata_id, started_at, completed_at, test_results_json,
                    num_passed, num_failed, num_skipped, duration_ms, quality_rating
             FROM sessions
             WHERE kata_id = ?1
             ORDER BY started_at DESC
             LIMIT ?2",
        )?;

        let sessions = stmt
            .query_map(params![kata_id, limit], row_to_session)?
            .collect::<Result<Vec<_>>>()?;

        Ok(sessions)
    }

    /// Retrieves all sessions for a specific date.
    ///
    /// # Arguments
    ///
    /// * `date` - Date in YYYY-MM-DD format
    pub fn get_sessions_for_date(&self, date: &str) -> Result<Vec<Session>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, kata_id, started_at, completed_at, test_results_json,
                    num_passed, num_failed, num_skipped, duration_ms, quality_rating
             FROM sessions
             WHERE date(completed_at, 'unixepoch') = ?1
             ORDER BY started_at DESC",
        )?;

        let sessions = stmt
            .query_map([date], row_to_session)?
            .collect::<Result<Vec<_>>>()?;

        Ok(sessions)
    }

    /// Counts successful sessions for each kata.
    ///
    /// A session is considered successful if it has a quality rating >= 1 (Hard or better).
    /// Used by the dependency graph to determine if prerequisites are satisfied.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use kata_sr::db::repo::KataRepository;
    /// # let repo = KataRepository::new("kata.db")?;
    /// let counts = repo.get_success_counts()?;
    /// if let Some(&count) = counts.get(&1) {
    ///     println!("Kata 1 has been completed successfully {} times", count);
    /// }
    /// # Ok::<(), rusqlite::Error>(())
    /// ```
    pub fn get_success_counts(&self) -> Result<HashMap<i64, i64>> {
        let mut stmt = self.conn.prepare(
            "SELECT kata_id, COUNT(*) as count
             FROM sessions
             WHERE quality_rating >= 1
             GROUP BY kata_id",
        )?;

        let mut counts = HashMap::new();
        let rows = stmt.query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?;

        for row in rows {
            let (kata_id, count): (i64, i64) = row?;
            counts.insert(kata_id, count);
        }

        Ok(counts)
    }

    /// Loads the complete dependency graph from the database.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use kata_sr::db::repo::KataRepository;
    /// # let repo = KataRepository::new("kata.db")?;
    /// let graph = repo.load_dependency_graph()?;
    /// let success_counts = repo.get_success_counts()?;
    ///
    /// if graph.is_unlocked(5, &success_counts) {
    ///     println!("Kata 5 is unlocked!");
    /// }
    /// # Ok::<(), rusqlite::Error>(())
    /// ```
    pub fn load_dependency_graph(&self) -> Result<DependencyGraph> {
        let mut graph = DependencyGraph::new();
        let mut stmt = self.conn.prepare(
            "SELECT kata_id, depends_on_kata_id, required_success_count FROM kata_dependencies",
        )?;

        let rows = stmt.query_map([], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)))?;

        for row in rows {
            let (kata_id, depends_on, required_count): (i64, i64, i64) = row?;
            graph.add_dependency(kata_id, depends_on, required_count);
        }

        Ok(graph)
    }

    /// Adds a dependency relationship between two katas.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use kata_sr::db::repo::KataRepository;
    /// # let repo = KataRepository::new("kata.db")?;
    /// // Kata 2 requires kata 1 to be completed 3 times
    /// repo.add_dependency(2, 1, 3)?;
    /// # Ok::<(), rusqlite::Error>(())
    /// ```
    pub fn add_dependency(
        &self,
        kata_id: i64,
        depends_on_kata_id: i64,
        required_success_count: i64,
    ) -> Result<()> {
        self.conn.execute(
            "INSERT INTO kata_dependencies (kata_id, depends_on_kata_id, required_success_count)
             VALUES (?1, ?2, ?3)",
            params![kata_id, depends_on_kata_id, required_success_count],
        )?;
        Ok(())
    }
}

/// Full kata record from the database.
#[derive(Debug, Clone, PartialEq)]
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
    pub created_at: DateTime<Utc>,
}

impl Kata {
    /// Returns the current SM-2 state for this kata.
    pub fn sm2_state(&self) -> SM2State {
        SM2State {
            ease_factor: self.current_ease_factor,
            interval_days: self.current_interval_days,
            repetition_count: self.current_repetition_count,
        }
    }
}

/// New kata to be inserted into the database.
#[derive(Debug, Clone)]
pub struct NewKata {
    pub name: String,
    pub category: String,
    pub description: String,
    pub base_difficulty: i32,
    pub parent_kata_id: Option<i64>,
    pub variation_params: Option<String>,
}

/// Practice session record.
#[derive(Debug, Clone)]
pub struct Session {
    pub id: i64,
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

/// New session to be inserted into the database.
#[derive(Debug, Clone)]
pub struct NewSession {
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

fn row_to_kata(row: &Row) -> Result<Kata> {
    let next_review_ts: Option<i64> = row.get(8)?;
    let last_reviewed_ts: Option<i64> = row.get(9)?;
    let created_ts: i64 = row.get(13)?;

    let next_review_at = convert_optional_timestamp(next_review_ts, 8, "next_review_at")?;
    let last_reviewed_at = convert_optional_timestamp(last_reviewed_ts, 9, "last_reviewed_at")?;
    let created_at = convert_timestamp(created_ts, 13, "created_at")?;

    Ok(Kata {
        id: row.get(0)?,
        name: row.get(1)?,
        category: row.get(2)?,
        description: row.get(3)?,
        base_difficulty: row.get(4)?,
        current_difficulty: row.get(5)?,
        parent_kata_id: row.get(6)?,
        variation_params: row.get(7)?,
        next_review_at,
        last_reviewed_at,
        current_ease_factor: row.get(10)?,
        current_interval_days: row.get(11)?,
        current_repetition_count: row.get(12)?,
        created_at,
    })
}

fn row_to_session(row: &Row) -> Result<Session> {
    let started_ts: i64 = row.get(2)?;
    let completed_ts: Option<i64> = row.get(3)?;

    let started_at = convert_timestamp(started_ts, 2, "started_at")?;
    let completed_at = convert_optional_timestamp(completed_ts, 3, "completed_at")?;

    Ok(Session {
        id: row.get(0)?,
        kata_id: row.get(1)?,
        started_at,
        completed_at,
        test_results_json: row.get(4)?,
        num_passed: row.get(5)?,
        num_failed: row.get(6)?,
        num_skipped: row.get(7)?,
        duration_ms: row.get(8)?,
        quality_rating: row.get(9)?,
    })
}

#[derive(Debug, Error)]
#[error("invalid timestamp {value} for column {column}")]
struct TimestampConversionError {
    column: &'static str,
    value: i64,
}

fn configure_connection(conn: &Connection) -> Result<()> {
    conn.execute("PRAGMA foreign_keys = ON;", [])?;
    Ok(())
}

fn convert_timestamp(
    value: i64,
    column_index: usize,
    column_name: &'static str,
) -> Result<DateTime<Utc>> {
    match Utc.timestamp_opt(value, 0) {
        LocalResult::Single(dt) => Ok(dt),
        LocalResult::None | LocalResult::Ambiguous(_, _) => Err(
            rusqlite::Error::FromSqlConversionFailure(
                column_index,
                Type::Integer,
                Box::new(TimestampConversionError {
                    column: column_name,
                    value,
                }),
            ),
        ),
    }
}

fn convert_optional_timestamp(
    value: Option<i64>,
    column_index: usize,
    column_name: &'static str,
) -> Result<Option<DateTime<Utc>>> {
    value
        .map(|ts| convert_timestamp(ts, column_index, column_name))
        .transpose()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup_test_repo() -> KataRepository {
        let repo = KataRepository::new_in_memory().unwrap();
        repo.run_migrations().unwrap();
        repo
    }

    #[test]
    fn test_create_and_get_kata() {
        let repo = setup_test_repo();
        let new_kata = NewKata {
            name: "test_kata".to_string(),
            category: "test".to_string(),
            description: "A test kata".to_string(),
            base_difficulty: 3,
            parent_kata_id: None,
            variation_params: None,
        };

        let kata_id = repo.create_kata(&new_kata, Utc::now()).unwrap();
        let kata = repo.get_kata_by_id(kata_id).unwrap().unwrap();

        assert_eq!(kata.name, "test_kata");
        assert_eq!(kata.base_difficulty, 3);
        assert_eq!(kata.current_difficulty, 3.0);
    }

    #[test]
    fn test_get_katas_due() {
        let repo = setup_test_repo();
        let new_kata = NewKata {
            name: "due_kata".to_string(),
            category: "test".to_string(),
            description: "Due kata".to_string(),
            base_difficulty: 2,
            parent_kata_id: None,
            variation_params: None,
        };

        repo.create_kata(&new_kata, Utc::now()).unwrap();

        let due = repo.get_katas_due(Utc::now()).unwrap();
        assert_eq!(due.len(), 1);
        assert_eq!(due[0].name, "due_kata");
    }

    #[test]
    fn test_update_kata_after_review() {
        let repo = setup_test_repo();
        let new_kata = NewKata {
            name: "review_kata".to_string(),
            category: "test".to_string(),
            description: "Review test".to_string(),
            base_difficulty: 3,
            parent_kata_id: None,
            variation_params: None,
        };

        let kata_id = repo.create_kata(&new_kata, Utc::now()).unwrap();

        let mut state = SM2State::new();
        state.update(crate::core::scheduler::QualityRating::Good);
        let next_review = Utc::now() + chrono::Duration::days(1);

        repo.update_kata_after_review(kata_id, &state, next_review, Utc::now())
            .unwrap();

        let kata = repo.get_kata_by_id(kata_id).unwrap().unwrap();
        assert!(kata.last_reviewed_at.is_some());
        assert!(kata.next_review_at.is_some());
    }

    #[test]
    fn test_create_and_get_session() {
        let repo = setup_test_repo();
        let new_kata = NewKata {
            name: "session_kata".to_string(),
            category: "test".to_string(),
            description: "Session test".to_string(),
            base_difficulty: 2,
            parent_kata_id: None,
            variation_params: None,
        };

        let kata_id = repo.create_kata(&new_kata, Utc::now()).unwrap();

        let session = NewSession {
            kata_id,
            started_at: Utc::now(),
            completed_at: Some(Utc::now()),
            test_results_json: Some("{}".to_string()),
            num_passed: Some(5),
            num_failed: Some(0),
            num_skipped: Some(0),
            duration_ms: Some(1000),
            quality_rating: Some(2),
        };

        let session_id = repo.create_session(&session).unwrap();
        let sessions = repo.get_recent_sessions(kata_id, 10).unwrap();

        assert_eq!(sessions.len(), 1);
        assert_eq!(sessions[0].id, session_id);
        assert_eq!(sessions[0].num_passed, Some(5));
    }

    #[test]
    fn test_success_counts() {
        let repo = setup_test_repo();
        let new_kata = NewKata {
            name: "count_kata".to_string(),
            category: "test".to_string(),
            description: "Count test".to_string(),
            base_difficulty: 2,
            parent_kata_id: None,
            variation_params: None,
        };

        let kata_id = repo.create_kata(&new_kata, Utc::now()).unwrap();

        for rating in [2, 1, 0, 2] {
            let session = NewSession {
                kata_id,
                started_at: Utc::now(),
                completed_at: Some(Utc::now()),
                test_results_json: None,
                num_passed: None,
                num_failed: None,
                num_skipped: None,
                duration_ms: None,
                quality_rating: Some(rating),
            };
            repo.create_session(&session).unwrap();
        }

        let counts = repo.get_success_counts().unwrap();
        assert_eq!(counts.get(&kata_id), Some(&3));
    }

    #[test]
    fn test_add_and_load_dependency() {
        let repo = setup_test_repo();

        // create katas first to satisfy foreign key constraints
        let kata1_id = repo
            .create_kata(
                &NewKata {
                    name: "kata1".to_string(),
                    category: "test".to_string(),
                    description: "First".to_string(),
                    base_difficulty: 2,
                    parent_kata_id: None,
                    variation_params: None,
                },
                Utc::now(),
            )
            .unwrap();

        let kata2_id = repo
            .create_kata(
                &NewKata {
                    name: "kata2".to_string(),
                    category: "test".to_string(),
                    description: "Second".to_string(),
                    base_difficulty: 3,
                    parent_kata_id: None,
                    variation_params: None,
                },
                Utc::now(),
            )
            .unwrap();

        repo.add_dependency(kata2_id, kata1_id, 3).unwrap();

        let graph = repo.load_dependency_graph().unwrap();
        let counts = HashMap::new();

        assert!(!graph.is_unlocked(kata2_id, &counts));
    }
}
