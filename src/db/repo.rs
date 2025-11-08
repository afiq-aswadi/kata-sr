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
use std::collections::HashMap;
use std::path::Path;
use thiserror::Error;

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

    /// Replaces all dependency edges for the given kata.
    ///
    /// Existing dependencies are removed and replaced with the provided list.
    /// Duplicate dependency IDs are ignored. Self-dependencies are skipped.
    ///
    /// # Arguments
    ///
    /// * `kata_id` - The kata whose dependencies should be replaced
    /// * `dependency_ids` - Slice of prerequisite kata IDs (each requires one success by default)
    pub fn replace_dependencies(&self, kata_id: i64, dependency_ids: &[i64]) -> Result<()> {
        let tx = self.conn.unchecked_transaction()?;
        tx.execute(
            "DELETE FROM kata_dependencies WHERE kata_id = ?1",
            params![kata_id],
        )?;

        let mut stmt = tx.prepare(
            "INSERT INTO kata_dependencies (kata_id, depends_on_kata_id, required_success_count)
             VALUES (?1, ?2, 1)",
        )?;

        let mut seen = std::collections::HashSet::new();
        for &dep_id in dependency_ids {
            if dep_id == kata_id || !seen.insert(dep_id) {
                continue;
            }
            stmt.execute(params![kata_id, dep_id])?;
        }

        drop(stmt);
        tx.commit()?;
        Ok(())
    }

    /// Calculates the current streak of consecutive days with completed sessions.
    ///
    /// Counts backwards from today, stopping at the first day with no completed sessions.
    /// Returns 0 if no sessions were completed today.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use kata_sr::db::repo::KataRepository;
    /// # let repo = KataRepository::new("kata.db")?;
    /// let streak = repo.get_current_streak()?;
    /// println!("Current streak: {} days", streak);
    /// # Ok::<(), rusqlite::Error>(())
    /// ```
    pub fn get_current_streak(&self) -> Result<i32> {
        // get all distinct dates with completed sessions, ordered descending
        let mut stmt = self.conn.prepare(
            "SELECT DISTINCT date(completed_at, 'unixepoch') as review_date
             FROM sessions
             WHERE completed_at IS NOT NULL
             ORDER BY review_date DESC",
        )?;

        let dates: Vec<String> = stmt
            .query_map([], |row| row.get(0))?
            .collect::<Result<Vec<_>>>()?;

        if dates.is_empty() {
            return Ok(0);
        }

        // get today's date in YYYY-MM-DD format
        let today = chrono::Utc::now().format("%Y-%m-%d").to_string();

        // if no sessions today, streak is 0
        if dates[0] != today {
            return Ok(0);
        }

        // count consecutive days
        let mut streak = 1;
        let mut current_date = chrono::Utc::now().date_naive();

        for date_str in dates.iter().skip(1) {
            // parse the date string
            let date = chrono::NaiveDate::parse_from_str(date_str, "%Y-%m-%d")
                .map_err(|_| rusqlite::Error::InvalidQuery)?;

            // check if it's the previous day
            let expected_date = current_date - chrono::Duration::days(1);
            if date == expected_date {
                streak += 1;
                current_date = date;
            } else {
                break;
            }
        }

        Ok(streak)
    }

    /// Counts the number of completed sessions today.
    ///
    /// Only counts sessions where completed_at is not NULL.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use kata_sr::db::repo::KataRepository;
    /// # let repo = KataRepository::new("kata.db")?;
    /// let count = repo.get_reviews_count_today()?;
    /// println!("Reviews completed today: {}", count);
    /// # Ok::<(), rusqlite::Error>(())
    /// ```
    pub fn get_reviews_count_today(&self) -> Result<i32> {
        let today = chrono::Utc::now().format("%Y-%m-%d").to_string();

        let count: i32 = self.conn.query_row(
            "SELECT COUNT(*)
             FROM sessions
             WHERE completed_at IS NOT NULL
               AND date(completed_at, 'unixepoch') = ?1",
            [today],
            |row| row.get(0),
        )?;

        Ok(count)
    }

    /// Calculates success rate over the last n days.
    ///
    /// Success is defined as quality_rating >= 2 (Good or Easy).
    /// Returns 0.0 if there are no completed sessions in the time period.
    ///
    /// # Arguments
    ///
    /// * `n` - Number of days to look back
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use kata_sr::db::repo::KataRepository;
    /// # let repo = KataRepository::new("kata.db")?;
    /// let success_rate = repo.get_success_rate_last_n_days(7)?;
    /// println!("Success rate (last 7 days): {:.1}%", success_rate * 100.0);
    /// # Ok::<(), rusqlite::Error>(())
    /// ```
    pub fn get_success_rate_last_n_days(&self, n: usize) -> Result<f64> {
        let cutoff_date = (chrono::Utc::now() - chrono::Duration::days(n as i64))
            .format("%Y-%m-%d")
            .to_string();

        let (total, successful): (i32, i32) = self.conn.query_row(
            "SELECT COUNT(*) as total,
                    COALESCE(SUM(CASE WHEN quality_rating >= 2 THEN 1 ELSE 0 END), 0) as successful
             FROM sessions
             WHERE completed_at IS NOT NULL
               AND date(completed_at, 'unixepoch') >= ?1",
            [cutoff_date],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )?;

        if total == 0 {
            Ok(0.0)
        } else {
            Ok(successful as f64 / total as f64)
        }
    }

    /// Returns the next scheduled review timestamp, if any katas are pending in the future.
    ///
    /// This is useful for displaying when the user should return after clearing the queue.
    pub fn get_next_scheduled_review(&self) -> Result<Option<DateTime<Utc>>> {
        let next_timestamp: Option<i64> = self.conn.query_row(
            "SELECT MIN(next_review_at)
             FROM katas
             WHERE next_review_at IS NOT NULL
               AND next_review_at > strftime('%s', 'now')",
            [],
            |row| row.get(0),
        )?;

        convert_optional_timestamp(next_timestamp, 0, "next_review_at_min")
    }

    // ===== Debug/Management Methods =====

    /// Resets a specific kata's SM-2 state to initial values.
    ///
    /// Sets ease_factor=2.5, interval_days=1, repetition_count=0.
    /// Sets next_review_at to now to make it immediately due.
    ///
    /// # Arguments
    ///
    /// * `kata_id` - ID of the kata to reset
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use kata_sr::db::repo::KataRepository;
    /// # let repo = KataRepository::new("kata.db")?;
    /// repo.reset_kata_sm2_state(1)?;
    /// # Ok::<(), rusqlite::Error>(())
    /// ```
    pub fn reset_kata_sm2_state(&self, kata_id: i64) -> Result<()> {
        let now = Utc::now().timestamp();
        self.conn.execute(
            "UPDATE katas
             SET current_ease_factor = 2.5,
                 current_interval_days = 1,
                 current_repetition_count = 0,
                 next_review_at = ?1
             WHERE id = ?2",
            params![now, kata_id],
        )?;
        Ok(())
    }

    /// Resets all katas' SM-2 states to initial values.
    ///
    /// Makes all katas immediately due for review.
    pub fn reset_all_sm2_states(&self) -> Result<()> {
        let now = Utc::now().timestamp();
        self.conn.execute(
            "UPDATE katas
             SET current_ease_factor = 2.5,
                 current_interval_days = 1,
                 current_repetition_count = 0,
                 next_review_at = ?1",
            [now],
        )?;
        Ok(())
    }

    /// Makes a kata due immediately without resetting its SM-2 state.
    ///
    /// Only sets next_review_at to now. Preserves ease_factor,
    /// interval_days, and repetition_count for continued progression.
    ///
    /// # Arguments
    ///
    /// * `kata_id` - ID of the kata to make due
    pub fn force_kata_due(&self, kata_id: i64) -> Result<()> {
        let now = Utc::now().timestamp();
        self.conn.execute(
            "UPDATE katas SET next_review_at = ?1 WHERE id = ?2",
            params![now, kata_id],
        )?;
        Ok(())
    }

    /// Deletes all session records.
    ///
    /// Use for testing or when you want to reset practice history
    /// while keeping kata definitions.
    pub fn clear_all_sessions(&self) -> Result<()> {
        self.conn.execute("DELETE FROM sessions", [])?;
        Ok(())
    }

    /// Deletes all daily statistics records.
    pub fn clear_daily_stats(&self) -> Result<()> {
        self.conn.execute("DELETE FROM daily_stats", [])?;
        Ok(())
    }

    /// Performs a complete database reset, deleting all data.
    ///
    /// Deletes all sessions, dependencies, katas, and daily stats.
    /// Returns the database to its initial state (empty but with schema intact).
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use kata_sr::db::repo::KataRepository;
    /// # let repo = KataRepository::new("kata.db")?;
    /// repo.delete_all_data()?;
    /// # Ok::<(), rusqlite::Error>(())
    /// ```
    pub fn delete_all_data(&self) -> Result<()> {
        // Delete in order to respect foreign key constraints
        self.conn.execute("DELETE FROM sessions", [])?;
        self.conn.execute("DELETE FROM kata_dependencies", [])?;
        self.conn.execute("DELETE FROM katas", [])?;
        self.conn.execute("DELETE FROM daily_stats", [])?;
        Ok(())
    }

    /// Inserts or replaces daily statistics for a specific date.
    ///
    /// # Arguments
    ///
    /// * `stats` - Daily statistics to upsert
    pub fn upsert_daily_stats(&self, stats: &DailyStats) -> Result<()> {
        self.conn.execute(
            "INSERT OR REPLACE INTO daily_stats
             (date, total_reviews, total_successes, success_rate, streak_days, categories_json)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                &stats.date,
                stats.total_reviews,
                stats.total_successes,
                stats.success_rate,
                stats.streak_days,
                &stats.categories_json,
            ],
        )?;
        Ok(())
    }

    /// Gets daily statistics for a specific date.
    ///
    /// # Arguments
    ///
    /// * `date` - Date string in YYYY-MM-DD format
    pub fn get_daily_stats(&self, date: &str) -> Result<Option<DailyStats>> {
        let mut stmt = self.conn.prepare(
            "SELECT date, total_reviews, total_successes, success_rate, streak_days, categories_json
             FROM daily_stats
             WHERE date = ?1",
        )?;

        let result = stmt.query_row([date], |row| {
            Ok(DailyStats {
                date: row.get(0)?,
                total_reviews: row.get(1)?,
                total_successes: row.get(2)?,
                success_rate: row.get(3)?,
                streak_days: row.get(4)?,
                categories_json: row.get(5)?,
            })
        });

        match result {
            Ok(stats) => Ok(Some(stats)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e),
        }
    }

    /// Gets daily statistics for a date range.
    ///
    /// # Arguments
    ///
    /// * `start_date` - Start date (inclusive) in YYYY-MM-DD format
    /// * `end_date` - End date (inclusive) in YYYY-MM-DD format
    pub fn get_daily_stats_range(
        &self,
        start_date: &str,
        end_date: &str,
    ) -> Result<Vec<DailyStats>> {
        let mut stmt = self.conn.prepare(
            "SELECT date, total_reviews, total_successes, success_rate, streak_days, categories_json
             FROM daily_stats
             WHERE date >= ?1 AND date <= ?2
             ORDER BY date ASC",
        )?;

        let stats = stmt
            .query_map([start_date, end_date], |row| {
                Ok(DailyStats {
                    date: row.get(0)?,
                    total_reviews: row.get(1)?,
                    total_successes: row.get(2)?,
                    success_rate: row.get(3)?,
                    streak_days: row.get(4)?,
                    categories_json: row.get(5)?,
                })
            })?
            .collect::<Result<Vec<_>>>()?;

        Ok(stats)
    }

    /// Deletes a kata and all its associated data.
    ///
    /// CASCADE deletes all sessions and dependencies for this kata.
    /// Foreign key constraints must be enabled.
    ///
    /// # Arguments
    ///
    /// * `kata_id` - ID of the kata to delete
    pub fn delete_kata(&self, kata_id: i64) -> Result<()> {
        // delete sessions first (foreign key constraint)
        self.conn
            .execute("DELETE FROM sessions WHERE kata_id = ?1", [kata_id])?;

        // delete dependencies where this kata is either side
        self.conn.execute(
            "DELETE FROM kata_dependencies WHERE kata_id = ?1 OR depends_on_kata_id = ?1",
            [kata_id],
        )?;

        // delete the kata itself
        self.conn
            .execute("DELETE FROM katas WHERE id = ?1", [kata_id])?;

        Ok(())
    }

    /// Finds a kata by its name.
    ///
    /// # Arguments
    ///
    /// * `name` - The kata name to search for
    ///
    /// # Returns
    ///
    /// * `Ok(Some(Kata))` if found
    /// * `Ok(None)` if not found
    pub fn get_kata_by_name(&self, name: &str) -> Result<Option<Kata>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, name, category, description, base_difficulty,
                    current_difficulty, parent_kata_id, variation_params,
                    next_review_at, last_reviewed_at, current_ease_factor,
                    current_interval_days, current_repetition_count, created_at
             FROM katas
             WHERE name = ?1",
        )?;

        let mut rows = stmt.query([name])?;

        if let Some(row) = rows.next()? {
            Ok(Some(row_to_kata(row)?))
        } else {
            Ok(None)
        }
    }

    /// Updates kata metadata without affecting SM-2 state.
    ///
    /// Used during reimport to update description, category, and base difficulty
    /// while preserving the user's review progress.
    ///
    /// # Arguments
    ///
    /// * `kata_id` - ID of the kata to update
    /// * `description` - New description
    /// * `category` - New category
    /// * `base_difficulty` - New base difficulty
    pub fn update_kata_metadata(
        &self,
        kata_id: i64,
        description: &str,
        category: &str,
        base_difficulty: i32,
    ) -> Result<()> {
        self.conn.execute(
            "UPDATE katas
             SET description = ?1,
                 category = ?2,
                 base_difficulty = ?3
             WHERE id = ?4",
            params![description, category, base_difficulty, kata_id],
        )?;
        Ok(())
    }

    /// Gets database statistics for debugging and inspection.
    ///
    /// Returns counts of katas, sessions, dependencies, and streak info.
    pub fn get_database_stats(&self) -> Result<DatabaseStats> {
        let katas_total: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM katas", [], |row| row.get(0))?;

        let now = Utc::now().timestamp();
        let katas_due: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM katas
             WHERE next_review_at IS NULL OR next_review_at <= ?1",
            [now],
            |row| row.get(0),
        )?;

        let katas_scheduled = katas_total - katas_due;

        let sessions_total: i64 =
            self.conn
                .query_row("SELECT COUNT(*) FROM sessions", [], |row| row.get(0))?;

        let (sessions_passed, sessions_failed): (i64, i64) = self.conn.query_row(
            "SELECT
                 COALESCE(SUM(CASE WHEN quality_rating >= 2 THEN 1 ELSE 0 END), 0) as passed,
                 COALESCE(SUM(CASE WHEN quality_rating < 2 THEN 1 ELSE 0 END), 0) as failed
             FROM sessions
             WHERE quality_rating IS NOT NULL",
            [],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )?;

        let dependencies_count: i64 =
            self.conn
                .query_row("SELECT COUNT(*) FROM kata_dependencies", [], |row| {
                    row.get(0)
                })?;

        let current_streak = self.get_current_streak()? as i64;

        Ok(DatabaseStats {
            katas_total,
            katas_due,
            katas_scheduled,
            sessions_total,
            sessions_passed,
            sessions_failed,
            dependencies_count,
            current_streak,
        })
    }
}

/// Database statistics for debug/inspection purposes.
#[derive(Debug, Clone, serde::Serialize)]
pub struct DatabaseStats {
    pub katas_total: i64,
    pub katas_due: i64,
    pub katas_scheduled: i64,
    pub sessions_total: i64,
    pub sessions_passed: i64,
    pub sessions_failed: i64,
    pub dependencies_count: i64,
    pub current_streak: i64,
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

/// Daily statistics record.
#[derive(Debug, Clone)]
pub struct DailyStats {
    pub date: String,
    pub total_reviews: i32,
    pub total_successes: i32,
    pub success_rate: f64,
    pub streak_days: i32,
    pub categories_json: String,
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
        LocalResult::None | LocalResult::Ambiguous(_, _) => {
            Err(rusqlite::Error::FromSqlConversionFailure(
                column_index,
                Type::Integer,
                Box::new(TimestampConversionError {
                    column: column_name,
                    value,
                }),
            ))
        }
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

    #[test]
    fn test_replace_dependencies() {
        let repo = setup_test_repo();

        let kata_a = repo
            .create_kata(
                &NewKata {
                    name: "a".to_string(),
                    category: "test".to_string(),
                    description: "A".to_string(),
                    base_difficulty: 1,
                    parent_kata_id: None,
                    variation_params: None,
                },
                Utc::now(),
            )
            .unwrap();

        let kata_b = repo
            .create_kata(
                &NewKata {
                    name: "b".to_string(),
                    category: "test".to_string(),
                    description: "B".to_string(),
                    base_difficulty: 1,
                    parent_kata_id: None,
                    variation_params: None,
                },
                Utc::now(),
            )
            .unwrap();

        let kata_c = repo
            .create_kata(
                &NewKata {
                    name: "c".to_string(),
                    category: "test".to_string(),
                    description: "C".to_string(),
                    base_difficulty: 1,
                    parent_kata_id: None,
                    variation_params: None,
                },
                Utc::now(),
            )
            .unwrap();

        repo.add_dependency(kata_c, kata_a, 1).unwrap();
        repo.add_dependency(kata_c, kata_b, 1).unwrap();

        repo.replace_dependencies(kata_c, &[kata_a]).unwrap();

        let graph = repo.load_dependency_graph().unwrap();
        let counts = HashMap::new();
        let blocking = graph.get_blocking_dependencies(kata_c, &counts);

        assert_eq!(blocking.len(), 1);
        assert_eq!(blocking[0].0, kata_a);
    }

    #[test]
    fn test_get_current_streak_no_sessions() {
        let repo = setup_test_repo();
        let streak = repo.get_current_streak().unwrap();
        assert_eq!(streak, 0);
    }

    #[test]
    fn test_get_current_streak_with_sessions() {
        let repo = setup_test_repo();
        let kata_id = repo
            .create_kata(
                &NewKata {
                    name: "streak_kata".to_string(),
                    category: "test".to_string(),
                    description: "Streak test".to_string(),
                    base_difficulty: 2,
                    parent_kata_id: None,
                    variation_params: None,
                },
                Utc::now(),
            )
            .unwrap();

        // create sessions for today and yesterday
        let now = Utc::now();
        let yesterday = now - chrono::Duration::days(1);

        for timestamp in [now, yesterday] {
            let session = NewSession {
                kata_id,
                started_at: timestamp,
                completed_at: Some(timestamp),
                test_results_json: None,
                num_passed: None,
                num_failed: None,
                num_skipped: None,
                duration_ms: None,
                quality_rating: Some(2),
            };
            repo.create_session(&session).unwrap();
        }

        let streak = repo.get_current_streak().unwrap();
        assert_eq!(streak, 2);
    }

    #[test]
    fn test_get_current_streak_broken() {
        let repo = setup_test_repo();
        let kata_id = repo
            .create_kata(
                &NewKata {
                    name: "broken_streak_kata".to_string(),
                    category: "test".to_string(),
                    description: "Broken streak test".to_string(),
                    base_difficulty: 2,
                    parent_kata_id: None,
                    variation_params: None,
                },
                Utc::now(),
            )
            .unwrap();

        // create session 2 days ago, but not today or yesterday
        let two_days_ago = Utc::now() - chrono::Duration::days(2);

        let session = NewSession {
            kata_id,
            started_at: two_days_ago,
            completed_at: Some(two_days_ago),
            test_results_json: None,
            num_passed: None,
            num_failed: None,
            num_skipped: None,
            duration_ms: None,
            quality_rating: Some(2),
        };
        repo.create_session(&session).unwrap();

        let streak = repo.get_current_streak().unwrap();
        assert_eq!(streak, 0);
    }

    #[test]
    fn test_get_reviews_count_today_no_sessions() {
        let repo = setup_test_repo();
        let count = repo.get_reviews_count_today().unwrap();
        assert_eq!(count, 0);
    }

    #[test]
    fn test_get_reviews_count_today_with_sessions() {
        let repo = setup_test_repo();
        let kata_id = repo
            .create_kata(
                &NewKata {
                    name: "review_count_kata".to_string(),
                    category: "test".to_string(),
                    description: "Review count test".to_string(),
                    base_difficulty: 2,
                    parent_kata_id: None,
                    variation_params: None,
                },
                Utc::now(),
            )
            .unwrap();

        // create 3 sessions today
        for _ in 0..3 {
            let session = NewSession {
                kata_id,
                started_at: Utc::now(),
                completed_at: Some(Utc::now()),
                test_results_json: None,
                num_passed: None,
                num_failed: None,
                num_skipped: None,
                duration_ms: None,
                quality_rating: Some(2),
            };
            repo.create_session(&session).unwrap();
        }

        // create 1 session yesterday (should not count)
        let yesterday = Utc::now() - chrono::Duration::days(1);
        let session = NewSession {
            kata_id,
            started_at: yesterday,
            completed_at: Some(yesterday),
            test_results_json: None,
            num_passed: None,
            num_failed: None,
            num_skipped: None,
            duration_ms: None,
            quality_rating: Some(2),
        };
        repo.create_session(&session).unwrap();

        let count = repo.get_reviews_count_today().unwrap();
        assert_eq!(count, 3);
    }

    #[test]
    fn test_next_scheduled_review_none_when_no_katas() {
        let repo = setup_test_repo();
        assert!(repo.get_next_scheduled_review().unwrap().is_none());
    }

    #[test]
    fn test_next_scheduled_review_returns_earliest_future() {
        let repo = setup_test_repo();
        let kata_a = repo
            .create_kata(
                &NewKata {
                    name: "future_a".to_string(),
                    category: "test".to_string(),
                    description: "future_a".to_string(),
                    base_difficulty: 2,
                    parent_kata_id: None,
                    variation_params: None,
                },
                Utc::now(),
            )
            .unwrap();

        let kata_b = repo
            .create_kata(
                &NewKata {
                    name: "future_b".to_string(),
                    category: "test".to_string(),
                    description: "future_b".to_string(),
                    base_difficulty: 3,
                    parent_kata_id: None,
                    variation_params: None,
                },
                Utc::now(),
            )
            .unwrap();

        let now = Utc::now();
        let soon = now + chrono::Duration::hours(8);
        let later = now + chrono::Duration::hours(30);

        repo.conn
            .execute(
                "UPDATE katas SET next_review_at = ?1 WHERE id = ?2",
                params![soon.timestamp(), kata_a],
            )
            .unwrap();

        repo.conn
            .execute(
                "UPDATE katas SET next_review_at = ?1 WHERE id = ?2",
                params![later.timestamp(), kata_b],
            )
            .unwrap();

        let next = repo.get_next_scheduled_review().unwrap().unwrap();
        assert!((next - soon).num_seconds().abs() < 5);
    }

    #[test]
    fn test_get_success_rate_last_n_days_no_sessions() {
        let repo = setup_test_repo();
        let rate = repo.get_success_rate_last_n_days(7).unwrap();
        assert_eq!(rate, 0.0);
    }

    #[test]
    fn test_get_success_rate_last_n_days_with_sessions() {
        let repo = setup_test_repo();
        let kata_id = repo
            .create_kata(
                &NewKata {
                    name: "success_rate_kata".to_string(),
                    category: "test".to_string(),
                    description: "Success rate test".to_string(),
                    base_difficulty: 2,
                    parent_kata_id: None,
                    variation_params: None,
                },
                Utc::now(),
            )
            .unwrap();

        // create sessions with different quality ratings
        // 2 successful (rating >= 2), 2 failed (rating < 2)
        for rating in [2, 3, 0, 1] {
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

        let rate = repo.get_success_rate_last_n_days(7).unwrap();
        assert_eq!(rate, 0.5);
    }

    #[test]
    fn test_get_success_rate_last_n_days_filters_by_date() {
        let repo = setup_test_repo();
        let kata_id = repo
            .create_kata(
                &NewKata {
                    name: "date_filter_kata".to_string(),
                    category: "test".to_string(),
                    description: "Date filter test".to_string(),
                    base_difficulty: 2,
                    parent_kata_id: None,
                    variation_params: None,
                },
                Utc::now(),
            )
            .unwrap();

        // create 1 successful session today
        let session = NewSession {
            kata_id,
            started_at: Utc::now(),
            completed_at: Some(Utc::now()),
            test_results_json: None,
            num_passed: None,
            num_failed: None,
            num_skipped: None,
            duration_ms: None,
            quality_rating: Some(2),
        };
        repo.create_session(&session).unwrap();

        // create 1 failed session 10 days ago (should not count for 7-day window)
        let ten_days_ago = Utc::now() - chrono::Duration::days(10);
        let session = NewSession {
            kata_id,
            started_at: ten_days_ago,
            completed_at: Some(ten_days_ago),
            test_results_json: None,
            num_passed: None,
            num_failed: None,
            num_skipped: None,
            duration_ms: None,
            quality_rating: Some(0),
        };
        repo.create_session(&session).unwrap();

        let rate = repo.get_success_rate_last_n_days(7).unwrap();
        assert_eq!(rate, 1.0);
    }

    #[test]
    fn test_delete_all_data() {
        let repo = setup_test_repo();

        // Create some test data
        let kata_id = repo
            .create_kata(
                &NewKata {
                    name: "test_kata".to_string(),
                    category: "test".to_string(),
                    description: "Test".to_string(),
                    base_difficulty: 2,
                    parent_kata_id: None,
                    variation_params: None,
                },
                Utc::now(),
            )
            .unwrap();

        let kata_id2 = repo
            .create_kata(
                &NewKata {
                    name: "test_kata2".to_string(),
                    category: "test".to_string(),
                    description: "Test 2".to_string(),
                    base_difficulty: 3,
                    parent_kata_id: None,
                    variation_params: None,
                },
                Utc::now(),
            )
            .unwrap();

        // Add dependency
        repo.add_dependency(kata_id2, kata_id, 1).unwrap();

        // Add session
        let session = NewSession {
            kata_id,
            started_at: Utc::now(),
            completed_at: Some(Utc::now()),
            test_results_json: None,
            num_passed: Some(5),
            num_failed: Some(0),
            num_skipped: Some(0),
            duration_ms: Some(1000),
            quality_rating: Some(2),
        };
        repo.create_session(&session).unwrap();

        // Add daily stats
        repo.upsert_daily_stats(&DailyStats {
            date: "2025-01-01".to_string(),
            total_reviews: 1,
            total_successes: 1,
            success_rate: 1.0,
            streak_days: 1,
            categories_json: "{}".to_string(),
        })
        .unwrap();

        // Verify data exists
        let katas = repo.get_all_katas().unwrap();
        assert_eq!(katas.len(), 2);

        let sessions = repo.get_recent_sessions(kata_id, 10).unwrap();
        assert_eq!(sessions.len(), 1);

        let deps = repo.load_dependency_graph().unwrap();
        assert!(!deps.is_unlocked(kata_id2, &HashMap::new()));

        let stats = repo.get_daily_stats("2025-01-01").unwrap();
        assert!(stats.is_some());

        // Perform full reset
        repo.delete_all_data().unwrap();

        // Verify all data is deleted
        let katas = repo.get_all_katas().unwrap();
        assert_eq!(katas.len(), 0);

        let sessions = repo.get_recent_sessions(kata_id, 10).unwrap();
        assert_eq!(sessions.len(), 0);

        let deps = repo.load_dependency_graph().unwrap();
        assert!(deps.is_unlocked(kata_id2, &HashMap::new())); // No dependencies means unlocked

        let stats = repo.get_daily_stats("2025-01-01").unwrap();
        assert!(stats.is_none());

        let db_stats = repo.get_database_stats().unwrap();
        assert_eq!(db_stats.katas_total, 0);
        assert_eq!(db_stats.sessions_total, 0);
        assert_eq!(db_stats.dependencies_count, 0);
    }
}
