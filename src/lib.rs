//! Kata Spaced Repetition Library
//!
//! This library provides the core infrastructure for a spaced repetition system
//! for coding exercises (katas). It includes:
//!
//! - **SM-2 Scheduling**: Anki-style spaced repetition algorithm for determining
//!   when katas should be reviewed next
//! - **Adaptive Difficulty**: Track user performance and adjust kata difficulty
//!   independent of scheduling
//! - **Dependency Management**: Prerequisites and unlock logic for structured
//!   learning paths
//! - **Database Layer**: SQLite-based persistence with migrations and repository
//!   pattern for all data access
//!
//! # Examples
//!
//! Basic usage:
//!
//! ```no_run
//! use kata_sr::db::repo::{KataRepository, NewKata};
//! use kata_sr::core::scheduler::{SM2State, QualityRating};
//! use chrono::Utc;
//!
//! // Setup database
//! let repo = KataRepository::new("kata.db")?;
//! repo.run_migrations()?;
//!
//! // Create a kata
//! let new_kata = NewKata {
//!     name: "multi_head_attention".to_string(),
//!     category: "transformers".to_string(),
//!     description: "Implement multi-head attention".to_string(),
//!     base_difficulty: 4,
//!     parent_kata_id: None,
//!     variation_params: None,
//! };
//! let kata_id = repo.create_kata(&new_kata, Utc::now())?;
//!
//! // Get katas due for review
//! let due_katas = repo.get_katas_due(Utc::now())?;
//!
//! // Update SM-2 state after review
//! let mut state = SM2State::new();
//! let interval = state.update(QualityRating::Good);
//! let next_review = Utc::now() + chrono::Duration::days(interval);
//! repo.update_kata_after_review(kata_id, &state, next_review, Utc::now())?;
//!
//! # Ok::<(), rusqlite::Error>(())
//! ```

pub mod core;
pub mod db;
pub mod python_env;
