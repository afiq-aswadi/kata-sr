//! Kata Spaced Repetition Library
//!
//! This library provides the core infrastructure for a spaced repetition system
//! for coding exercises (katas). It includes:
//!
//! - **FSRS-5 Scheduling**: Modern spaced repetition algorithm for determining
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
//! use kata_sr::core::fsrs::{FsrsCard, FsrsParams, Rating};
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
//! // Update FSRS state after review
//! let mut card = FsrsCard::new();
//! let params = FsrsParams::default();
//! card.schedule(Rating::Good, &params, Utc::now());
//! let next_review = Utc::now() + chrono::Duration::days(card.scheduled_days as i64);
//! repo.update_kata_after_fsrs_review(kata_id, &card, next_review, Utc::now())?;
//!
//! # Ok::<(), rusqlite::Error>(())
//! ```

pub mod cli;
pub mod config;
pub mod core;
pub mod db;
pub mod python_env;
pub mod runner;
pub mod tui;
