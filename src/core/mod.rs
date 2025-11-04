//! Core logic for the kata spaced repetition system.
//!
//! This module provides the fundamental algorithms and data structures:
//! - SM-2 spaced repetition scheduling
//! - Adaptive difficulty tracking based on performance
//! - Dependency graph management for prerequisite relationships

pub mod dependencies;
pub mod difficulty;
pub mod scheduler;
