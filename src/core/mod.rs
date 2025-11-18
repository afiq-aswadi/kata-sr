//! Core logic for the kata spaced repetition system.
//!
//! This module provides the fundamental algorithms and data structures:
//! - FSRS-5 spaced repetition scheduling
//! - Adaptive difficulty tracking based on performance
//! - Dependency graph management for prerequisite relationships
//! - Kata loading from manifest files
//! - Analytics and statistics aggregation

pub mod analytics;
pub mod course_loader;
pub mod dependencies;
pub mod difficulty;
pub mod fsrs;
pub mod fsrs_optimizer;
pub mod kata_generator;
pub mod kata_loader;
pub mod kata_validation;
