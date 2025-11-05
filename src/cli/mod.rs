//! CLI command handling
//!
//! This module contains all CLI subcommand implementations.
//! Currently supports debug commands for development and testing.

pub mod debug;

use clap::{Parser, Subcommand};

/// Kata Spaced Repetition CLI
#[derive(Parser)]
#[command(name = "kata-sr")]
#[command(about = "TUI tool for practicing coding patterns with spaced repetition")]
pub struct Cli {
    /// Path to the database file (for testing)
    #[arg(long, value_name = "PATH")]
    pub db_path: Option<String>,

    /// Subcommand to execute (if none, launch TUI)
    #[command(subcommand)]
    pub command: Option<Command>,
}

#[derive(Subcommand)]
pub enum Command {
    /// Debug commands for development and testing
    Debug {
        #[command(subcommand)]
        operation: debug::DebugOperation,
    },
}
