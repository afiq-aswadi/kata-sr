//! Debug command handlers
//!
//! Commands for development and testing workflows:
//! - Reset SM-2 states
//! - Clear session history
//! - Reimport katas from manifests
//! - Database inspection

use crate::core::kata_loader;
use crate::db::repo::KataRepository;
use anyhow::{Context, Result};
use clap::Subcommand;
use serde_json;

#[derive(Subcommand)]
pub enum DebugOperation {
    /// Reset all kata SM-2 states to initial values
    ResetAll,

    /// Reset specific kata SM-2 state
    Reset {
        /// Name of the kata to reset
        kata_name: String,
    },

    /// Make a kata due immediately (preserves SM-2 state)
    ForceDue {
        /// Name of the kata to make due
        kata_name: String,
    },

    /// Clear all session history
    ClearSessions,

    /// Clear daily statistics
    ClearStats,

    /// Reimport katas from manifest files
    Reimport {
        /// Delete katas not found in exercises/ directory
        #[arg(long)]
        prune: bool,
    },

    /// Delete a kata and all its data
    Delete {
        /// Name of the kata to delete
        kata_name: String,
    },

    /// Show database statistics
    Stats {
        /// Output as JSON
        #[arg(long)]
        json: bool,
    },

    /// List all katas
    List {
        /// Show only due katas
        #[arg(long)]
        due: bool,
    },
}

impl DebugOperation {
    pub fn execute(&self, repo: &KataRepository) -> Result<()> {
        match self {
            DebugOperation::ResetAll => reset_all(repo),
            DebugOperation::Reset { kata_name } => reset_kata(repo, kata_name),
            DebugOperation::ForceDue { kata_name } => force_due(repo, kata_name),
            DebugOperation::ClearSessions => clear_sessions(repo),
            DebugOperation::ClearStats => clear_stats(repo),
            DebugOperation::Reimport { prune } => reimport(repo, *prune),
            DebugOperation::Delete { kata_name } => delete_kata(repo, kata_name),
            DebugOperation::Stats { json } => show_stats(repo, *json),
            DebugOperation::List { due } => list_katas(repo, *due),
        }
    }
}

fn reset_all(repo: &KataRepository) -> Result<()> {
    repo.reset_all_sm2_states()
        .context("Failed to reset all SM-2 states")?;
    println!("✓ Reset all kata SM-2 states to initial values");
    Ok(())
}

fn reset_kata(repo: &KataRepository, kata_name: &str) -> Result<()> {
    let kata = repo
        .get_kata_by_name(kata_name)
        .context("Failed to look up kata")?
        .ok_or_else(|| anyhow::anyhow!("Kata not found: {}", kata_name))?;

    repo.reset_kata_sm2_state(kata.id)
        .context("Failed to reset kata SM-2 state")?;

    println!("✓ Reset SM-2 state for kata: {}", kata_name);
    Ok(())
}

fn force_due(repo: &KataRepository, kata_name: &str) -> Result<()> {
    let kata = repo
        .get_kata_by_name(kata_name)
        .context("Failed to look up kata")?
        .ok_or_else(|| anyhow::anyhow!("Kata not found: {}", kata_name))?;

    repo.force_kata_due(kata.id)
        .context("Failed to force kata due")?;

    println!("✓ Forced kata due: {}", kata_name);
    Ok(())
}

fn clear_sessions(repo: &KataRepository) -> Result<()> {
    repo.clear_all_sessions()
        .context("Failed to clear sessions")?;
    println!("✓ Cleared all session history");
    Ok(())
}

fn clear_stats(repo: &KataRepository) -> Result<()> {
    repo.clear_daily_stats()
        .context("Failed to clear daily stats")?;
    println!("✓ Cleared all daily statistics");
    Ok(())
}

fn reimport(repo: &KataRepository, prune: bool) -> Result<()> {
    let result = kata_loader::reimport_katas(repo, prune)
        .context("Failed to reimport katas from manifests")?;

    println!("✓ Reimport complete:");
    println!("  Added:   {}", result.added);
    println!("  Updated: {}", result.updated);
    if prune {
        println!("  Deleted: {}", result.deleted);
    }

    Ok(())
}

fn delete_kata(repo: &KataRepository, kata_name: &str) -> Result<()> {
    let kata = repo
        .get_kata_by_name(kata_name)
        .context("Failed to look up kata")?
        .ok_or_else(|| anyhow::anyhow!("Kata not found: {}", kata_name))?;

    repo.delete_kata(kata.id)
        .context("Failed to delete kata")?;

    println!("✓ Deleted kata: {}", kata_name);
    Ok(())
}

fn show_stats(repo: &KataRepository, json: bool) -> Result<()> {
    let stats = repo
        .get_database_stats()
        .context("Failed to get database stats")?;

    if json {
        let json_output = serde_json::to_string_pretty(&stats)
            .context("Failed to serialize stats to JSON")?;
        println!("{}", json_output);
    } else {
        println!("Database Statistics");
        println!("===================");
        println!("Katas:        {} total ({} due, {} scheduled)",
            stats.katas_total, stats.katas_due, stats.katas_scheduled);
        println!("Sessions:     {} total ({} passed, {} failed)",
            stats.sessions_total, stats.sessions_passed, stats.sessions_failed);
        println!("Dependencies: {} edges", stats.dependencies_count);
        println!("Streak:       {} days", stats.current_streak);
    }

    Ok(())
}

fn list_katas(repo: &KataRepository, due: bool) -> Result<()> {
    let katas = if due {
        repo.get_katas_due(chrono::Utc::now())
            .context("Failed to get due katas")?
    } else {
        repo.get_all_katas()
            .context("Failed to get all katas")?
    };

    if katas.is_empty() {
        println!("No katas found");
        return Ok(());
    }

    println!("{} kata(s):", katas.len());
    println!();

    for kata in katas {
        let status = if kata.next_review_at.is_none() {
            "never reviewed".to_string()
        } else if kata.next_review_at.unwrap() <= chrono::Utc::now() {
            "due now".to_string()
        } else {
            format!("due {}", kata.next_review_at.unwrap().format("%Y-%m-%d"))
        };

        println!("  {} ({})", kata.name, status);
        println!("    Category: {}", kata.category);
        println!("    Difficulty: {:.1}/5", kata.current_difficulty);
        println!("    SM-2: ease={:.2}, interval={} days, reps={}",
            kata.current_ease_factor,
            kata.current_interval_days,
            kata.current_repetition_count);
        println!();
    }

    Ok(())
}
