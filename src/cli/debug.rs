//! Debug command handlers
//!
//! Commands for development and testing workflows:
//! - Reset SM-2 states
//! - Clear session history
//! - Reimport katas from manifests
//! - Database inspection

use crate::core::kata_loader;
use crate::core::workbook::{generate_workbook_html, load_workbooks};
use crate::db::repo::KataRepository;
use anyhow::{bail, Context, Result};
use clap::Subcommand;
use serde_json;

#[derive(Subcommand)]
pub enum DebugOperation {
    /// Perform a complete database reset (deletes all katas, sessions, dependencies, and stats)
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

    /// List all problematic katas (JSON output)
    ListProblematic,

    /// Unflag a kata as no longer problematic
    Unflag {
        /// ID of the kata to unflag
        kata_id: i64,
    },

    /// Export session history for a kata
    ExportSessions {
        /// Name of the kata to export sessions for
        kata_name: String,
        /// Output format (json or csv)
        #[arg(long, default_value = "json")]
        format: String,
        /// Output file path (optional, defaults to stdout)
        #[arg(long)]
        output: Option<String>,
    },

    /// Generate workbook HTML files from manifests
    GenerateWorkbook {
        /// Workbook ID to generate (when omitted, generate all)
        #[arg(long)]
        topic: Option<String>,
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
            DebugOperation::ListProblematic => list_problematic(repo),
            DebugOperation::Unflag { kata_id } => unflag_kata(repo, *kata_id),
            DebugOperation::ExportSessions {
                kata_name,
                format,
                output,
            } => export_sessions(repo, kata_name, format, output.as_deref()),
            DebugOperation::GenerateWorkbook { topic } => generate_workbook_pages(topic.as_deref()),
        }
    }
}

fn reset_all(repo: &KataRepository) -> Result<()> {
    repo.delete_all_data()
        .context("Failed to delete all data")?;
    println!("✓ Performed full database reset");
    println!("  - Cleared all sessions");
    println!("  - Cleared all dependencies");
    println!("  - Removed all katas");
    println!("  - Cleared all daily stats");
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

    repo.delete_kata(kata.id).context("Failed to delete kata")?;

    println!("✓ Deleted kata: {}", kata_name);
    Ok(())
}

fn show_stats(repo: &KataRepository, json: bool) -> Result<()> {
    let stats = repo
        .get_database_stats()
        .context("Failed to get database stats")?;

    if json {
        let json_output =
            serde_json::to_string_pretty(&stats).context("Failed to serialize stats to JSON")?;
        println!("{}", json_output);
    } else {
        println!("Database Statistics");
        println!("===================");
        println!(
            "Katas:        {} total ({} due, {} scheduled)",
            stats.katas_total, stats.katas_due, stats.katas_scheduled
        );
        println!(
            "Sessions:     {} total ({} passed, {} failed)",
            stats.sessions_total, stats.sessions_passed, stats.sessions_failed
        );
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
        repo.get_all_katas().context("Failed to get all katas")?
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
        println!(
            "    SM-2: ease={:.2}, interval={} days, reps={}",
            kata.current_ease_factor, kata.current_interval_days, kata.current_repetition_count
        );
        println!();
    }

    Ok(())
}

fn list_problematic(repo: &KataRepository) -> Result<()> {
    use serde::Serialize;

    let katas = repo
        .get_problematic_katas()
        .context("Failed to get problematic katas")?;

    #[derive(Serialize)]
    struct ProblematicKata {
        id: i64,
        name: String,
        directory: String,
        notes: Option<String>,
        flagged_at: String,
    }

    let output: Vec<ProblematicKata> = katas
        .iter()
        .map(|kata| {
            let directory = format!("katas/exercises/{}", kata.name);
            let flagged_at = kata
                .flagged_at
                .map(|dt| dt.to_rfc3339())
                .unwrap_or_else(|| "unknown".to_string());

            ProblematicKata {
                id: kata.id,
                name: kata.name.clone(),
                directory,
                notes: kata.problematic_notes.clone(),
                flagged_at,
            }
        })
        .collect();

    let json_output = serde_json::to_string_pretty(&output)
        .context("Failed to serialize problematic katas to JSON")?;
    println!("{}", json_output);

    Ok(())
}

fn unflag_kata(repo: &KataRepository, kata_id: i64) -> Result<()> {
    // Verify kata exists
    let kata = repo
        .get_kata_by_id(kata_id)
        .context("Failed to look up kata")?
        .ok_or_else(|| anyhow::anyhow!("Kata with ID {} not found", kata_id))?;

    if !kata.is_problematic {
        println!(
            "⚠ Kata '{}' (ID {}) is not currently flagged as problematic",
            kata.name, kata_id
        );
        return Ok(());
    }

    repo.unflag_kata(kata_id).context("Failed to unflag kata")?;

    println!("✓ Unflagged kata: {} (ID {})", kata.name, kata_id);
    Ok(())
}

fn generate_workbook_pages(topic: Option<&str>) -> Result<()> {
    let workbooks = load_workbooks().context("Failed to load workbooks")?;
    if workbooks.is_empty() {
        bail!("No workbooks found");
    }

    let mut generated = 0usize;
    for workbook in workbooks {
        if let Some(topic_id) = topic {
            if workbook.meta.id != topic_id {
                continue;
            }
        }

        generate_workbook_html(&workbook).with_context(|| {
            format!("Failed to generate HTML for workbook {}", workbook.meta.id)
        })?;
        println!(
            "✓ Generated {} → {}",
            workbook.meta.id,
            workbook.html_path.display()
        );
        generated += 1;
    }

    if generated == 0 {
        bail!("No workbook matched the requested topic");
    }

    Ok(())
}

fn export_sessions(
    repo: &KataRepository,
    kata_name: &str,
    format: &str,
    output_path: Option<&str>,
) -> Result<()> {
    // Get the kata
    let kata = repo
        .get_kata_by_name(kata_name)
        .context("Failed to look up kata")?
        .ok_or_else(|| anyhow::anyhow!("Kata not found: {}", kata_name))?;

    // Get all sessions for this kata
    let sessions = repo
        .get_all_sessions_for_kata(kata.id)
        .context("Failed to get sessions")?;

    if sessions.is_empty() {
        println!("No sessions found for kata: {}", kata_name);
        return Ok(());
    }

    // Generate output based on format
    let output = match format.to_lowercase().as_str() {
        "json" => export_sessions_json(&kata, &sessions)?,
        "csv" => export_sessions_csv(&kata, &sessions)?,
        _ => {
            return Err(anyhow::anyhow!(
                "Unsupported format: {}. Use 'json' or 'csv'",
                format
            ))
        }
    };

    // Write to file or stdout
    if let Some(path) = output_path {
        std::fs::write(path, output).context("Failed to write output file")?;
        println!("✓ Exported {} sessions to {}", sessions.len(), path);
    } else {
        println!("{}", output);
    }

    Ok(())
}

fn export_sessions_json(
    kata: &crate::db::repo::Kata,
    sessions: &[crate::db::repo::Session],
) -> Result<String> {
    use serde::Serialize;

    #[derive(Serialize)]
    struct SessionExport {
        session_id: i64,
        kata_id: i64,
        kata_name: String,
        started_at: String,
        completed_at: Option<String>,
        duration_ms: Option<i64>,
        num_passed: Option<i32>,
        num_failed: Option<i32>,
        num_skipped: Option<i32>,
        quality_rating: Option<i32>,
        quality_rating_label: Option<String>,
    }

    let exports: Vec<SessionExport> = sessions
        .iter()
        .map(|s| {
            let quality_label = s.quality_rating.map(|r| match r {
                1 => "Again".to_string(),
                2 => "Hard".to_string(),
                3 => "Good".to_string(),
                4 => "Easy".to_string(),
                _ => format!("{}", r),
            });

            SessionExport {
                session_id: s.id,
                kata_id: kata.id,
                kata_name: kata.name.clone(),
                started_at: s.started_at.format("%Y-%m-%d %H:%M:%S").to_string(),
                completed_at: s
                    .completed_at
                    .map(|dt| dt.format("%Y-%m-%d %H:%M:%S").to_string()),
                duration_ms: s.duration_ms,
                num_passed: s.num_passed,
                num_failed: s.num_failed,
                num_skipped: s.num_skipped,
                quality_rating: s.quality_rating,
                quality_rating_label: quality_label,
            }
        })
        .collect();

    serde_json::to_string_pretty(&exports).context("Failed to serialize sessions to JSON")
}

fn export_sessions_csv(
    kata: &crate::db::repo::Kata,
    sessions: &[crate::db::repo::Session],
) -> Result<String> {
    let mut output = String::new();

    // Header
    output.push_str("date,kata,passed,failed,skipped,duration_ms,rating,rating_label\n");

    // Rows
    for session in sessions {
        let date = if let Some(completed_at) = session.completed_at {
            completed_at.format("%Y-%m-%d %H:%M:%S").to_string()
        } else {
            session.started_at.format("%Y-%m-%d %H:%M:%S").to_string()
        };

        let passed = session
            .num_passed
            .map_or("-".to_string(), |n| n.to_string());
        let failed = session
            .num_failed
            .map_or("-".to_string(), |n| n.to_string());
        let skipped = session
            .num_skipped
            .map_or("-".to_string(), |n| n.to_string());
        let duration = session
            .duration_ms
            .map_or("-".to_string(), |ms| ms.to_string());
        let rating = session
            .quality_rating
            .map_or("-".to_string(), |r| r.to_string());
        let rating_label = session.quality_rating.map_or("-".to_string(), |r| match r {
            1 => "Again".to_string(),
            2 => "Hard".to_string(),
            3 => "Good".to_string(),
            4 => "Easy".to_string(),
            _ => format!("{}", r),
        });

        output.push_str(&format!(
            "{},{},{},{},{},{},{},{}\n",
            date, kata.name, passed, failed, skipped, duration, rating, rating_label
        ));
    }

    Ok(output)
}
