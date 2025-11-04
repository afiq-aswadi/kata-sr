//! Kata Spaced Repetition CLI
//!
//! This is the main entry point for the kata spaced repetition system.
//! It handles:
//!
//! - Python environment bootstrap
//! - Database initialization with migrations
//! - Configuration and setup verification
//!
//! The CLI ensures all prerequisites are met before handing off to the TUI
//! (Terminal User Interface) which will be implemented by Agent 3.

use anyhow::{Context, Result};
use kata_sr::db::repo::KataRepository;
use kata_sr::python_env::PythonEnv;
use std::path::PathBuf;

/// Returns the database path, creating parent directories if needed.
///
/// The database is stored at `~/.local/share/kata-sr/kata.db`.
/// This function expands the home directory and creates any missing
/// parent directories.
///
/// # Errors
///
/// Returns an error if:
/// - Home directory cannot be determined
/// - Parent directories cannot be created
fn get_db_path() -> Result<PathBuf> {
    let home = std::env::var("HOME").context("HOME environment variable not set")?;

    let db_dir = PathBuf::from(home)
        .join(".local")
        .join("share")
        .join("kata-sr");

    std::fs::create_dir_all(&db_dir)
        .context("Failed to create database directory at ~/.local/share/kata-sr")?;

    Ok(db_dir.join("kata.db"))
}

fn main() -> Result<()> {
    println!("Kata Spaced Repetition - Backend Initialization");
    println!("================================================\n");

    // setup Python environment
    println!("Step 1: Checking Python environment...");
    let python_env = PythonEnv::setup()
        .map_err(|e| anyhow::anyhow!("Failed to setup Python environment: {}", e))?;
    println!(
        "  ✓ Python interpreter: {}\n",
        python_env.interpreter_path().display()
    );

    // initialize database
    println!("Step 2: Initializing database...");
    let db_path = get_db_path()?;
    println!("  Database location: {}", db_path.display());

    let repo = KataRepository::new(&db_path).context("Failed to open database connection")?;

    repo.run_migrations()
        .context("Failed to run database migrations")?;

    println!("  ✓ Database initialized with schema\n");

    // verify setup
    println!("Step 3: Verifying setup...");
    let kata_count = repo.get_all_katas().context("Failed to query katas")?.len();
    println!("  Current kata count: {}", kata_count);

    let due_count = repo
        .get_katas_due(chrono::Utc::now())
        .context("Failed to query due katas")?
        .len();
    println!("  Katas due for review: {}\n", due_count);

    // success message with next steps
    println!("✓ Backend initialization complete!\n");
    println!("Next steps:");
    println!("  - Database ready at: {}", db_path.display());
    println!("  - Python environment ready at: katas/.venv/");
    println!("  - Run tests with: cargo test");
    println!("  - Generate docs with: cargo doc --open");
    println!("\nNote: TUI interface will be implemented by Agent 3.");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_db_path_structure() {
        let result = get_db_path();
        if let Ok(path) = result {
            // verify path ends with kata.db
            assert_eq!(path.file_name().unwrap(), "kata.db");

            // verify path contains .local/share/kata-sr
            let path_str = path.to_string_lossy();
            assert!(path_str.contains(".local/share/kata-sr"));
        }
    }
}
