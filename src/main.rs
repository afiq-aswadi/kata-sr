//! Kata Spaced Repetition CLI
//!
//! This is the main entry point for the kata spaced repetition system.
//! It handles:
//!
//! - Python environment bootstrap
//! - Database initialization with migrations
//! - TUI application launch
//!
//! The CLI ensures all prerequisites are met before launching the TUI
//! (Terminal User Interface) for kata practice and review.

use anyhow::{Context, Result};
use kata_sr::db::repo::KataRepository;
use kata_sr::python_env::PythonEnv;
use kata_sr::tui::app::App;
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
    // setup Python environment
    let _python_env = PythonEnv::setup()
        .map_err(|e| anyhow::anyhow!("Failed to setup Python environment: {}", e))?;

    // initialize database
    let db_path = get_db_path()?;
    let repo = KataRepository::new(&db_path).context("Failed to open database connection")?;

    repo.run_migrations()
        .context("Failed to run database migrations")?;

    // launch TUI application
    let mut app = App::new(repo)?;
    app.run()?;

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
