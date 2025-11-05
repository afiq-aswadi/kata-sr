//! Kata Spaced Repetition CLI
//!
//! This is the main entry point for the kata spaced repetition system.
//! It handles:
//!
//! - CLI argument parsing (debug commands or launch TUI)
//! - Python environment bootstrap
//! - Database initialization with migrations
//! - TUI application launch
//!
//! The CLI ensures all prerequisites are met before launching the TUI
//! (Terminal User Interface) for kata practice and review.

use anyhow::{Context, Result};
use clap::Parser;
use kata_sr::cli::{Cli, Command};
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
    // parse CLI arguments
    let cli = Cli::parse();

    // setup Python environment (required for TUI, optional for debug commands)
    let python_env = match &cli.command {
        Some(Command::Debug { .. }) => {
            // debug commands don't need Python environment
            None
        }
        None => {
            // TUI needs Python environment
            Some(PythonEnv::setup()
                .map_err(|e| anyhow::anyhow!("Failed to setup Python environment: {}", e))?)
        }
    };

    if let Some(ref env) = python_env {
        configure_python_env_vars(env)?;
    }

    // initialize database
    let db_path = if let Some(ref path) = cli.db_path {
        PathBuf::from(path)
    } else {
        get_db_path()?
    };

    let repo = KataRepository::new(&db_path).context("Failed to open database connection")?;

    repo.run_migrations()
        .context("Failed to run database migrations")?;

    // route to appropriate command
    match cli.command {
        Some(Command::Debug { operation }) => {
            operation.execute(&repo)?;
        }
        None => {
            // launch TUI application (default behavior)
            let mut app = App::new(repo)?;
            app.run()?;
        }
    }

    Ok(())
}

fn configure_python_env_vars(env: &PythonEnv) -> Result<()> {
    let interpreter = env.interpreter_path();
    std::env::set_var("KATA_SR_PYTHON", interpreter);

    if let Some(katas_dir) = interpreter
        .parent()
        .and_then(|p| p.parent())
        .and_then(|p| p.parent())
    {
        let katas_dir = katas_dir.to_path_buf();
        let canonical = katas_dir.canonicalize().unwrap_or(katas_dir);
        std::env::set_var("KATA_SR_KATAS_DIR", canonical);
    } else {
        anyhow::bail!(
            "Unable to determine katas directory from interpreter path: {}",
            interpreter.display()
        );
    }

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
