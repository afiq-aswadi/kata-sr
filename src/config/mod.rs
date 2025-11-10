//! Configuration management for kata-sr.
//!
//! This module provides configuration file support (~/.config/kata-sr/config.toml)
//! and runtime settings management.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;

/// Main application configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    /// Editor configuration
    #[serde(default)]
    pub editor: EditorConfig,

    /// File paths configuration
    #[serde(default)]
    pub paths: PathsConfig,

    /// Display settings
    #[serde(default)]
    pub display: DisplayConfig,

    /// Review behavior settings
    #[serde(default)]
    pub review: ReviewConfig,

    /// Library view settings
    #[serde(default)]
    pub library: LibraryConfig,
}

/// Editor configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EditorConfig {
    /// Editor command to use (e.g., "nvim", "vim", "code")
    pub command: String,

    /// Additional arguments to pass to the editor
    #[serde(default)]
    pub args: Vec<String>,
}

/// File paths configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathsConfig {
    /// Database file path
    pub database: String,

    /// Template directory for practice files
    pub templates: String,
}

/// Display settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DisplayConfig {
    /// Color theme (currently only "default" supported)
    pub theme: String,

    /// Number of days to show in heatmap
    pub heatmap_days: usize,

    /// Date format string (strftime format)
    pub date_format: String,
}

/// Review behavior settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReviewConfig {
    /// Daily review limit (None = unlimited)
    pub daily_limit: Option<usize>,

    /// Default rating when skipping manual selection (1-4: Again/Hard/Good/Easy)
    pub default_rating: u8,

    /// Whether to persist sort/filter preferences between sessions
    pub persist_sort_mode: bool,
}

/// Library view settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LibraryConfig {
    /// Default sort mode
    pub default_sort: String,

    /// Default sort direction (true = ascending)
    pub default_sort_ascending: bool,
}

impl Default for EditorConfig {
    fn default() -> Self {
        Self {
            // Try EDITOR env var, fallback to nvim
            command: std::env::var("EDITOR").unwrap_or_else(|_| "nvim".to_string()),
            args: Vec::new(),
        }
    }
}

impl Default for PathsConfig {
    fn default() -> Self {
        Self {
            database: "~/.local/share/kata-sr/kata.db".to_string(),
            templates: "/tmp/kata_practice".to_string(),
        }
    }
}

impl Default for DisplayConfig {
    fn default() -> Self {
        Self {
            theme: "default".to_string(),
            heatmap_days: 90,
            date_format: "%Y-%m-%d".to_string(),
        }
    }
}

impl Default for ReviewConfig {
    fn default() -> Self {
        Self {
            daily_limit: None,
            default_rating: 3, // Good
            persist_sort_mode: true,
        }
    }
}

impl Default for LibraryConfig {
    fn default() -> Self {
        Self {
            default_sort: "Name".to_string(),
            default_sort_ascending: true,
        }
    }
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            editor: EditorConfig::default(),
            paths: PathsConfig::default(),
            display: DisplayConfig::default(),
            review: ReviewConfig::default(),
            library: LibraryConfig::default(),
        }
    }
}

impl AppConfig {
    /// Returns the default config file path: ~/.config/kata-sr/config.toml
    pub fn config_path() -> Result<PathBuf> {
        let home = std::env::var("HOME")
            .or_else(|_| std::env::var("USERPROFILE"))
            .context("Could not determine home directory")?;

        Ok(PathBuf::from(home)
            .join(".config")
            .join("kata-sr")
            .join("config.toml"))
    }

    /// Loads configuration from the config file, falling back to defaults if not found.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use kata_sr::config::AppConfig;
    /// let config = AppConfig::load().unwrap();
    /// println!("Editor: {}", config.editor.command);
    /// ```
    pub fn load() -> Result<Self> {
        let config_path = Self::config_path()?;

        if !config_path.exists() {
            // No config file, use defaults
            return Ok(Self::default());
        }

        let contents = fs::read_to_string(&config_path)
            .with_context(|| format!("Failed to read config file: {:?}", config_path))?;

        let config: AppConfig = toml::from_str(&contents)
            .with_context(|| format!("Failed to parse config file: {:?}", config_path))?;

        Ok(config)
    }

    /// Saves the configuration to the config file.
    ///
    /// Creates the config directory if it doesn't exist.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use kata_sr::config::AppConfig;
    /// let mut config = AppConfig::load().unwrap();
    /// config.editor.command = "vim".to_string();
    /// config.save().unwrap();
    /// ```
    pub fn save(&self) -> Result<()> {
        let config_path = Self::config_path()?;

        // Create config directory if it doesn't exist
        if let Some(parent) = config_path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("Failed to create config directory: {:?}", parent))?;
        }

        let contents = toml::to_string_pretty(self)
            .context("Failed to serialize config to TOML")?;

        fs::write(&config_path, contents)
            .with_context(|| format!("Failed to write config file: {:?}", config_path))?;

        Ok(())
    }

    /// Expands ~ in paths to the user's home directory.
    pub fn expand_path(&self, path: &str) -> Result<PathBuf> {
        if path.starts_with("~/") {
            let home = std::env::var("HOME")
                .or_else(|_| std::env::var("USERPROFILE"))
                .context("Could not determine home directory")?;
            Ok(PathBuf::from(home).join(&path[2..]))
        } else if path.starts_with('~') {
            let home = std::env::var("HOME")
                .or_else(|_| std::env::var("USERPROFILE"))
                .context("Could not determine home directory")?;
            Ok(PathBuf::from(home).join(&path[1..]))
        } else {
            Ok(PathBuf::from(path))
        }
    }

    /// Returns the expanded database path.
    pub fn database_path(&self) -> Result<PathBuf> {
        self.expand_path(&self.paths.database)
    }

    /// Returns the expanded template directory path.
    pub fn template_dir(&self) -> Result<PathBuf> {
        self.expand_path(&self.paths.templates)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = AppConfig::default();
        assert!(config.editor.command.len() > 0);
        assert_eq!(config.display.heatmap_days, 90);
        assert_eq!(config.review.default_rating, 3);
        assert!(config.review.persist_sort_mode);
    }

    #[test]
    fn test_expand_path() {
        let config = AppConfig::default();

        // Test tilde expansion
        if let Ok(expanded) = config.expand_path("~/test") {
            assert!(!expanded.to_string_lossy().contains('~'));
        }

        // Test absolute path (no expansion)
        let abs_path = config.expand_path("/tmp/test").unwrap();
        assert_eq!(abs_path, PathBuf::from("/tmp/test"));
    }

    #[test]
    fn test_serialization() {
        let config = AppConfig::default();
        let toml_str = toml::to_string(&config).unwrap();
        assert!(toml_str.contains("[editor]"));
        assert!(toml_str.contains("[paths]"));
        assert!(toml_str.contains("[display]"));
        assert!(toml_str.contains("[review]"));
    }

    #[test]
    fn test_deserialization() {
        let toml_str = r#"
            [editor]
            command = "vim"
            args = ["+10", "-c", "startinsert"]

            [paths]
            database = "~/.local/share/kata-sr/custom.db"
            templates = "/tmp/kata_custom"

            [display]
            theme = "dark"
            heatmap_days = 120
            date_format = "%d-%m-%Y"

            [review]
            daily_limit = 50
            default_rating = 2
            persist_sort_mode = false

            [library]
            default_sort = "Difficulty"
            default_sort_ascending = false
        "#;

        let config: AppConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.editor.command, "vim");
        assert_eq!(config.editor.args, vec!["+10", "-c", "startinsert"]);
        assert_eq!(config.display.heatmap_days, 120);
        assert_eq!(config.review.daily_limit, Some(50));
        assert_eq!(config.review.default_rating, 2);
        assert!(!config.review.persist_sort_mode);
    }
}
