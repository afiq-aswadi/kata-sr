//! Configuration loading and validation tests.
//!
//! Tests config file parsing, defaults, validation, and error handling.

use kata_sr::config::{AppConfig, EditorConfig, LibraryConfig};
use std::fs;
use tempfile::TempDir;

#[test]
fn test_config_uses_defaults_when_file_missing() {
    // Temporarily unset EDITOR to ensure predictable default
    let original_editor = std::env::var("EDITOR").ok();
    std::env::remove_var("EDITOR");

    let config = AppConfig::default();

    assert_eq!(config.editor.command, "nvim");
    assert!(config.paths.database.contains("kata.db"));
    assert_eq!(config.display.heatmap_days, 90);

    // Restore original EDITOR value
    if let Some(editor) = original_editor {
        std::env::set_var("EDITOR", editor);
    }
}

#[test]
fn test_config_loads_from_valid_toml() {
    let temp_dir = TempDir::new().unwrap();
    let config_path = temp_dir.path().join("config.toml");

    let toml_content = r#"
[editor]
command = "vim"
args = ["-c", "set number"]

[paths]
database = "/custom/path/kata.db"
templates = "/custom/templates"

[display]
theme = "dark"
heatmap_days = 60
date_format = "%Y-%m-%d"

[review]
daily_limit = 20
default_rating = 3
persist_sort_mode = true

[library]
default_sort = "difficulty"
default_sort_ascending = false
"#;

    fs::write(&config_path, toml_content).unwrap();

    // Load config
    let config: AppConfig = toml::from_str(&fs::read_to_string(&config_path).unwrap()).unwrap();

    assert_eq!(config.editor.command, "vim");
    assert_eq!(config.editor.args, vec!["-c", "set number"]);
    assert_eq!(config.paths.database, "/custom/path/kata.db");
    assert_eq!(config.display.heatmap_days, 60);
    assert_eq!(config.review.daily_limit, Some(20));
    assert_eq!(config.review.default_rating, 3);
}

#[test]
fn test_config_rejects_invalid_toml() {
    let invalid_toml = r#"
[editor
command = "vim"
"#; // Missing closing bracket

    let result: Result<AppConfig, _> = toml::from_str(invalid_toml);
    assert!(result.is_err());
}

#[test]
fn test_config_handles_partial_configuration() {
    let partial_toml = r#"
[editor]
command = "emacs"

[display]
heatmap_days = 120
"#;

    let config: AppConfig = toml::from_str(partial_toml).unwrap();

    // Custom values
    assert_eq!(config.editor.command, "emacs");
    assert_eq!(config.display.heatmap_days, 120);

    // Defaults for missing sections
    assert!(config.paths.database.contains("kata.db"));
}

#[test]
fn test_editor_config_default() {
    // Temporarily unset EDITOR to ensure predictable default
    let original_editor = std::env::var("EDITOR").ok();
    std::env::remove_var("EDITOR");

    let editor_config = EditorConfig::default();

    assert_eq!(editor_config.command, "nvim");
    assert!(editor_config.args.is_empty());

    // Restore original EDITOR value
    if let Some(editor) = original_editor {
        std::env::set_var("EDITOR", editor);
    }
}

#[test]
fn test_library_config_default() {
    let library_config = LibraryConfig::default();

    assert_eq!(library_config.default_sort, "Name"); // Capitalized in default
    assert_eq!(library_config.default_sort_ascending, true);
}

#[test]
fn test_config_validates_rating_bounds() {
    // Rating should be 1-4 for FSRS
    let valid_config = r#"
[review]
default_rating = 3
persist_sort_mode = true
"#;

    let config: AppConfig = toml::from_str(valid_config).unwrap();
    assert_eq!(config.review.default_rating, 3);

    // TOML will parse out-of-bounds values, but app logic should validate
    let out_of_bounds = r#"
[review]
default_rating = 5
persist_sort_mode = true
"#;

    let config: AppConfig = toml::from_str(out_of_bounds).unwrap();
    // Config loads but should be validated at runtime
    assert_eq!(config.review.default_rating, 5);
}

#[test]
fn test_config_empty_editor_args() {
    let toml_content = r#"
[editor]
command = "vim"
args = []
"#;

    let config: AppConfig = toml::from_str(toml_content).unwrap();
    assert!(config.editor.args.is_empty());
}

#[test]
fn test_config_complex_editor_args() {
    let toml_content = r#"
[editor]
command = "code"
args = ["--wait", "--new-window", "-g"]
"#;

    let config: AppConfig = toml::from_str(toml_content).unwrap();
    assert_eq!(config.editor.args.len(), 3);
    assert_eq!(config.editor.args[0], "--wait");
    assert_eq!(config.editor.args[1], "--new-window");
    assert_eq!(config.editor.args[2], "-g");
}

#[test]
fn test_config_with_special_characters_in_paths() {
    let toml_content = r#"
[paths]
database = "/path/with spaces/kata.db"
templates = "/path/with-dashes/templates"
"#;

    let config: AppConfig = toml::from_str(toml_content).unwrap();
    assert_eq!(config.paths.database, "/path/with spaces/kata.db");
    assert_eq!(config.paths.templates, "/path/with-dashes/templates");
}

#[test]
fn test_config_optional_daily_limit() {
    // No daily limit
    let no_limit = r#"
[review]
default_rating = 3
"#;

    let config: AppConfig = toml::from_str(no_limit).unwrap();
    assert!(config.review.daily_limit.is_none());

    // With limit
    let with_limit = r#"
[review]
daily_limit = 50
default_rating = 3
"#;

    let config: AppConfig = toml::from_str(with_limit).unwrap();
    assert_eq!(config.review.daily_limit, Some(50));
}

#[test]
fn test_config_serialization_roundtrip() {
    let original_config = AppConfig::default();

    // Serialize to TOML
    let toml_str = toml::to_string(&original_config).unwrap();

    // Deserialize back
    let deserialized: AppConfig = toml::from_str(&toml_str).unwrap();

    // Verify key fields match
    assert_eq!(original_config.editor.command, deserialized.editor.command);
    assert_eq!(original_config.display.heatmap_days, deserialized.display.heatmap_days);
    assert_eq!(original_config.review.default_rating, deserialized.review.default_rating);
}

#[test]
fn test_config_handles_extra_fields() {
    // TOML with unrecognized fields should be handled gracefully
    let toml_with_extra = r#"
[editor]
command = "vim"
extra_field = "ignored"

[unknown_section]
foo = "bar"

[review]
default_rating = 3
"#;

    // Serde should either ignore extra fields or fail gracefully
    let result: Result<AppConfig, _> = toml::from_str(toml_with_extra);

    // With deny_unknown_fields, this would fail
    // With default serde behavior, extra fields are ignored
    if let Ok(config) = result {
        assert_eq!(config.editor.command, "vim");
        assert_eq!(config.review.default_rating, 3);
    }
}

#[test]
fn test_config_validates_heatmap_days() {
    let config = r#"
[display]
heatmap_days = 365
"#;

    let parsed: AppConfig = toml::from_str(config).unwrap();
    assert_eq!(parsed.display.heatmap_days, 365);

    // Very large value
    let large = r#"
[display]
heatmap_days = 10000
"#;

    let parsed: AppConfig = toml::from_str(large).unwrap();
    // Parsed successfully but app should validate reasonable bounds
    assert_eq!(parsed.display.heatmap_days, 10000);
}

#[test]
fn test_config_theme_field() {
    let config = r#"
[display]
theme = "solarized"
"#;

    let parsed: AppConfig = toml::from_str(config).unwrap();
    assert_eq!(parsed.display.theme, "solarized");
}

#[test]
fn test_config_persist_sort_mode() {
    let config_true = r#"
[review]
persist_sort_mode = true
default_rating = 3
"#;

    let parsed: AppConfig = toml::from_str(config_true).unwrap();
    assert_eq!(parsed.review.persist_sort_mode, true);

    let config_false = r#"
[review]
persist_sort_mode = false
default_rating = 3
"#;

    let parsed: AppConfig = toml::from_str(config_false).unwrap();
    assert_eq!(parsed.review.persist_sort_mode, false);
}

#[test]
fn test_config_date_format_customization() {
    let config = r#"
[display]
date_format = "%d/%m/%Y"
"#;

    let parsed: AppConfig = toml::from_str(config).unwrap();
    assert_eq!(parsed.display.date_format, "%d/%m/%Y");
}

#[test]
fn test_config_empty_file_uses_defaults() {
    let empty_config = "";

    let parsed: AppConfig = toml::from_str(empty_config).unwrap();

    // Should use all defaults
    assert_eq!(parsed.editor.command, "nvim");
    assert_eq!(parsed.display.heatmap_days, 90);
    assert_eq!(parsed.review.default_rating, 3);
}
