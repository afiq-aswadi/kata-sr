//! Kata name validation and slugification utilities.
//!
//! This module provides pure functions for validating kata names and dependencies
//! before creating new katas. These functions are designed to be called both from
//! the TUI layer (for inline validation) and the generator layer (for defense-in-depth).

use std::path::Path;
use thiserror::Error;

#[derive(Debug, Error, PartialEq)]
pub enum ValidationError {
    #[error("Kata name cannot be empty")]
    EmptyName,

    #[error("Kata '{0}' already exists")]
    KataExists(String),

    #[error("Kata name cannot start with a digit")]
    InvalidStart,

    #[error("Dependencies not found: {}", .0.join(", "))]
    MissingDependency(Vec<String>),
}

/// Converts a user-provided kata name to a valid slug (snake_case identifier).
///
/// Rules:
/// - Convert to lowercase
/// - Replace spaces, hyphens with underscores
/// - Strip all non-alphanumeric except underscore
/// - Remove leading/trailing underscores
/// - Collapse multiple consecutive underscores to one
///
/// # Examples
///
/// ```
/// # use kata_sr::core::kata_validation::slugify_kata_name;
/// assert_eq!(slugify_kata_name("Multi-Head Attention"), "multi_head_attention");
/// assert_eq!(slugify_kata_name("BFS/DFS"), "bfs_dfs");
/// assert_eq!(slugify_kata_name("  spaces  "), "spaces");
/// ```
pub fn slugify_kata_name(input: &str) -> String {
    input
        .to_lowercase()
        // Replace common separators with underscore
        .chars()
        .map(|c| match c {
            ' ' | '-' => '_',
            c if c.is_alphanumeric() || c == '_' => c,
            _ => '_', // Any other character becomes underscore
        })
        .collect::<String>()
        // Collapse consecutive underscores
        .split('_')
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join("_")
}

/// Validates that a kata name (slug) is valid and doesn't conflict with existing katas.
///
/// Checks:
/// - Non-empty after slugification
/// - Doesn't start with a digit
/// - Doesn't already exist in the exercises directory
///
/// # Arguments
///
/// * `slug` - The slugified kata name to validate
/// * `exercises_dir` - Path to the exercises directory (e.g., "katas/exercises/")
///
/// # Errors
///
/// Returns `ValidationError` if the name is invalid or already exists.
pub fn validate_kata_name(slug: &str, exercises_dir: &Path) -> Result<(), ValidationError> {
    // Check non-empty
    if slug.is_empty() {
        return Err(ValidationError::EmptyName);
    }

    // Check doesn't start with digit
    if slug.chars().next().unwrap().is_ascii_digit() {
        return Err(ValidationError::InvalidStart);
    }

    // Check uniqueness
    let kata_path = exercises_dir.join(slug);
    if kata_path.exists() {
        return Err(ValidationError::KataExists(slug.to_string()));
    }

    Ok(())
}

/// Validates that all specified dependencies exist in the exercises directory.
///
/// # Arguments
///
/// * `deps` - List of kata names (slugs) that are required as dependencies
/// * `exercises_dir` - Path to the exercises directory
///
/// # Errors
///
/// Returns `ValidationError::MissingDependency` with a list of missing kata names.
pub fn validate_dependencies(deps: &[String], exercises_dir: &Path) -> Result<(), ValidationError> {
    let missing: Vec<String> = deps
        .iter()
        .filter(|dep| !exercises_dir.join(dep).exists())
        .cloned()
        .collect();

    if !missing.is_empty() {
        return Err(ValidationError::MissingDependency(missing));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn test_slugify_basic() {
        assert_eq!(
            slugify_kata_name("Multi-Head Attention"),
            "multi_head_attention"
        );
        assert_eq!(slugify_kata_name("BFS/DFS"), "bfs_dfs");
        assert_eq!(slugify_kata_name("  spaces  "), "spaces");
        assert_eq!(slugify_kata_name("123abc"), "123abc");
    }

    #[test]
    fn test_slugify_edge_cases() {
        // Leading/trailing underscores removed
        assert_eq!(slugify_kata_name("___test___"), "test");

        // Multiple consecutive separators collapsed
        assert_eq!(slugify_kata_name("foo---bar___baz"), "foo_bar_baz");

        // Special characters become underscores
        assert_eq!(slugify_kata_name("hello@world!"), "hello_world");

        // Empty string
        assert_eq!(slugify_kata_name(""), "");

        // Only special characters
        assert_eq!(slugify_kata_name("!!!"), "");
    }

    #[test]
    fn test_validate_kata_name_valid() {
        let temp_dir = TempDir::new().unwrap();

        // Valid new kata
        assert!(validate_kata_name("new_kata", temp_dir.path()).is_ok());
        assert!(validate_kata_name("another_kata", temp_dir.path()).is_ok());
    }

    #[test]
    fn test_validate_kata_name_exists() {
        let temp_dir = TempDir::new().unwrap();

        // Create conflicting directory
        fs::create_dir(temp_dir.path().join("existing")).unwrap();

        let result = validate_kata_name("existing", temp_dir.path());
        assert!(matches!(result, Err(ValidationError::KataExists(_))));
        assert_eq!(
            result.unwrap_err(),
            ValidationError::KataExists("existing".to_string())
        );
    }

    #[test]
    fn test_validate_kata_name_empty() {
        let temp_dir = TempDir::new().unwrap();

        let result = validate_kata_name("", temp_dir.path());
        assert!(matches!(result, Err(ValidationError::EmptyName)));
    }

    #[test]
    fn test_validate_kata_name_starts_with_digit() {
        let temp_dir = TempDir::new().unwrap();

        let result = validate_kata_name("123abc", temp_dir.path());
        assert!(matches!(result, Err(ValidationError::InvalidStart)));
    }

    #[test]
    fn test_validate_dependencies_valid() {
        let temp_dir = TempDir::new().unwrap();

        // Create some kata directories
        fs::create_dir(temp_dir.path().join("kata_a")).unwrap();
        fs::create_dir(temp_dir.path().join("kata_b")).unwrap();

        // Valid dependencies
        let deps = vec!["kata_a".to_string(), "kata_b".to_string()];
        assert!(validate_dependencies(&deps, temp_dir.path()).is_ok());

        // Empty dependencies list is valid
        assert!(validate_dependencies(&[], temp_dir.path()).is_ok());
    }

    #[test]
    fn test_validate_dependencies_missing() {
        let temp_dir = TempDir::new().unwrap();

        // Create only one kata
        fs::create_dir(temp_dir.path().join("kata_a")).unwrap();

        // Missing dependency
        let deps = vec![
            "kata_a".to_string(),
            "missing".to_string(),
            "also_missing".to_string(),
        ];
        let result = validate_dependencies(&deps, temp_dir.path());

        assert!(matches!(result, Err(ValidationError::MissingDependency(_))));

        if let Err(ValidationError::MissingDependency(missing)) = result {
            assert_eq!(missing.len(), 2);
            assert!(missing.contains(&"missing".to_string()));
            assert!(missing.contains(&"also_missing".to_string()));
        }
    }
}
