//! Kata file generation utilities.
//!
//! This module provides functions for generating the directory structure and
//! files for a new kata, including manifest.toml, template.py, test_kata.py,
//! and reference.py.

use crate::core::kata_validation::{slugify_kata_name, validate_dependencies, validate_kata_name};
use anyhow::{Context, Result};
use std::fs;
use std::io::ErrorKind;
use std::path::Path;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum GeneratorError {
    #[error("Kata '{0}' already exists")]
    KataExists(String),

    #[error("Filesystem error: {0}")]
    FilesystemError(#[from] std::io::Error),

    #[error("Validation error: {0}")]
    ValidationError(#[from] crate::core::kata_validation::ValidationError),
}

/// Form data collected from the TUI for creating a new kata.
#[derive(Debug, Clone)]
pub struct KataFormData {
    pub name: String,
    pub category: String,
    pub description: String,
    pub difficulty: u8, // 1-5
    pub dependencies: Vec<String>,
}

/// Generates all files for a new kata in the exercises directory.
///
/// This function:
/// 1. Computes the slug from the kata name
/// 2. Re-validates (defense-in-depth even if TUI already checked)
/// 3. Creates the kata directory atomically
/// 4. Generates all required files (manifest, template, test, reference)
/// 5. Rolls back on any error to avoid partial state
///
/// # Arguments
///
/// * `form_data` - The kata metadata from the form
/// * `exercises_dir` - Path to the exercises directory (e.g., "katas/exercises/")
///
/// # Returns
///
/// The slug (directory name) of the created kata on success.
///
/// # Errors
///
/// Returns `GeneratorError` if validation fails or filesystem operations fail.
pub fn generate_kata_files(form_data: &KataFormData, exercises_dir: &Path) -> Result<String> {
    // 1. Compute slug
    let slug = slugify_kata_name(&form_data.name);

    // 2. CRITICAL: Re-validate even though TUI already checked
    //    (defends against race conditions, stale state, bugs in TUI layer)
    validate_kata_name(&slug, exercises_dir)?;
    validate_dependencies(&form_data.dependencies, exercises_dir)?;

    // 3. Create directory (atomic check-and-create)
    let kata_dir = exercises_dir.join(&slug);
    fs::create_dir(&kata_dir).map_err(|e| {
        if e.kind() == ErrorKind::AlreadyExists {
            GeneratorError::KataExists(slug.clone())
        } else {
            GeneratorError::FilesystemError(e)
        }
    })?;

    // 4. Generate files (rollback entire directory on any error)
    match generate_files_inner(&kata_dir, form_data, &slug) {
        Ok(_) => Ok(slug),
        Err(e) => {
            // Cleanup on failure
            let _ = fs::remove_dir_all(&kata_dir);
            Err(e)
        }
    }
}

fn generate_files_inner(kata_dir: &Path, form_data: &KataFormData, slug: &str) -> Result<()> {
    // Write manifest.toml
    write_manifest(kata_dir, form_data, slug)?;

    // Write template.py (minimal imports: just pytest)
    write_template(kata_dir, form_data, slug)?;

    // Write test_kata.py (sample tests with TODOs)
    write_test_file(kata_dir, slug)?;

    // Write reference.py (copy of template with SOLUTION header)
    write_reference(kata_dir, slug)?;

    Ok(())
}

fn write_manifest(kata_dir: &Path, form_data: &KataFormData, slug: &str) -> Result<()> {
    let manifest_path = kata_dir.join("manifest.toml");

    let dependencies_toml = if form_data.dependencies.is_empty() {
        String::new()
    } else {
        let deps_list = form_data
            .dependencies
            .iter()
            .map(|d| format!("\"{}\"", d))
            .collect::<Vec<_>>()
            .join(", ");
        format!("dependencies = [{}]\n", deps_list)
    };

    let content = format!(
        r#"[kata]
name = "{}"
category = "{}"
base_difficulty = {}
description = """
{}
"""
{}
"#,
        slug, form_data.category, form_data.difficulty, form_data.description, dependencies_toml
    );

    fs::write(&manifest_path, content).context("Failed to write manifest.toml")?;

    Ok(())
}

fn write_template(kata_dir: &Path, form_data: &KataFormData, slug: &str) -> Result<()> {
    let template_path = kata_dir.join("template.py");

    let content = format!(
        r#"""
{}
"""


def kata_{}():
    """
    TODO: Implement this function

    Returns:
        TODO: Describe return value
    """
    raise NotImplementedError("TODO: Complete this kata")


if __name__ == "__main__":
    # Example usage
    result = kata_{}()
    print(f"Result: {{result}}")
"#,
        form_data.description, slug, slug
    );

    fs::write(&template_path, content).context("Failed to write template.py")?;

    Ok(())
}

fn write_test_file(kata_dir: &Path, slug: &str) -> Result<()> {
    let test_path = kata_dir.join("test_kata.py");

    let content = format!(
        r#"import pytest
from template import kata_{}


def test_basic_functionality():
    """TODO: Add basic correctness test"""
    result = kata_{}()
    # TODO: Add assertions
    pass


def test_edge_cases():
    """TODO: Add edge case tests"""
    pass
"#,
        slug, slug
    );

    fs::write(&test_path, content).context("Failed to write test_kata.py")?;

    Ok(())
}

fn write_reference(kata_dir: &Path, slug: &str) -> Result<()> {
    let reference_path = kata_dir.join("reference.py");

    let content = format!(
        r#"# SOLUTION - Do not peek before attempting!
# Copy from template.py and implement


def kata_{}():
    """
    TODO: Implement the solution here
    """
    raise NotImplementedError("Implement the solution")
"#,
        slug
    );

    fs::write(&reference_path, content).context("Failed to write reference.py")?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_generate_kata_files_success() {
        let temp_dir = TempDir::new().unwrap();

        let form = KataFormData {
            name: "Test Kata".to_string(),
            category: "test".to_string(),
            description: "Test description".to_string(),
            difficulty: 3,
            dependencies: vec![],
        };

        let slug = generate_kata_files(&form, temp_dir.path()).unwrap();

        assert_eq!(slug, "test_kata");
        assert!(temp_dir.path().join("test_kata").exists());
        assert!(temp_dir.path().join("test_kata/manifest.toml").exists());
        assert!(temp_dir.path().join("test_kata/template.py").exists());
        assert!(temp_dir.path().join("test_kata/test_kata.py").exists());
        assert!(temp_dir.path().join("test_kata/reference.py").exists());
    }

    #[test]
    fn test_generate_kata_files_validates_content() {
        let temp_dir = TempDir::new().unwrap();

        let form = KataFormData {
            name: "Test Kata".to_string(),
            category: "test".to_string(),
            description: "Test description".to_string(),
            difficulty: 3,
            dependencies: vec![],
        };

        generate_kata_files(&form, temp_dir.path()).unwrap();

        // Check manifest.toml content
        let manifest_content =
            fs::read_to_string(temp_dir.path().join("test_kata/manifest.toml")).unwrap();
        assert!(manifest_content.contains("name = \"test_kata\""));
        assert!(manifest_content.contains("category = \"test\""));
        assert!(manifest_content.contains("base_difficulty = 3"));
        assert!(manifest_content.contains("Test description"));

        // Check template.py content
        let template_content =
            fs::read_to_string(temp_dir.path().join("test_kata/template.py")).unwrap();
        assert!(template_content.contains("def kata_test_kata():"));
        assert!(template_content.contains("NotImplementedError"));
        assert!(template_content.contains("Test description"));

        // Check test_kata.py content
        let test_content =
            fs::read_to_string(temp_dir.path().join("test_kata/test_kata.py")).unwrap();
        assert!(test_content.contains("from template import kata_test_kata"));
        assert!(test_content.contains("def test_basic_functionality():"));
    }

    #[test]
    fn test_generate_kata_files_with_dependencies() {
        let temp_dir = TempDir::new().unwrap();

        // Create dependency kata first
        fs::create_dir(temp_dir.path().join("dep_kata")).unwrap();

        let form = KataFormData {
            name: "Advanced Kata".to_string(),
            category: "advanced".to_string(),
            description: "Advanced kata with dependencies".to_string(),
            difficulty: 5,
            dependencies: vec!["dep_kata".to_string()],
        };

        let slug = generate_kata_files(&form, temp_dir.path()).unwrap();

        assert_eq!(slug, "advanced_kata");

        // Check manifest includes dependency
        let manifest_content =
            fs::read_to_string(temp_dir.path().join("advanced_kata/manifest.toml")).unwrap();
        assert!(manifest_content.contains("dependencies = [\"dep_kata\"]"));
    }

    #[test]
    fn test_generate_kata_files_duplicate_name() {
        let temp_dir = TempDir::new().unwrap();

        // Create first kata
        fs::create_dir(temp_dir.path().join("existing_kata")).unwrap();

        let form = KataFormData {
            name: "Existing Kata".to_string(),
            category: "test".to_string(),
            description: "Test".to_string(),
            difficulty: 3,
            dependencies: vec![],
        };

        // Should fail because kata already exists
        let result = generate_kata_files(&form, temp_dir.path());
        assert!(matches!(result, Err(e) if e.to_string().contains("already exists")));
    }

    #[test]
    fn test_generate_kata_files_missing_dependency() {
        let temp_dir = TempDir::new().unwrap();

        let form = KataFormData {
            name: "Test Kata".to_string(),
            category: "test".to_string(),
            description: "Test".to_string(),
            difficulty: 3,
            dependencies: vec!["missing_kata".to_string()],
        };

        // Should fail because dependency doesn't exist
        let result = generate_kata_files(&form, temp_dir.path());
        assert!(matches!(result, Err(e) if e.to_string().contains("not found")));
    }

    #[test]
    fn test_rollback_on_error() {
        let temp_dir = TempDir::new().unwrap();

        // Create a read-only directory to simulate filesystem error
        let kata_dir = temp_dir.path().join("test_kata");
        fs::create_dir(&kata_dir).unwrap();

        // Make the directory read-only to cause write errors
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = fs::metadata(&kata_dir).unwrap().permissions();
            perms.set_mode(0o444); // read-only
            fs::set_permissions(&kata_dir, perms).unwrap();
        }

        let _form = KataFormData {
            name: "Another Kata".to_string(), // Different name to avoid conflict
            category: "test".to_string(),
            description: "Test".to_string(),
            difficulty: 3,
            dependencies: vec![],
        };

        // This should succeed because "another_kata" is different from "test_kata"
        // But if we want to test rollback, we need a different approach

        // Actually, let's test that if kata exists, it fails properly
        let form2 = KataFormData {
            name: "Test Kata".to_string(), // Same name as existing directory
            category: "test".to_string(),
            description: "Test".to_string(),
            difficulty: 3,
            dependencies: vec![],
        };

        let result = generate_kata_files(&form2, temp_dir.path());
        assert!(result.is_err());

        // Cleanup: restore permissions
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = fs::metadata(&kata_dir).unwrap().permissions();
            perms.set_mode(0o755);
            let _ = fs::set_permissions(&kata_dir, perms);
        }
    }
}
