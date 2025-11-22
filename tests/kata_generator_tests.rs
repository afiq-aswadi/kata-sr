//! Kata generator and validation tests.
//!
//! Tests kata file generation, rollback behavior, slugification, and validation.

use kata_sr::core::kata_generator::{generate_kata_files, KataFormData};
use kata_sr::core::kata_validation::{slugify_kata_name, validate_dependencies, validate_kata_name, ValidationError};
use std::fs;
use tempfile::TempDir;

fn create_test_exercises_dir() -> TempDir {
    let temp_dir = TempDir::new().unwrap();
    let exercises_dir = temp_dir.path().join("exercises");
    fs::create_dir_all(&exercises_dir).unwrap();
    temp_dir
}

#[test]
fn test_generate_kata_creates_all_files() {
    let temp_dir = create_test_exercises_dir();
    let exercises_dir = temp_dir.path().join("exercises");

    let form_data = KataFormData {
        name: "Test Kata".to_string(),
        category: "algorithms".to_string(),
        description: "A test kata".to_string(),
        difficulty: 3,
        dependencies: vec![],
    };

    let slug = generate_kata_files(&form_data, &exercises_dir).unwrap();
    assert_eq!(slug, "test_kata");

    let kata_dir = exercises_dir.join("test_kata");
    assert!(kata_dir.exists());
    assert!(kata_dir.join("manifest.toml").exists());
    assert!(kata_dir.join("template.py").exists());
    assert!(kata_dir.join("test_kata.py").exists());
    assert!(kata_dir.join("reference.py").exists());
}

#[test]
fn test_generate_kata_manifest_contains_correct_metadata() {
    let temp_dir = create_test_exercises_dir();
    let exercises_dir = temp_dir.path().join("exercises");

    let form_data = KataFormData {
        name: "Fibonacci".to_string(),
        category: "dynamic_programming".to_string(),
        description: "Calculate Fibonacci numbers".to_string(),
        difficulty: 4,
        dependencies: vec![],
    };

    generate_kata_files(&form_data, &exercises_dir).unwrap();

    let manifest_content = fs::read_to_string(exercises_dir.join("fibonacci/manifest.toml")).unwrap();
    assert!(manifest_content.contains("name = \"fibonacci\""));
    assert!(manifest_content.contains("category = \"dynamic_programming\""));
    assert!(manifest_content.contains("base_difficulty = 4"));
    assert!(manifest_content.contains("Calculate Fibonacci numbers"));
}

#[test]
fn test_generate_kata_with_dependencies() {
    let temp_dir = create_test_exercises_dir();
    let exercises_dir = temp_dir.path().join("exercises");

    // Create prerequisite kata first
    let prereq_dir = exercises_dir.join("basic_kata");
    fs::create_dir_all(&prereq_dir).unwrap();
    fs::write(
        prereq_dir.join("manifest.toml"),
        r#"[kata]
name = "basic_kata"
category = "fundamentals"
base_difficulty = 1
description = "Basic kata"
"#,
    )
    .unwrap();

    // Create kata with dependency
    let form_data = KataFormData {
        name: "Advanced Kata".to_string(),
        category: "advanced".to_string(),
        description: "Builds on basic_kata".to_string(),
        difficulty: 4,
        dependencies: vec!["basic_kata".to_string()],
    };

    generate_kata_files(&form_data, &exercises_dir).unwrap();

    let manifest_content = fs::read_to_string(exercises_dir.join("advanced_kata/manifest.toml")).unwrap();
    assert!(manifest_content.contains("dependencies = [\"basic_kata\"]"));
}

#[test]
fn test_generate_kata_rejects_existing_name() {
    let temp_dir = create_test_exercises_dir();
    let exercises_dir = temp_dir.path().join("exercises");

    // Create kata first time
    let form_data = KataFormData {
        name: "duplicate".to_string(),
        category: "test".to_string(),
        description: "Test".to_string(),
        difficulty: 2,
        dependencies: vec![],
    };

    generate_kata_files(&form_data, &exercises_dir).unwrap();

    // Try to create again (should fail)
    let result = generate_kata_files(&form_data, &exercises_dir);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("already exists"));
}

#[test]
#[cfg(unix)] // Unix-only test (uses Unix file permissions)
fn test_generate_kata_rolls_back_on_failure() {
    use std::os::unix::fs::PermissionsExt;

    let temp_dir = create_test_exercises_dir();
    let exercises_dir = temp_dir.path().join("exercises");

    // Make exercises_dir read-only to force a failure during file creation
    let kata_dir = exercises_dir.join("will_fail");
    fs::create_dir_all(&kata_dir).unwrap();

    // Set directory to read-only (no write permission)
    let mut perms = fs::metadata(&kata_dir).unwrap().permissions();
    perms.set_mode(0o444);
    fs::set_permissions(&kata_dir, perms).unwrap();

    let form_data = KataFormData {
        name: "will_fail".to_string(),
        category: "test".to_string(),
        description: "This should fail".to_string(),
        difficulty: 2,
        dependencies: vec![],
    };

    // This should fail and rollback (but kata_dir already exists)
    let result = generate_kata_files(&form_data, &exercises_dir);
    assert!(result.is_err());

    // Reset permissions for cleanup
    let mut perms = fs::metadata(&kata_dir).unwrap().permissions();
    perms.set_mode(0o755);
    fs::set_permissions(&kata_dir, perms).unwrap();
}

#[test]
fn test_slugify_basic_transformations() {
    assert_eq!(slugify_kata_name("Multi-Head Attention"), "multi_head_attention");
    assert_eq!(slugify_kata_name("BFS/DFS"), "bfs_dfs");
    assert_eq!(slugify_kata_name("  spaces  "), "spaces");
    assert_eq!(slugify_kata_name("CamelCase"), "camelcase");
    assert_eq!(slugify_kata_name("snake_case"), "snake_case");
}

#[test]
fn test_slugify_handles_special_characters() {
    assert_eq!(slugify_kata_name("A & B"), "a_b");
    assert_eq!(slugify_kata_name("foo@bar.com"), "foo_bar_com");
    assert_eq!(slugify_kata_name("test#123"), "test_123");
    assert_eq!(slugify_kata_name("hello(world)"), "hello_world");
}

#[test]
fn test_slugify_collapses_underscores() {
    assert_eq!(slugify_kata_name("foo___bar"), "foo_bar");
    assert_eq!(slugify_kata_name("___leading"), "leading");
    assert_eq!(slugify_kata_name("trailing___"), "trailing");
}

#[test]
fn test_slugify_handles_unicode() {
    // is_alphanumeric() includes Unicode alphanumerics, so they're preserved
    assert_eq!(slugify_kata_name("Café"), "café");
    assert_eq!(slugify_kata_name("Héllo Wörld"), "héllo_wörld");
    // CJK characters are also alphanumeric
    assert_eq!(slugify_kata_name("日本語"), "日本語");
    // Mixed ASCII and Unicode
    assert_eq!(slugify_kata_name("Test-Café"), "test_café");
}

#[test]
fn test_slugify_preserves_numbers() {
    assert_eq!(slugify_kata_name("test123"), "test123");
    assert_eq!(slugify_kata_name("123test"), "123test");
    assert_eq!(slugify_kata_name("a1b2c3"), "a1b2c3");
}

#[test]
fn test_validate_kata_name_accepts_valid_names() {
    let temp_dir = create_test_exercises_dir();
    let exercises_dir = temp_dir.path().join("exercises");

    assert!(validate_kata_name("valid_kata", &exercises_dir).is_ok());
    assert!(validate_kata_name("kata123", &exercises_dir).is_ok());
    assert!(validate_kata_name("a_b_c", &exercises_dir).is_ok());
}

#[test]
fn test_validate_kata_name_rejects_empty() {
    let temp_dir = create_test_exercises_dir();
    let exercises_dir = temp_dir.path().join("exercises");

    let result = validate_kata_name("", &exercises_dir);
    assert_eq!(result, Err(ValidationError::EmptyName));
}

#[test]
fn test_validate_kata_name_rejects_starting_with_digit() {
    let temp_dir = create_test_exercises_dir();
    let exercises_dir = temp_dir.path().join("exercises");

    let result = validate_kata_name("123kata", &exercises_dir);
    assert_eq!(result, Err(ValidationError::InvalidStart));
}

#[test]
fn test_validate_kata_name_rejects_existing() {
    let temp_dir = create_test_exercises_dir();
    let exercises_dir = temp_dir.path().join("exercises");

    // Create existing kata directory
    fs::create_dir_all(exercises_dir.join("existing_kata")).unwrap();

    let result = validate_kata_name("existing_kata", &exercises_dir);
    assert!(matches!(result, Err(ValidationError::KataExists(_))));
}

#[test]
fn test_validate_dependencies_with_valid_deps() {
    let temp_dir = create_test_exercises_dir();
    let exercises_dir = temp_dir.path().join("exercises");

    // Create prerequisite katas
    fs::create_dir_all(exercises_dir.join("kata1")).unwrap();
    fs::create_dir_all(exercises_dir.join("kata2")).unwrap();

    let deps = vec!["kata1".to_string(), "kata2".to_string()];
    assert!(validate_dependencies(&deps, &exercises_dir).is_ok());
}

#[test]
fn test_validate_dependencies_with_missing_deps() {
    let temp_dir = create_test_exercises_dir();
    let exercises_dir = temp_dir.path().join("exercises");

    // Only create kata1
    fs::create_dir_all(exercises_dir.join("kata1")).unwrap();

    let deps = vec!["kata1".to_string(), "missing_kata".to_string()];
    let result = validate_dependencies(&deps, &exercises_dir);

    assert!(matches!(result, Err(ValidationError::MissingDependency(_))));
    if let Err(ValidationError::MissingDependency(missing)) = result {
        assert_eq!(missing, vec!["missing_kata"]);
    }
}

#[test]
fn test_validate_dependencies_empty_list() {
    let temp_dir = create_test_exercises_dir();
    let exercises_dir = temp_dir.path().join("exercises");

    let deps: Vec<String> = vec![];
    assert!(validate_dependencies(&deps, &exercises_dir).is_ok());
}

#[test]
fn test_generate_kata_with_empty_description() {
    let temp_dir = create_test_exercises_dir();
    let exercises_dir = temp_dir.path().join("exercises");

    let form_data = KataFormData {
        name: "empty_desc".to_string(),
        category: "test".to_string(),
        description: "".to_string(),
        difficulty: 2,
        dependencies: vec![],
    };

    // Should still succeed (description can be empty)
    let result = generate_kata_files(&form_data, &exercises_dir);
    assert!(result.is_ok());
}

#[test]
fn test_generate_kata_with_multiline_description() {
    let temp_dir = create_test_exercises_dir();
    let exercises_dir = temp_dir.path().join("exercises");

    let form_data = KataFormData {
        name: "multiline".to_string(),
        category: "test".to_string(),
        description: "Line 1\nLine 2\nLine 3".to_string(),
        difficulty: 3,
        dependencies: vec![],
    };

    generate_kata_files(&form_data, &exercises_dir).unwrap();

    let manifest = fs::read_to_string(exercises_dir.join("multiline/manifest.toml")).unwrap();
    assert!(manifest.contains("Line 1"));
    assert!(manifest.contains("Line 2"));
    assert!(manifest.contains("Line 3"));
}

#[test]
fn test_generate_kata_with_special_chars_in_description() {
    let temp_dir = create_test_exercises_dir();
    let exercises_dir = temp_dir.path().join("exercises");

    let form_data = KataFormData {
        name: "special_chars".to_string(),
        category: "test".to_string(),
        description: "Contains \"quotes\" and 'apostrophes' and \\backslashes\\".to_string(),
        difficulty: 2,
        dependencies: vec![],
    };

    generate_kata_files(&form_data, &exercises_dir).unwrap();

    // Verify file was created successfully
    assert!(exercises_dir.join("special_chars/manifest.toml").exists());
}

#[test]
fn test_template_file_contains_function_definition() {
    let temp_dir = create_test_exercises_dir();
    let exercises_dir = temp_dir.path().join("exercises");

    let form_data = KataFormData {
        name: "test_function".to_string(),
        category: "test".to_string(),
        description: "Test function generation".to_string(),
        difficulty: 2,
        dependencies: vec![],
    };

    generate_kata_files(&form_data, &exercises_dir).unwrap();

    let template = fs::read_to_string(exercises_dir.join("test_function/template.py")).unwrap();
    assert!(template.contains("def kata_test_function():"));
    assert!(template.contains("TODO: Implement this function"));
    assert!(template.contains("NotImplementedError"));
}

#[test]
fn test_test_file_imports_from_template() {
    let temp_dir = create_test_exercises_dir();
    let exercises_dir = temp_dir.path().join("exercises");

    let form_data = KataFormData {
        name: "import_test".to_string(),
        category: "test".to_string(),
        description: "Test imports".to_string(),
        difficulty: 2,
        dependencies: vec![],
    };

    generate_kata_files(&form_data, &exercises_dir).unwrap();

    let test_file = fs::read_to_string(exercises_dir.join("import_test/test_kata.py")).unwrap();
    assert!(test_file.contains("from template import kata_import_test"));
    assert!(test_file.contains("import pytest"));
}

#[test]
fn test_reference_file_has_solution_header() {
    let temp_dir = create_test_exercises_dir();
    let exercises_dir = temp_dir.path().join("exercises");

    let form_data = KataFormData {
        name: "reference_test".to_string(),
        category: "test".to_string(),
        description: "Test reference".to_string(),
        difficulty: 2,
        dependencies: vec![],
    };

    generate_kata_files(&form_data, &exercises_dir).unwrap();

    let reference = fs::read_to_string(exercises_dir.join("reference_test/reference.py")).unwrap();
    assert!(reference.contains("# SOLUTION"));
    assert!(reference.contains("Do not peek"));
}

#[test]
fn test_generate_kata_with_extreme_difficulty() {
    let temp_dir = create_test_exercises_dir();
    let exercises_dir = temp_dir.path().join("exercises");

    // Difficulty should be constrained to 1-5
    let form_data_min = KataFormData {
        name: "easy".to_string(),
        category: "test".to_string(),
        description: "Easy".to_string(),
        difficulty: 1,
        dependencies: vec![],
    };

    let form_data_max = KataFormData {
        name: "hard".to_string(),
        category: "test".to_string(),
        description: "Hard".to_string(),
        difficulty: 5,
        dependencies: vec![],
    };

    assert!(generate_kata_files(&form_data_min, &exercises_dir).is_ok());
    assert!(generate_kata_files(&form_data_max, &exercises_dir).is_ok());
}

#[test]
fn test_slugify_edge_case_only_special_chars() {
    let result = slugify_kata_name("!@#$%^&*()");
    // Should collapse to empty or single underscore
    assert!(result.is_empty() || result == "_");
}

#[test]
fn test_slugify_very_long_name() {
    let long_name = "a".repeat(1000);
    let slug = slugify_kata_name(&long_name);
    assert_eq!(slug.len(), 1000);
    assert_eq!(slug, "a".repeat(1000));
}

#[test]
fn test_generate_kata_with_very_long_name() {
    let temp_dir = create_test_exercises_dir();
    let exercises_dir = temp_dir.path().join("exercises");

    let long_name = "Very Long Kata Name ".repeat(10);
    let form_data = KataFormData {
        name: long_name.clone(),
        category: "test".to_string(),
        description: "Long name test".to_string(),
        difficulty: 3,
        dependencies: vec![],
    };

    let result = generate_kata_files(&form_data, &exercises_dir);
    assert!(result.is_ok());

    // Verify slug is properly formed
    let slug = result.unwrap();
    assert!(!slug.contains(' '));
    assert!(exercises_dir.join(&slug).exists());
}
