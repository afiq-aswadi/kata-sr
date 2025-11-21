//! Comprehensive tests for the workbook system.
//!
//! Tests workbook manifest loading, validation, HTML generation, and error handling.

use kata_sr::core::workbook::{generate_workbook_html, load_workbooks, Workbook, WorkbookExercise, WorkbookMeta, WorkbookResource};
use std::fs;
use std::path::PathBuf;
use tempfile::TempDir;

/// Helper to create a minimal valid workbook manifest
fn create_minimal_manifest(kata_name: &str) -> String {
    format!(
        r#"
[workbook]
id = "test_workbook"
title = "Test Workbook"

[[exercises]]
slug = "ex1"
title = "Exercise 1"
kata = "{}"
objective = "Learn something"
"#,
        kata_name
    )
}

/// Helper to create a kata directory with template
fn setup_kata_in_exercises(exercises_dir: &std::path::Path, kata_name: &str) {
    let kata_dir = exercises_dir.join(kata_name);
    fs::create_dir_all(&kata_dir).unwrap();

    // Create minimal manifest.toml
    let manifest = format!(
        r#"
[kata]
name = "{}"
category = "test"
base_difficulty = 2
description = "Test kata"
"#,
        kata_name
    );
    fs::write(kata_dir.join("manifest.toml"), manifest).unwrap();

    // Create template.py
    fs::write(
        kata_dir.join("template.py"),
        "# TODO: Implement this\ndef solution():\n    pass\n",
    )
    .unwrap();

    // Create test_kata.py
    fs::write(
        kata_dir.join("test_kata.py"),
        "def test_solution():\n    assert True\n",
    )
    .unwrap();
}

#[test]
fn test_workbook_loads_valid_manifest() {
    let temp_dir = TempDir::new().unwrap();
    let workbooks_dir = temp_dir.path().join("workbooks");
    let exercises_dir = temp_dir.path().join("katas/exercises");
    fs::create_dir_all(&workbooks_dir).unwrap();
    fs::create_dir_all(&exercises_dir).unwrap();

    // Create a test kata
    setup_kata_in_exercises(&exercises_dir, "test_kata");

    // Create workbook
    let wb_dir = workbooks_dir.join("basic");
    fs::create_dir_all(&wb_dir).unwrap();
    let manifest = create_minimal_manifest("test_kata");
    fs::write(wb_dir.join("manifest.toml"), manifest).unwrap();

    // Load workbooks (need to set working directory)
    std::env::set_current_dir(&temp_dir).unwrap();
    let workbooks = load_workbooks().unwrap();

    assert_eq!(workbooks.len(), 1);
    assert_eq!(workbooks[0].meta.id, "test_workbook");
    assert_eq!(workbooks[0].meta.title, "Test Workbook");
    assert_eq!(workbooks[0].exercises.len(), 1);
    assert_eq!(workbooks[0].exercises[0].slug, "ex1");
}

#[test]
fn test_workbook_rejects_invalid_toml() {
    let temp_dir = TempDir::new().unwrap();
    let workbooks_dir = temp_dir.path().join("workbooks");
    fs::create_dir_all(&workbooks_dir).unwrap();

    let wb_dir = workbooks_dir.join("broken");
    fs::create_dir_all(&wb_dir).unwrap();

    // Invalid TOML: missing closing bracket
    let invalid_toml = r#"
[workbook
id = "test"
"#;
    fs::write(wb_dir.join("manifest.toml"), invalid_toml).unwrap();

    std::env::set_current_dir(&temp_dir).unwrap();
    let result = load_workbooks();
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("failed to parse"));
}

#[test]
fn test_workbook_validates_kata_references() {
    let temp_dir = TempDir::new().unwrap();
    let workbooks_dir = temp_dir.path().join("workbooks");
    let exercises_dir = temp_dir.path().join("katas/exercises");
    fs::create_dir_all(&workbooks_dir).unwrap();
    fs::create_dir_all(&exercises_dir).unwrap();

    // Note: NOT creating the kata "nonexistent_kata"

    let wb_dir = workbooks_dir.join("invalid");
    fs::create_dir_all(&wb_dir).unwrap();
    let manifest = create_minimal_manifest("nonexistent_kata");
    fs::write(wb_dir.join("manifest.toml"), manifest).unwrap();

    std::env::set_current_dir(&temp_dir).unwrap();
    let result = load_workbooks();

    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("missing"));
}

#[test]
fn test_workbook_rejects_empty_id() {
    let temp_dir = TempDir::new().unwrap();
    let workbooks_dir = temp_dir.path().join("workbooks");
    let exercises_dir = temp_dir.path().join("katas/exercises");
    fs::create_dir_all(&workbooks_dir).unwrap();
    fs::create_dir_all(&exercises_dir).unwrap();

    setup_kata_in_exercises(&exercises_dir, "test_kata");

    let wb_dir = workbooks_dir.join("empty_id");
    fs::create_dir_all(&wb_dir).unwrap();
    let manifest = r#"
[workbook]
id = "  "
title = "Test"

[[exercises]]
slug = "ex1"
title = "Exercise 1"
kata = "test_kata"
objective = "Learn"
"#;
    fs::write(wb_dir.join("manifest.toml"), manifest).unwrap();

    std::env::set_current_dir(&temp_dir).unwrap();
    let result = load_workbooks();
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("id cannot be empty"));
}

#[test]
fn test_workbook_rejects_empty_title() {
    let temp_dir = TempDir::new().unwrap();
    let workbooks_dir = temp_dir.path().join("workbooks");
    let exercises_dir = temp_dir.path().join("katas/exercises");
    fs::create_dir_all(&workbooks_dir).unwrap();
    fs::create_dir_all(&exercises_dir).unwrap();

    setup_kata_in_exercises(&exercises_dir, "test_kata");

    let wb_dir = workbooks_dir.join("empty_title");
    fs::create_dir_all(&wb_dir).unwrap();
    let manifest = r#"
[workbook]
id = "test"
title = ""

[[exercises]]
slug = "ex1"
title = "Exercise 1"
kata = "test_kata"
objective = "Learn"
"#;
    fs::write(wb_dir.join("manifest.toml"), manifest).unwrap();

    std::env::set_current_dir(&temp_dir).unwrap();
    let result = load_workbooks();
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("title cannot be empty"));
}

#[test]
fn test_workbook_requires_at_least_one_exercise() {
    let temp_dir = TempDir::new().unwrap();
    let workbooks_dir = temp_dir.path().join("workbooks");
    let exercises_dir = temp_dir.path().join("katas/exercises");
    fs::create_dir_all(&workbooks_dir).unwrap();
    fs::create_dir_all(&exercises_dir).unwrap();

    let wb_dir = workbooks_dir.join("no_exercises");
    fs::create_dir_all(&wb_dir).unwrap();
    let manifest = r#"
[workbook]
id = "test"
title = "Test Workbook"

exercises = []
"#;
    fs::write(wb_dir.join("manifest.toml"), manifest).unwrap();

    std::env::set_current_dir(&temp_dir).unwrap();
    let result = load_workbooks();
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("at least one exercise"));
}

#[test]
fn test_workbook_rejects_duplicate_exercise_slugs() {
    let temp_dir = TempDir::new().unwrap();
    let workbooks_dir = temp_dir.path().join("workbooks");
    let exercises_dir = temp_dir.path().join("katas/exercises");
    fs::create_dir_all(&workbooks_dir).unwrap();
    fs::create_dir_all(&exercises_dir).unwrap();

    setup_kata_in_exercises(&exercises_dir, "test_kata");

    let wb_dir = workbooks_dir.join("duplicate_slug");
    fs::create_dir_all(&wb_dir).unwrap();
    let manifest = r#"
[workbook]
id = "test"
title = "Test Workbook"

[[exercises]]
slug = "ex1"
title = "Exercise 1"
kata = "test_kata"
objective = "Learn something"

[[exercises]]
slug = "ex1"
title = "Exercise 2"
kata = "test_kata"
objective = "Learn more"
"#;
    fs::write(wb_dir.join("manifest.toml"), manifest).unwrap();

    std::env::set_current_dir(&temp_dir).unwrap();
    let result = load_workbooks();
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("duplicate"));
}

#[test]
fn test_workbook_validates_exercise_dependencies() {
    let temp_dir = TempDir::new().unwrap();
    let workbooks_dir = temp_dir.path().join("workbooks");
    let exercises_dir = temp_dir.path().join("katas/exercises");
    fs::create_dir_all(&workbooks_dir).unwrap();
    fs::create_dir_all(&exercises_dir).unwrap();

    setup_kata_in_exercises(&exercises_dir, "test_kata");

    let wb_dir = workbooks_dir.join("invalid_dep");
    fs::create_dir_all(&wb_dir).unwrap();
    let manifest = r#"
[workbook]
id = "test"
title = "Test Workbook"

[[exercises]]
slug = "ex1"
title = "Exercise 1"
kata = "test_kata"
objective = "Learn"
dependencies = ["nonexistent"]
"#;
    fs::write(wb_dir.join("manifest.toml"), manifest).unwrap();

    std::env::set_current_dir(&temp_dir).unwrap();
    let result = load_workbooks();
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("unknown slug"));
}

#[test]
fn test_workbook_detects_duplicate_ids() {
    let temp_dir = TempDir::new().unwrap();
    let workbooks_dir = temp_dir.path().join("workbooks");
    let exercises_dir = temp_dir.path().join("katas/exercises");
    fs::create_dir_all(&workbooks_dir).unwrap();
    fs::create_dir_all(&exercises_dir).unwrap();

    setup_kata_in_exercises(&exercises_dir, "test_kata");

    // Create two workbooks with the same ID
    for i in 1..=2 {
        let wb_dir = workbooks_dir.join(format!("wb{}", i));
        fs::create_dir_all(&wb_dir).unwrap();
        let manifest = r#"
[workbook]
id = "duplicate_id"
title = "Test Workbook"

[[exercises]]
slug = "ex1"
title = "Exercise 1"
kata = "test_kata"
objective = "Learn"
"#;
        fs::write(wb_dir.join("manifest.toml"), manifest).unwrap();
    }

    std::env::set_current_dir(&temp_dir).unwrap();
    let result = load_workbooks();
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("duplicate workbook id"));
}

#[test]
fn test_html_generation_with_full_metadata() {
    let temp_dir = TempDir::new().unwrap();
    let assets_dir = temp_dir.path().join("assets/workbooks/test");

    let workbook = Workbook {
        meta: WorkbookMeta {
            id: "test".to_string(),
            title: "Test Workbook".to_string(),
            summary: "A test workbook for unit tests".to_string(),
            learning_goals: vec!["Goal 1".to_string(), "Goal 2".to_string()],
            prerequisites: vec!["Python basics".to_string()],
            resources: vec![WorkbookResource {
                title: "Documentation".to_string(),
                url: "https://example.com".to_string(),
            }],
            kata_namespace: None,
        },
        exercises: vec![WorkbookExercise {
            slug: "ex1".to_string(),
            title: "Exercise 1".to_string(),
            kata: "test_kata".to_string(),
            objective: "Learn basics".to_string(),
            acceptance: vec!["Tests pass".to_string()],
            hints: vec!["Use a loop".to_string()],
            assets: vec!["diagram.png".to_string()],
            dependencies: vec![],
        }],
        manifest_path: PathBuf::from("workbooks/test/manifest.toml"),
        html_path: assets_dir.join("index.html"),
    };

    let result = generate_workbook_html(&workbook);
    assert!(result.is_ok());

    let html_content = fs::read_to_string(&workbook.html_path).unwrap();

    // Verify HTML contains expected content
    assert!(html_content.contains("Test Workbook"));
    assert!(html_content.contains("A test workbook for unit tests"));
    assert!(html_content.contains("Goal 1"));
    assert!(html_content.contains("Goal 2"));
    assert!(html_content.contains("Python basics"));
    assert!(html_content.contains("https://example.com"));
    assert!(html_content.contains("Exercise 1"));
    assert!(html_content.contains("Learn basics"));
}

#[test]
fn test_html_escapes_special_characters() {
    let temp_dir = TempDir::new().unwrap();
    let assets_dir = temp_dir.path().join("assets/workbooks/escape_test");

    let workbook = Workbook {
        meta: WorkbookMeta {
            id: "escape_test".to_string(),
            title: "<script>alert('xss')</script>".to_string(),
            summary: "Test & verify < > \" ' escaping".to_string(),
            learning_goals: vec!["<b>Bold goal</b>".to_string()],
            prerequisites: vec![],
            resources: vec![],
            kata_namespace: None,
        },
        exercises: vec![WorkbookExercise {
            slug: "ex1".to_string(),
            title: "Test <img> tags".to_string(),
            kata: "test_kata".to_string(),
            objective: "Prevent XSS & injection".to_string(),
            acceptance: vec![],
            hints: vec![],
            assets: vec![],
            dependencies: vec![],
        }],
        manifest_path: PathBuf::from("workbooks/escape_test/manifest.toml"),
        html_path: assets_dir.join("index.html"),
    };

    let result = generate_workbook_html(&workbook);
    assert!(result.is_ok());

    let html_content = fs::read_to_string(&workbook.html_path).unwrap();

    // Verify dangerous characters are escaped
    assert!(html_content.contains("&lt;script&gt;"));
    assert!(html_content.contains("&amp;"));
    assert!(html_content.contains("&lt;b&gt;Bold goal&lt;/b&gt;"));
    assert!(html_content.contains("&lt;img&gt;"));
    assert!(!html_content.contains("<script>alert('xss')</script>"));
}

#[test]
fn test_workbook_returns_empty_when_directory_missing() {
    let temp_dir = TempDir::new().unwrap();
    std::env::set_current_dir(&temp_dir).unwrap();

    // No workbooks directory exists
    let workbooks = load_workbooks().unwrap();
    assert_eq!(workbooks.len(), 0);
}

#[test]
fn test_workbook_skips_non_directory_entries() {
    let temp_dir = TempDir::new().unwrap();
    let workbooks_dir = temp_dir.path().join("workbooks");
    fs::create_dir_all(&workbooks_dir).unwrap();

    // Create a regular file (not a directory)
    fs::write(workbooks_dir.join("not_a_workbook.txt"), "test").unwrap();

    std::env::set_current_dir(&temp_dir).unwrap();
    let workbooks = load_workbooks().unwrap();
    assert_eq!(workbooks.len(), 0);
}

#[test]
fn test_workbook_skips_directories_without_manifest() {
    let temp_dir = TempDir::new().unwrap();
    let workbooks_dir = temp_dir.path().join("workbooks");
    fs::create_dir_all(&workbooks_dir).unwrap();

    // Create a directory but no manifest.toml
    let wb_dir = workbooks_dir.join("incomplete");
    fs::create_dir_all(&wb_dir).unwrap();

    std::env::set_current_dir(&temp_dir).unwrap();
    let workbooks = load_workbooks().unwrap();
    assert_eq!(workbooks.len(), 0);
}

#[test]
fn test_html_path_construction() {
    let temp_dir = TempDir::new().unwrap();
    let workbooks_dir = temp_dir.path().join("workbooks");
    let exercises_dir = temp_dir.path().join("katas/exercises");
    fs::create_dir_all(&workbooks_dir).unwrap();
    fs::create_dir_all(&exercises_dir).unwrap();

    setup_kata_in_exercises(&exercises_dir, "test_kata");

    let wb_dir = workbooks_dir.join("path_test");
    fs::create_dir_all(&wb_dir).unwrap();
    let manifest = r#"
[workbook]
id = "my_workbook"
title = "Path Test"

[[exercises]]
slug = "ex1"
title = "Exercise 1"
kata = "test_kata"
objective = "Test paths"
"#;
    fs::write(wb_dir.join("manifest.toml"), manifest).unwrap();

    std::env::set_current_dir(&temp_dir).unwrap();
    let workbooks = load_workbooks().unwrap();

    assert_eq!(workbooks.len(), 1);
    assert_eq!(
        workbooks[0].html_path,
        PathBuf::from("assets/workbooks/my_workbook/index.html")
    );
}

#[test]
fn test_workbook_with_all_optional_fields() {
    let temp_dir = TempDir::new().unwrap();
    let workbooks_dir = temp_dir.path().join("workbooks");
    let exercises_dir = temp_dir.path().join("katas/exercises");
    fs::create_dir_all(&workbooks_dir).unwrap();
    fs::create_dir_all(&exercises_dir).unwrap();

    setup_kata_in_exercises(&exercises_dir, "test_kata");

    let wb_dir = workbooks_dir.join("complete");
    fs::create_dir_all(&wb_dir).unwrap();
    let manifest = r#"
[workbook]
id = "complete"
title = "Complete Workbook"
summary = "Has all optional fields"
learning_goals = ["Goal 1", "Goal 2"]
prerequisites = ["Prereq 1"]
kata_namespace = "my_namespace"

[[workbook.resources]]
title = "Resource 1"
url = "https://example.com/1"

[[workbook.resources]]
title = "Resource 2"
url = "https://example.com/2"

[[exercises]]
slug = "ex1"
title = "Exercise 1"
kata = "test_kata"
objective = "Complete example"
acceptance = ["Criterion 1", "Criterion 2"]
hints = ["Hint 1", "Hint 2"]
assets = ["asset1.png", "asset2.png"]
dependencies = []
"#;
    fs::write(wb_dir.join("manifest.toml"), manifest).unwrap();

    std::env::set_current_dir(&temp_dir).unwrap();
    let workbooks = load_workbooks().unwrap();

    assert_eq!(workbooks.len(), 1);
    let wb = &workbooks[0];
    assert_eq!(wb.meta.summary, "Has all optional fields");
    assert_eq!(wb.meta.learning_goals.len(), 2);
    assert_eq!(wb.meta.prerequisites.len(), 1);
    assert_eq!(wb.meta.resources.len(), 2);
    assert_eq!(wb.meta.kata_namespace, Some("my_namespace".to_string()));
    assert_eq!(wb.exercises[0].acceptance.len(), 2);
    assert_eq!(wb.exercises[0].hints.len(), 2);
    assert_eq!(wb.exercises[0].assets.len(), 2);
}
