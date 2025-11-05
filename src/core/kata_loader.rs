//! Kata loader for scanning and loading kata manifests from the exercises directory.

use anyhow::{Context, Result};
use serde::Deserialize;
use std::fs;
use std::path::Path;

/// Represents a kata found in the exercises directory
#[derive(Debug, Clone, Deserialize)]
pub struct AvailableKata {
    pub name: String,
    pub category: String,
    pub base_difficulty: i32,
    pub description: String,
    #[serde(default)]
    pub dependencies: Vec<String>,
}

/// Internal structure for deserializing the manifest TOML file
#[derive(Debug, Deserialize)]
struct KataManifest {
    kata: AvailableKata,
}

/// Scans the katas/exercises/ directory and loads all available katas from manifest files.
///
/// For each subdirectory in exercises/, attempts to read and parse manifest.toml.
/// Skips directories without manifests or with invalid manifests rather than failing.
///
/// # Returns
/// Vector of AvailableKata structs, one per valid kata found.
/// Returns empty vector if exercises directory doesn't exist.
pub fn load_available_katas() -> Result<Vec<AvailableKata>> {
    let exercises_path = Path::new("katas/exercises");

    if !exercises_path.exists() {
        return Ok(Vec::new());
    }

    let mut katas = Vec::new();

    let entries = fs::read_dir(exercises_path).context("failed to read exercises directory")?;

    for entry in entries {
        let entry = match entry {
            Ok(e) => e,
            Err(e) => {
                eprintln!("warning: failed to read directory entry: {}", e);
                continue;
            }
        };

        let path = entry.path();

        if !path.is_dir() {
            continue;
        }

        let manifest_path = path.join("manifest.toml");
        if !manifest_path.exists() {
            continue;
        }

        match load_kata_from_manifest(&manifest_path) {
            Ok(kata) => katas.push(kata),
            Err(e) => {
                eprintln!(
                    "warning: failed to load manifest at {}: {}",
                    manifest_path.display(),
                    e
                );
            }
        }
    }

    Ok(katas)
}

fn load_kata_from_manifest(manifest_path: &Path) -> Result<AvailableKata> {
    let content = fs::read_to_string(manifest_path).context("failed to read manifest file")?;

    let manifest: KataManifest =
        toml::from_str(&content).context("failed to parse manifest TOML")?;

    Ok(manifest.kata)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn test_load_available_katas_empty_dir() {
        let result = load_available_katas();
        assert!(result.is_ok());
    }

    #[test]
    fn test_load_kata_from_manifest() {
        let temp_dir = TempDir::new().unwrap();
        let manifest_path = temp_dir.path().join("manifest.toml");

        let manifest_content = r#"
[kata]
name = "test_kata"
category = "test_category"
base_difficulty = 3
description = "A test kata"
dependencies = ["kata1", "kata2"]
"#;

        fs::write(&manifest_path, manifest_content).unwrap();

        let kata = load_kata_from_manifest(&manifest_path).unwrap();

        assert_eq!(kata.name, "test_kata");
        assert_eq!(kata.category, "test_category");
        assert_eq!(kata.base_difficulty, 3);
        assert_eq!(kata.description, "A test kata");
        assert_eq!(kata.dependencies, vec!["kata1", "kata2"]);
    }

    #[test]
    fn test_load_kata_from_manifest_no_dependencies() {
        let temp_dir = TempDir::new().unwrap();
        let manifest_path = temp_dir.path().join("manifest.toml");

        let manifest_content = r#"
[kata]
name = "simple_kata"
category = "simple"
base_difficulty = 1
description = "Simple kata without dependencies"
"#;

        fs::write(&manifest_path, manifest_content).unwrap();

        let kata = load_kata_from_manifest(&manifest_path).unwrap();

        assert_eq!(kata.name, "simple_kata");
        assert_eq!(kata.dependencies.len(), 0);
    }

    #[test]
    fn test_load_kata_from_manifest_invalid_toml() {
        let temp_dir = TempDir::new().unwrap();
        let manifest_path = temp_dir.path().join("manifest.toml");

        fs::write(&manifest_path, "invalid toml content {{{").unwrap();

        let result = load_kata_from_manifest(&manifest_path);
        assert!(result.is_err());
    }
}
