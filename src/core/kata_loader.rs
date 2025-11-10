//! Kata loader for scanning and loading kata manifests from the exercises directory.

use anyhow::{Context, Result};
use serde::Deserialize;
use std::fs;
use std::path::Path;

/// Represents a kata found in the exercises directory
#[derive(Debug, Clone, Deserialize)]
pub struct AvailableKata {
    pub name: String,
    #[serde(default)]
    pub category: String,
    #[serde(default)]
    pub tags: Vec<String>,
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

/// Extracts all unique categories from a list of katas.
///
/// # Arguments
///
/// * `katas` - Slice of available katas
///
/// Extracts all unique tags from katas (including both category and tags fields).
///
/// # Returns
///
/// Sorted vector of unique category/tag strings
pub fn get_unique_categories(katas: &[AvailableKata]) -> Vec<String> {
    use std::collections::HashSet;

    let mut all_tags = HashSet::new();

    for kata in katas {
        // Add the primary category
        if !kata.category.is_empty() {
            all_tags.insert(kata.category.clone());
        }

        // Add all tags
        for tag in &kata.tags {
            if !tag.is_empty() {
                all_tags.insert(tag.clone());
            }
        }
    }

    let mut categories: Vec<String> = all_tags.into_iter().collect();
    categories.sort();
    categories
}

fn load_kata_from_manifest(manifest_path: &Path) -> Result<AvailableKata> {
    let content = fs::read_to_string(manifest_path).context("failed to read manifest file")?;

    let manifest: KataManifest =
        toml::from_str(&content).context("failed to parse manifest TOML")?;

    Ok(manifest.kata)
}

/// Result of a kata reimport operation
#[derive(Debug)]
pub struct ReimportResult {
    pub added: usize,
    pub updated: usize,
    pub deleted: usize,
}

/// Reimports katas from manifest files into the database.
///
/// Scans katas/exercises/ directory and:
/// - Updates existing katas (description, difficulty, category) while preserving SM-2 state
/// - Adds new katas found in the directory
/// - Optionally deletes katas not found in the directory (if prune=true)
///
/// # Arguments
///
/// * `repo` - The kata repository
/// * `prune` - If true, delete katas from DB that aren't in exercises/ directory
///
/// # Returns
///
/// ReimportResult with counts of added, updated, and deleted katas
pub fn reimport_katas(
    repo: &crate::db::repo::KataRepository,
    prune: bool,
) -> Result<ReimportResult> {
    use crate::db::repo::NewKata;
    use std::collections::{HashMap, HashSet};

    // load all katas from disk
    let available_katas = load_available_katas()?;
    let available_names: HashSet<String> = available_katas.iter().map(|k| k.name.clone()).collect();

    // load all katas from database
    let existing_katas = repo.get_all_katas()?;
    let existing_by_name: std::collections::HashMap<String, crate::db::repo::Kata> = existing_katas
        .into_iter()
        .map(|k| (k.name.clone(), k))
        .collect();
    let mut name_to_id: HashMap<String, i64> = existing_by_name
        .iter()
        .map(|(name, kata)| (name.clone(), kata.id))
        .collect();

    let mut added = 0;
    let mut updated = 0;

    // process each available kata
    for available in &available_katas {
        // Determine tags: prefer tags array, fall back to category if tags is empty
        let tags = if !available.tags.is_empty() {
            available.tags.clone()
        } else if !available.category.is_empty() {
            vec![available.category.clone()]
        } else {
            vec![]
        };

        if let Some(existing) = existing_by_name.get(&available.name) {
            // kata exists - update metadata if changed (preserve SM-2 state)
            let needs_update = existing.description != available.description
                || existing.category != available.category
                || existing.base_difficulty != available.base_difficulty;

            if needs_update {
                repo.update_kata_metadata(
                    existing.id,
                    &available.description,
                    &available.category,
                    available.base_difficulty,
                )?;
                updated += 1;
            }

            // Update tags
            repo.set_kata_tags(existing.id, &tags)?;

            name_to_id.insert(available.name.clone(), existing.id);
        } else {
            // new kata - create it
            let new_kata = NewKata {
                name: available.name.clone(),
                category: available.category.clone(),
                description: available.description.clone(),
                base_difficulty: available.base_difficulty,
                parent_kata_id: None,
                variation_params: None,
            };

            let kata_id = repo.create_kata(&new_kata, chrono::Utc::now())?;

            // Set tags for new kata
            repo.set_kata_tags(kata_id, &tags)?;

            name_to_id.insert(available.name.clone(), kata_id);
            added += 1;
        }
    }

    // synchronize dependency edges with manifests
    for available in &available_katas {
        let Some(&kata_id) = name_to_id.get(&available.name) else {
            continue;
        };

        let mut dependency_ids = Vec::new();
        let mut seen = HashSet::new();

        for dep_name in &available.dependencies {
            if dep_name == &available.name {
                eprintln!(
                    "warning: kata '{}' lists itself as a dependency; skipping self edge",
                    available.name
                );
                continue;
            }

            match name_to_id.get(dep_name) {
                Some(&dep_id) => {
                    if seen.insert(dep_id) {
                        dependency_ids.push(dep_id);
                    }
                }
                None => {
                    eprintln!(
                        "warning: dependency '{}' for kata '{}' was not found during reimport",
                        dep_name, available.name
                    );
                }
            }
        }

        repo.replace_dependencies(kata_id, &dependency_ids)?;
    }

    // optionally delete katas not in exercises/ directory
    let mut deleted = 0;
    if prune {
        for existing_name in existing_by_name.keys() {
            if !available_names.contains(existing_name) {
                if let Some(kata) = existing_by_name.get(existing_name) {
                    repo.delete_kata(kata.id)?;
                    deleted += 1;
                }
            }
        }
    }

    Ok(ReimportResult {
        added,
        updated,
        deleted,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::repo::KataRepository;
    use std::collections::HashMap;
    use std::fs;
    use std::path::Path;
    use tempfile::TempDir;

    struct CwdGuard {
        original: std::path::PathBuf,
    }

    impl CwdGuard {
        fn change_to(path: &Path) -> Self {
            let original = std::env::current_dir().unwrap();
            std::env::set_current_dir(path).unwrap();
            Self { original }
        }
    }

    impl Drop for CwdGuard {
        fn drop(&mut self) {
            let _ = std::env::set_current_dir(&self.original);
        }
    }

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
tags = ["tag1", "tag2"]
base_difficulty = 3
description = "A test kata"
dependencies = ["kata1", "kata2"]
"#;

        fs::write(&manifest_path, manifest_content).unwrap();

        let kata = load_kata_from_manifest(&manifest_path).unwrap();

        assert_eq!(kata.name, "test_kata");
        assert_eq!(kata.category, "test_category");
        assert_eq!(kata.tags, vec!["tag1", "tag2"]);
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

    #[test]
    fn test_reimport_katas_syncs_dependencies() {
        let temp_dir = TempDir::new().unwrap();
        let _guard = CwdGuard::change_to(temp_dir.path());

        let exercises_dir = temp_dir.path().join("katas/exercises");
        fs::create_dir_all(exercises_dir.join("intro")).unwrap();
        fs::create_dir_all(exercises_dir.join("advanced")).unwrap();

        fs::write(
            exercises_dir.join("intro/manifest.toml"),
            r#"
[kata]
name = "intro"
category = "basics"
base_difficulty = 1
description = "Intro kata"
dependencies = []
"#,
        )
        .unwrap();

        fs::write(
            exercises_dir.join("advanced/manifest.toml"),
            r#"
[kata]
name = "advanced"
category = "advanced"
base_difficulty = 3
description = "Advanced kata"
dependencies = ["intro"]
"#,
        )
        .unwrap();

        let repo = KataRepository::new_in_memory().unwrap();
        repo.run_migrations().unwrap();

        let result = reimport_katas(&repo, false).unwrap();
        assert_eq!(result.added, 2);

        let intro_id = repo.get_kata_by_name("intro").unwrap().unwrap().id;
        let advanced_id = repo.get_kata_by_name("advanced").unwrap().unwrap().id;

        let graph = repo.load_dependency_graph().unwrap();
        let counts = HashMap::new();
        let blocking = graph.get_blocking_dependencies(advanced_id, &counts);

        assert_eq!(blocking.len(), 1);
        assert_eq!(blocking[0].0, intro_id);
    }
}
