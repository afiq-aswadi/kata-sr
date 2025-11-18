//! Course loader for scanning and loading course manifests from the courses directory.

use anyhow::{Context, Result};
use serde::Deserialize;
use std::fs;
use std::path::Path;

/// Represents a course found in the courses directory
#[derive(Debug, Clone, Deserialize)]
pub struct AvailableCourse {
    pub name: String,
    pub title: String,
    pub description: String,
    #[serde(default)]
    pub author: Option<String>,
    pub sections: Vec<CourseSection>,
}

/// Represents a section within a course
#[derive(Debug, Clone, Deserialize)]
pub struct CourseSection {
    pub title: String,
    pub html_file: String,
    #[serde(default)]
    pub exercise_kata: Option<String>,
}

/// Internal structure for deserializing the manifest TOML file
#[derive(Debug, Deserialize)]
struct CourseManifest {
    course: CourseMeta,
    sections: Vec<CourseSection>,
}

#[derive(Debug, Deserialize)]
struct CourseMeta {
    name: String,
    title: String,
    description: String,
    #[serde(default)]
    author: Option<String>,
}

/// Scans the courses/ directory and loads all available courses from manifest files.
///
/// For each subdirectory in courses/, attempts to read and parse course.toml.
/// Skips directories without manifests or with invalid manifests rather than failing.
///
/// # Returns
/// Vector of AvailableCourse structs, one per valid course found.
/// Returns empty vector if courses directory doesn't exist.
pub fn load_available_courses() -> Result<Vec<AvailableCourse>> {
    let courses_path = Path::new("courses");

    if !courses_path.exists() {
        return Ok(Vec::new());
    }

    let mut courses = Vec::new();

    let entries = fs::read_dir(courses_path).context("failed to read courses directory")?;

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

        let manifest_path = path.join("course.toml");
        if !manifest_path.exists() {
            continue;
        }

        match load_course_from_manifest(&manifest_path) {
            Ok(course) => courses.push(course),
            Err(e) => {
                eprintln!(
                    "warning: failed to load course manifest at {}: {}",
                    manifest_path.display(),
                    e
                );
            }
        }
    }

    Ok(courses)
}

/// Loads a single course from a manifest.toml file.
///
/// # Arguments
///
/// * `manifest_path` - Path to the course.toml file
///
/// # Returns
///
/// AvailableCourse struct parsed from TOML
fn load_course_from_manifest(manifest_path: &Path) -> Result<AvailableCourse> {
    let toml_str = fs::read_to_string(manifest_path)
        .with_context(|| format!("failed to read {}", manifest_path.display()))?;

    let manifest: CourseManifest = toml::from_str(&toml_str)
        .with_context(|| format!("failed to parse TOML from {}", manifest_path.display()))?;

    Ok(AvailableCourse {
        name: manifest.course.name,
        title: manifest.course.title,
        description: manifest.course.description,
        author: manifest.course.author,
        sections: manifest.sections,
    })
}

/// Imports a course and its sections into the database.
///
/// # Arguments
///
/// * `repo` - Database repository
/// * `course_dir` - Path to the course directory (e.g., courses/intro_to_transformers)
///
/// # Returns
///
/// The database ID of the created course
pub fn import_course_to_db(
    repo: &crate::db::repo::KataRepository,
    course_dir: &Path,
) -> Result<i64> {
    let manifest_path = course_dir.join("course.toml");
    let available_course = load_course_from_manifest(&manifest_path)?;

    // Create course in database
    let new_course = crate::db::repo::NewCourse {
        name: available_course.name.clone(),
        title: available_course.title,
        description: available_course.description,
        author: available_course.author,
    };

    let course_id = repo.create_course(&new_course)?;

    // Import sections
    for (order_num, section) in available_course.sections.iter().enumerate() {
        // Build relative path to HTML file (relative to course dir)
        let html_path = format!(
            "courses/{}/{}",
            available_course.name,
            section.html_file
        );

        let new_section = crate::db::repo::NewCourseSection {
            course_id,
            order_num: order_num as i32,
            title: section.title.clone(),
            html_path,
            exercise_kata_name: section.exercise_kata.clone(),
        };

        repo.create_course_section(&new_section)?;
    }

    Ok(course_id)
}

/// Imports all available courses into the database.
///
/// Skips courses that already exist (based on name).
///
/// # Returns
///
/// Number of courses imported
pub fn import_all_courses(repo: &crate::db::repo::KataRepository) -> Result<usize> {
    let available_courses = load_available_courses()?;
    let mut imported_count = 0;

    for course in available_courses {
        // Check if course already exists
        if repo.get_course_by_name(&course.name)?.is_some() {
            eprintln!("skipping course '{}' (already exists)", course.name);
            continue;
        }

        let course_dir = Path::new("courses").join(&course.name);
        match import_course_to_db(repo, &course_dir) {
            Ok(_) => {
                eprintln!("imported course '{}'", course.name);
                imported_count += 1;
            }
            Err(e) => {
                eprintln!("failed to import course '{}': {}", course.name, e);
            }
        }
    }

    Ok(imported_count)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_available_courses() {
        // This test just verifies the function doesn't panic when courses dir doesn't exist
        let courses = load_available_courses();
        assert!(courses.is_ok());
    }
}
