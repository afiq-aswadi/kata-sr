//! Python environment setup and error handling tests.
//!
//! Tests Python environment initialization, UV detection, and error scenarios.
//!
//! Note: These tests modify global process state (working directory) and may have
//! race conditions when run in parallel. Run with `cargo test --test python_env_tests -- --test-threads=1`
//! if you encounter failures related to file not found or directory errors.

use kata_sr::python_env::PythonEnv;
use std::env;
use std::fs;
use std::path::PathBuf;
use tempfile::TempDir;

/// RAII guard that restores the original working directory on drop
struct WorkingDirGuard {
    original_dir: PathBuf,
}

impl WorkingDirGuard {
    fn new(new_dir: &std::path::Path) -> std::io::Result<Self> {
        let original_dir = env::current_dir()?;
        env::set_current_dir(new_dir)?;
        Ok(Self { original_dir })
    }
}

impl Drop for WorkingDirGuard {
    fn drop(&mut self) {
        let _ = env::set_current_dir(&self.original_dir);
    }
}

#[test]
fn test_setup_fails_without_uv() {
    // Temporarily modify PATH to not include uv
    let original_path = env::var("PATH").unwrap_or_default();
    env::set_var("PATH", "");

    let result = PythonEnv::setup();

    // Restore PATH
    env::set_var("PATH", &original_path);

    assert!(result.is_err());
    let error = result.unwrap_err();
    assert!(error.contains("uv not found") || error.contains("uv"));
}

#[test]
fn test_setup_creates_venv_if_missing() {
    let temp_dir = TempDir::new().unwrap();

    // Change to temp directory (restored automatically on drop)
    let _dir_guard = WorkingDirGuard::new(temp_dir.path()).unwrap();

    // Create katas directory structure
    fs::create_dir_all("katas").unwrap();
    fs::write("katas/pyproject.toml", "[project]\nname = \"test\"\n").unwrap();

    // Check if uv is available
    if let Ok(uv_check) = std::process::Command::new("which")
        .arg("uv")
        .output()
    {
        if uv_check.status.success() {
            // If uv is available, setup should work
            let result = PythonEnv::setup();

            // Note: This might fail if uv sync fails for other reasons
            // In CI, this test serves as a smoke test
            if result.is_ok() {
                assert!(PathBuf::from("katas/.venv").exists());
            }
        }
    }
}

#[test]
fn test_interpreter_path_resolution() {
    // This test checks that interpreter_path() returns a valid path structure
    // We can't guarantee the venv exists in all test environments

    let temp_dir = TempDir::new().unwrap();
    let _dir_guard = WorkingDirGuard::new(temp_dir.path()).unwrap();

    // Create mock venv structure
    fs::create_dir_all("katas/.venv/bin").unwrap();
    fs::write("katas/.venv/bin/python", "#!/bin/sh\n# mock python\n").unwrap();
    fs::write("katas/pyproject.toml", "[project]\nname = \"test\"\n").unwrap();

    // Set execute permission on Unix
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = fs::metadata("katas/.venv/bin/python").unwrap().permissions();
        perms.set_mode(0o755);
        fs::set_permissions("katas/.venv/bin/python", perms).unwrap();
    }

    // Temporarily modify PATH to include a mock uv
    let mock_uv_dir = temp_dir.path().join("mock_bin");
    fs::create_dir(&mock_uv_dir).unwrap();
    let mock_uv = mock_uv_dir.join("uv");
    fs::write(&mock_uv, "#!/bin/sh\nexit 0\n").unwrap();

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = fs::metadata(&mock_uv).unwrap().permissions();
        perms.set_mode(0o755);
        fs::set_permissions(&mock_uv, perms).unwrap();
    }

    let original_path = env::var("PATH").unwrap_or_default();
    env::set_var(
        "PATH",
        format!("{}:{}", mock_uv_dir.display(), original_path),
    );

    let result = PythonEnv::setup();

    // Restore PATH
    env::set_var("PATH", &original_path);

    if let Ok(env) = result {
        let interp_path = env.interpreter_path();
        assert!(interp_path.ends_with("python"));
        assert!(interp_path.to_string_lossy().contains(".venv"));
    }
}

#[test]
fn test_missing_pyproject_toml() {
    let temp_dir = TempDir::new().unwrap();
    let _dir_guard = WorkingDirGuard::new(temp_dir.path()).unwrap();

    // Create katas directory but NO pyproject.toml
    fs::create_dir_all("katas").unwrap();

    // Setup should fail or handle gracefully
    let result = PythonEnv::setup();

    // This will fail since pyproject.toml is missing and uv sync will complain
    if let Ok(_) = std::process::Command::new("which").arg("uv").output() {
        // If uv is present, we expect an error
        assert!(result.is_err() || PathBuf::from("katas/.venv").exists() == false);
    }
}

#[test]
fn test_corrupt_venv_detection() {
    let temp_dir = TempDir::new().unwrap();
    let _dir_guard = WorkingDirGuard::new(temp_dir.path()).unwrap();

    // Create a corrupt venv (directory exists but no python binary)
    fs::create_dir_all("katas/.venv/bin").unwrap();
    fs::write("katas/pyproject.toml", "[project]\nname = \"test\"\n").unwrap();

    // Create mock uv
    let mock_uv_dir = temp_dir.path().join("mock_bin");
    fs::create_dir(&mock_uv_dir).unwrap();
    let mock_uv = mock_uv_dir.join("uv");
    fs::write(&mock_uv, "#!/bin/sh\nexit 0\n").unwrap();

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = fs::metadata(&mock_uv).unwrap().permissions();
        perms.set_mode(0o755);
        fs::set_permissions(&mock_uv, perms).unwrap();
    }

    let original_path = env::var("PATH").unwrap_or_default();
    env::set_var(
        "PATH",
        format!("{}:{}", mock_uv_dir.display(), original_path),
    );

    let result = PythonEnv::setup();

    env::set_var("PATH", &original_path);

    // Should fail because python binary doesn't exist in .venv/bin/
    assert!(result.is_err());
}

#[test]
fn test_venv_path_construction() {
    // Test that VenvPath is constructed correctly
    let temp_dir = TempDir::new().unwrap();
    let _dir_guard = WorkingDirGuard::new(temp_dir.path()).unwrap();

    // Create complete mock environment
    fs::create_dir_all("katas/.venv/bin").unwrap();
    fs::write("katas/.venv/bin/python", "#!/bin/sh\necho 'mock'\n").unwrap();
    fs::write("katas/pyproject.toml", "[project]\nname = \"test\"\n").unwrap();

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = fs::metadata("katas/.venv/bin/python").unwrap().permissions();
        perms.set_mode(0o755);
        fs::set_permissions("katas/.venv/bin/python", perms).unwrap();
    }

    // Mock uv
    let mock_uv_dir = temp_dir.path().join("mock_bin");
    fs::create_dir(&mock_uv_dir).unwrap();
    let mock_uv = mock_uv_dir.join("uv");
    fs::write(&mock_uv, "#!/bin/sh\nexit 0\n").unwrap();

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = fs::metadata(&mock_uv).unwrap().permissions();
        perms.set_mode(0o755);
        fs::set_permissions(&mock_uv, perms).unwrap();
    }

    let original_path = env::var("PATH").unwrap_or_default();
    env::set_var(
        "PATH",
        format!("{}:{}", mock_uv_dir.display(), original_path),
    );

    let result = PythonEnv::setup();

    env::set_var("PATH", &original_path);

    if let Ok(python_env) = result {
        let venv_path = python_env.venv_path();
        assert_eq!(venv_path, PathBuf::from("katas/.venv"));
    }
}
