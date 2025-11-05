use serde::Deserialize;
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Debug, Clone, Deserialize)]
pub struct TestResults {
    pub passed: bool,
    pub num_passed: i32,
    pub num_failed: i32,
    pub num_skipped: i32,
    pub duration_ms: i64,
    pub results: Vec<TestResult>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TestResult {
    pub test_name: String,
    pub status: String,
    pub output: String,
}

impl TestResults {
    pub fn error(message: String) -> Self {
        Self {
            passed: false,
            num_passed: 0,
            num_failed: 1,
            num_skipped: 0,
            duration_ms: 0,
            results: vec![TestResult {
                test_name: "runner_error".to_string(),
                status: "failed".to_string(),
                output: message,
            }],
        }
    }
}

pub fn run_python_tests(kata_id: &str, template_path: &Path) -> TestResults {
    let interpreter_path = resolve_interpreter_path();
    if !interpreter_path.exists() {
        return TestResults::error(format!(
            "Python interpreter not found at {}",
            interpreter_path.display()
        ));
    }

    let katas_dir = resolve_katas_dir();
    if !katas_dir.exists() {
        return TestResults::error(format!(
            "Katas directory not found at {}",
            katas_dir.display()
        ));
    }

    let template_str = match template_path.to_str() {
        Some(s) => s,
        None => {
            return TestResults::error(
                "Invalid template path: contains non-UTF8 characters".to_string(),
            )
        }
    };

    let output = Command::new(&interpreter_path)
        .args(["-m", "runner", kata_id, template_str])
        .current_dir(&katas_dir)
        .output();

    let output = match output {
        Ok(output) => output,
        Err(err) => return TestResults::error(format!("Failed to spawn Python process: {}", err)),
    };

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return TestResults::error(format!(
            "Python runner failed (exit code {:?}):\n{}",
            output.status.code(),
            stderr
        ));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);

    match serde_json::from_str(&stdout) {
        Ok(results) => results,
        Err(e) => TestResults::error(format!(
            "Failed to parse JSON output: {}\nOutput was:\n{}",
            e, stdout
        )),
    }
}

fn resolve_interpreter_path() -> PathBuf {
    std::env::var("KATA_SR_PYTHON")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("katas/.venv/bin/python"))
}

fn resolve_katas_dir() -> PathBuf {
    std::env::var("KATA_SR_KATAS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("katas"))
}
