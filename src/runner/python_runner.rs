use serde::Deserialize;
use std::path::Path;
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
    fn error(message: String) -> Self {
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
    let python_path = "katas/.venv/bin/python";

    let template_str = match template_path.to_str() {
        Some(s) => s,
        None => {
            return TestResults::error(
                "Invalid template path: contains non-UTF8 characters".to_string(),
            )
        }
    };

    let output = Command::new(python_path)
        .args(["-m", "runner", kata_id, template_str])
        .current_dir("katas")
        .output()
        .expect("Failed to spawn Python process");

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return TestResults::error(format!("Python runner failed:\n{}", stderr));
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
