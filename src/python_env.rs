//! Python environment bootstrap and management.
//!
//! This module handles the setup and initialization of the Python environment
//! used for running kata exercises. It checks for the `uv` package manager,
//! ensures the virtual environment is created, and provides access to the
//! Python interpreter path.
//!
//! # Workflow
//!
//! 1. Check that `uv` is installed on the system
//! 2. Verify that `katas/.venv/` exists, or create it via `uv sync`
//! 3. Resolve the interpreter path at `katas/.venv/bin/python`
//! 4. Cache the interpreter path for subsequent use
//!
//! # Examples
//!
//! ```no_run
//! use kata_sr::python_env::PythonEnv;
//!
//! let env = PythonEnv::setup()?;
//! println!("Python interpreter: {:?}", env.interpreter_path());
//! # Ok::<(), String>(())
//! ```

use std::path::{Path, PathBuf};
use std::process::Command;

/// Manages the Python virtual environment for running kata exercises.
///
/// This struct encapsulates the Python environment setup process and provides
/// access to the Python interpreter path. It ensures that the required
/// dependencies are installed via `uv` before any kata execution.
#[derive(Debug, Clone)]
pub struct PythonEnv {
    interpreter_path: PathBuf,
    venv_path: PathBuf,
}

impl PythonEnv {
    /// Sets up the Python environment and returns a configured `PythonEnv`.
    ///
    /// This method performs the following steps:
    ///
    /// 1. Checks if `uv` is installed by running `which uv`
    /// 2. Checks if `katas/.venv/` directory exists
    /// 3. If the venv doesn't exist, runs `uv sync --directory katas/` to create it
    /// 4. Resolves and validates the interpreter path at `katas/.venv/bin/python`
    /// 5. Returns a `PythonEnv` with the cached interpreter path
    ///
    /// # Errors
    ///
    /// Returns an error with a helpful message if:
    /// - `uv` is not installed
    /// - `uv sync` fails to create the virtual environment
    /// - The Python interpreter is not found in the expected location
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use kata_sr::python_env::PythonEnv;
    /// let env = PythonEnv::setup()?;
    /// println!("Environment ready at: {:?}", env.interpreter_path());
    /// # Ok::<(), String>(())
    /// ```
    pub fn setup() -> Result<Self, String> {
        // check that uv is installed
        let uv_check = Command::new("which")
            .arg("uv")
            .output()
            .map_err(|e| format!("Failed to check for uv: {}", e))?;

        if !uv_check.status.success() {
            return Err(
                "uv not found. Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh"
                    .to_string(),
            );
        }

        // check if venv exists
        let venv_path = Path::new("katas/.venv");
        if !venv_path.exists() {
            println!("Virtual environment not found. Setting up Python environment...");

            let sync_status = Command::new("uv")
                .args(["sync", "--directory", "katas/"])
                .status()
                .map_err(|e| format!("Failed to run uv sync: {}", e))?;

            if !sync_status.success() {
                return Err(
                    "uv sync failed. Check that katas/pyproject.toml exists and is valid."
                        .to_string(),
                );
            }

            println!("Python environment setup complete.");
        }

        // resolve interpreter path
        let interpreter = venv_path.join("bin/python");
        if !interpreter.exists() {
            return Err(format!(
                "Python interpreter not found at expected path: {}",
                interpreter.display()
            ));
        }

        Ok(Self {
            interpreter_path: interpreter,
            venv_path: venv_path.to_path_buf(),
        })
    }

    /// Returns the path to the Python interpreter in the virtual environment.
    ///
    /// This path can be used to execute Python scripts with the kata dependencies
    /// available.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use kata_sr::python_env::PythonEnv;
    /// # let env = PythonEnv::setup()?;
    /// let python_path = env.interpreter_path();
    /// println!("Python at: {}", python_path.display());
    /// # Ok::<(), String>(())
    /// ```
    pub fn interpreter_path(&self) -> &Path {
        &self.interpreter_path
    }

    /// Returns the root of the managed virtual environment.
    pub fn venv_path(&self) -> &Path {
        &self.venv_path
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_setup_checks_uv() {
        // this test validates that setup performs the uv check
        // actual setup depends on system environment, so we just verify
        // the error message when uv is not found would be helpful
        let result = PythonEnv::setup();

        // if uv is not installed, we should get a helpful error
        if let Err(e) = result {
            assert!(
                e.contains("uv not found") || e.contains("Python environment"),
                "Error message should be helpful: {}",
                e
            );
        }
    }

    #[test]
    fn test_interpreter_path_getter() {
        // create a mock PythonEnv to test the getter
        let env = PythonEnv {
            interpreter_path: PathBuf::from("katas/.venv/bin/python"),
            venv_path: PathBuf::from("katas/.venv"),
        };

        assert_eq!(env.interpreter_path(), Path::new("katas/.venv/bin/python"));
        assert_eq!(env.venv_path(), Path::new("katas/.venv"));
    }
}
