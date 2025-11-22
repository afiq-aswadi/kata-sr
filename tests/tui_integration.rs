//! TUI integration tests with Agent 4's katas.
//!
//! Tests that TUI components work correctly with real kata data:
//! - Dashboard loads katas and displays stats
//! - PracticeScreen copies templates to temp directory
//! - Python runner executes tests and returns results
//!
//! Note: Python runner tests require running from project root due to relative paths
//! in the runner implementation. Tests will be skipped if Python environment is not available.
//!
//! Note: Some tests may have race conditions when run in parallel due to shared /tmp files.
//! Run with `cargo test -- --test-threads=1` if you encounter failures.

use chrono::{Duration, Utc};
use kata_sr::config::EditorConfig;
use kata_sr::db::repo::{KataRepository, NewKata};
use kata_sr::runner::python_runner::run_python_tests;
use kata_sr::tui::dashboard::Dashboard;
use std::fs;
use std::path::PathBuf;

fn python_runner_available() -> bool {
    // check if we can actually run the Python command
    // the python_runner uses a relative path that changes with current_dir
    // so we need to test if it actually works
    use std::process::Command;

    Command::new("katas/.venv/bin/python")
        .args(["--version"])
        .current_dir("katas")
        .output()
        .is_ok()
}

fn setup_repo_with_katas() -> KataRepository {
    let repo = KataRepository::new_in_memory().unwrap();
    repo.run_migrations().unwrap();

    // create test katas matching Agent 4's exercises
    let katas = vec![
        NewKata {
            name: "mlp".to_string(),
            category: "neural_networks".to_string(),
            description: "Implement a multi-layer perceptron (MLP) with ReLU activations."
                .to_string(),
            base_difficulty: 2,
            parent_kata_id: None,
            variation_params: None,
        },
        NewKata {
            name: "softmax".to_string(),
            category: "fundamentals".to_string(),
            description: "Implement the softmax function with numerical stability.".to_string(),
            base_difficulty: 1,
            parent_kata_id: None,
            variation_params: None,
        },
        NewKata {
            name: "dfs_bfs".to_string(),
            category: "graphs".to_string(),
            description: "Implement depth-first search (DFS) and breadth-first search (BFS)."
                .to_string(),
            base_difficulty: 2,
            parent_kata_id: None,
            variation_params: None,
        },
    ];

    for kata in katas {
        repo.create_kata(&kata, Utc::now()).unwrap();
    }

    repo
}

#[test]
fn test_dashboard_loads_with_katas() {
    let repo = setup_repo_with_katas();

    // all katas should be due (never reviewed)
    let dashboard = Dashboard::load(&repo, 90).unwrap();

    assert_eq!(dashboard.katas_due.len(), 3);
    assert_eq!(dashboard.stats.streak_days, 0);
    assert_eq!(dashboard.stats.total_reviews_today, 0);
    assert_eq!(dashboard.stats.success_rate_7d, 0.0);

    // verify kata names are correct
    let kata_names: Vec<&str> = dashboard
        .katas_due
        .iter()
        .map(|k| k.name.as_str())
        .collect();
    assert!(kata_names.contains(&"mlp"));
    assert!(kata_names.contains(&"softmax"));
    assert!(kata_names.contains(&"dfs_bfs"));
}

#[test]
fn test_dashboard_filters_katas_by_due_date() {
    let repo = setup_repo_with_katas();

    // set one kata to be reviewed in the future
    let katas = repo.get_all_katas().unwrap();
    let kata_id = katas[0].id;

    let mut card = katas[0].fsrs_card();
    let params = kata_sr::core::fsrs::FsrsParams::default();
    card.schedule(kata_sr::core::fsrs::Rating::Good, &params, Utc::now());

    let future_review = Utc::now() + Duration::days(5);
    repo.update_kata_after_fsrs_review(kata_id, &card, future_review, Utc::now())
        .unwrap();

    // dashboard should only show 2 katas due
    let dashboard = Dashboard::load(&repo, 90).unwrap();
    assert_eq!(dashboard.katas_due.len(), 2);

    // the reviewed kata should not be in due list
    let due_names: Vec<&str> = dashboard
        .katas_due
        .iter()
        .map(|k| k.name.as_str())
        .collect();
    assert!(!due_names.contains(&katas[0].name.as_str()));
}

#[test]
fn test_dashboard_shows_stats_after_reviews() {
    let repo = setup_repo_with_katas();

    let katas = repo.get_all_katas().unwrap();

    // create sessions for today
    for kata in &katas {
        let session = kata_sr::db::repo::NewSession {
            kata_id: kata.id,
            started_at: Utc::now(),
            completed_at: Some(Utc::now()),
            test_results_json: None,
            num_passed: Some(5),
            num_failed: Some(0),
            num_skipped: Some(0),
            duration_ms: Some(1000),
            quality_rating: Some(3), // Good (FSRS)
            code_attempt: None,
        };
        repo.create_session(&session).unwrap();
    }

    let dashboard = Dashboard::load(&repo, 90).unwrap();

    assert_eq!(dashboard.stats.streak_days, 1);
    assert_eq!(dashboard.stats.total_reviews_today, 3);
    assert_eq!(dashboard.stats.success_rate_7d, 1.0);
}

#[test]
fn test_practice_screen_copies_template() {
    use tempfile::TempDir;

    let repo = setup_repo_with_katas();
    let katas = repo.get_all_katas().unwrap();
    let mlp_kata = katas.iter().find(|k| k.name == "mlp").unwrap();

    // Use isolated temp directory
    let temp_dir = TempDir::new().unwrap();

    // create practice screen with custom temp directory
    let editor_config = EditorConfig::default();
    let _practice = kata_sr::tui::practice::PracticeScreen::new_with_temp_dir(
        mlp_kata.clone(),
        editor_config,
        Some(temp_dir.path().to_path_buf()),
    )
    .unwrap();

    // verify template was copied to temp directory
    let template_path = temp_dir.path().join(format!("kata_{}.py", mlp_kata.id));
    assert!(template_path.exists());

    // verify template contains some content
    let content = fs::read_to_string(&template_path).unwrap();
    assert!(!content.is_empty());
    assert!(content.contains("mlp") || content.contains("MLP"));

    // tempdir automatically cleans up
}

#[test]
fn test_practice_screen_copies_correct_template_for_each_kata() {
    use tempfile::TempDir;

    let repo = setup_repo_with_katas();
    let katas = repo.get_all_katas().unwrap();

    // test softmax and dfs_bfs
    let test_katas: Vec<_> = katas
        .iter()
        .filter(|k| k.name == "softmax" || k.name == "dfs_bfs")
        .collect();

    for kata in test_katas {
        // Each kata gets its own isolated temp directory
        let temp_dir = TempDir::new().unwrap();

        let editor_config = EditorConfig::default();
        let _practice = kata_sr::tui::practice::PracticeScreen::new_with_temp_dir(
            kata.clone(),
            editor_config,
            Some(temp_dir.path().to_path_buf()),
        )
        .unwrap();

        let template_path = temp_dir.path().join(format!("kata_{}.py", kata.id));
        assert!(
            template_path.exists(),
            "Template should exist for kata {}",
            kata.name
        );

        // read content
        let content = fs::read_to_string(&template_path).unwrap();
        assert!(!content.is_empty(), "Template should not be empty");

        // tempdir automatically cleans up
    }
}

#[test]
fn test_python_runner_executes_with_reference() {
    if !python_runner_available() {
        eprintln!("Skipping test: Python environment not available");
        return;
    }

    // test with softmax kata using reference.py as the template
    // this ensures tests will pass

    let kata_name = "softmax";
    let kata_dir = PathBuf::from("katas/exercises").join(kata_name);
    let reference_path = kata_dir.join("reference.py");

    if !reference_path.exists() {
        panic!("Reference file not found: {:?}", reference_path);
    }

    // copy reference to a temp file
    let template_path = PathBuf::from("/tmp/test_kata_reference.py");
    fs::copy(&reference_path, &template_path).unwrap();

    // run tests
    let results = run_python_tests(kata_name, &template_path);

    // verify results
    assert!(
        results.passed,
        "Tests should pass with reference implementation. Results: {:?}",
        results
    );
    assert!(results.num_passed > 0, "Should have passing tests");
    assert_eq!(results.num_failed, 0, "Should have no failures");

    // cleanup
    fs::remove_file(&template_path).ok();
}

#[test]
fn test_python_runner_executes_with_template() {
    if !python_runner_available() {
        eprintln!("Skipping test: Python environment not available");
        return;
    }

    // test with template.py which has TODOs
    // this should fail

    let kata_name = "softmax";
    let kata_dir = PathBuf::from("katas/exercises").join(kata_name);
    let template_source = kata_dir.join("template.py");

    if !template_source.exists() {
        panic!("Template file not found: {:?}", template_source);
    }

    // copy template to a temp file
    let template_path = PathBuf::from("/tmp/test_kata_template.py");
    fs::copy(&template_source, &template_path).unwrap();

    // run tests
    let results = run_python_tests(kata_name, &template_path);

    // template has TODOs, so tests should fail
    assert!(!results.passed, "Tests should fail with template TODOs");
    assert!(results.num_failed > 0, "Should have failing tests");

    // cleanup
    fs::remove_file(&template_path).ok();
}

#[test]
fn test_python_runner_with_mlp_kata() {
    if !python_runner_available() {
        eprintln!("Skipping test: Python environment not available");
        return;
    }

    let kata_name = "mlp";
    let kata_dir = PathBuf::from("katas/exercises").join(kata_name);
    let reference_path = kata_dir.join("reference.py");

    if !reference_path.exists() {
        panic!("Reference file not found: {:?}", reference_path);
    }

    let template_path = PathBuf::from("/tmp/test_mlp_reference.py");
    fs::copy(&reference_path, &template_path).unwrap();

    let results = run_python_tests(kata_name, &template_path);

    assert!(
        results.passed,
        "MLP tests should pass with reference implementation. Results: {:?}",
        results
    );
    assert!(results.num_passed > 0);
    assert_eq!(results.num_failed, 0);
    assert!(results.duration_ms > 0, "Should track duration");

    // verify results structure
    assert!(!results.results.is_empty(), "Should have test results");
    for result in &results.results {
        assert_eq!(result.status, "passed");
        assert!(!result.test_name.is_empty());
    }

    fs::remove_file(&template_path).ok();
}

#[test]
fn test_python_runner_with_dfs_bfs_kata() {
    if !python_runner_available() {
        eprintln!("Skipping test: Python environment not available");
        return;
    }

    let kata_name = "dfs_bfs";
    let kata_dir = PathBuf::from("katas/exercises").join(kata_name);
    let reference_path = kata_dir.join("reference.py");

    if !reference_path.exists() {
        panic!("Reference file not found: {:?}", reference_path);
    }

    let template_path = PathBuf::from("/tmp/test_dfs_bfs_reference.py");
    fs::copy(&reference_path, &template_path).unwrap();

    let results = run_python_tests(kata_name, &template_path);

    assert!(
        results.passed,
        "DFS/BFS tests should pass with reference implementation. Results: {:?}",
        results
    );
    assert!(results.num_passed > 0);
    assert_eq!(results.num_failed, 0);

    fs::remove_file(&template_path).ok();
}

#[test]
fn test_python_runner_handles_nonexistent_kata() {
    if !python_runner_available() {
        eprintln!("Skipping test: Python environment not available");
        return;
    }

    let kata_name = "nonexistent_kata";
    let template_path = PathBuf::from("/tmp/test_nonexistent.py");

    // create a dummy template file
    fs::write(&template_path, "# dummy content").unwrap();

    let results = run_python_tests(kata_name, &template_path);

    // should fail because kata doesn't exist
    assert!(!results.passed);
    assert_eq!(results.num_passed, 0);

    fs::remove_file(&template_path).ok();
}

#[test]
fn test_python_runner_handles_missing_template() {
    if !python_runner_available() {
        eprintln!("Skipping test: Python environment not available");
        return;
    }

    let kata_name = "softmax";
    let template_path = PathBuf::from("/tmp/missing_template.py");

    // ensure template doesn't exist
    fs::remove_file(&template_path).ok();

    let results = run_python_tests(kata_name, &template_path);

    // should fail because template doesn't exist
    assert!(!results.passed);
    assert_eq!(results.num_passed, 0);
}

#[test]
fn test_end_to_end_kata_workflow() {
    use tempfile::TempDir;

    if !python_runner_available() {
        eprintln!("Skipping test: Python environment not available");
        return;
    }

    // simulate complete workflow: create kata -> load dashboard -> practice -> run tests
    let repo = setup_repo_with_katas();

    // load dashboard
    let dashboard = Dashboard::load(&repo, 90).unwrap();
    assert_eq!(dashboard.katas_due.len(), 3);

    // select first kata
    let kata = &dashboard.katas_due[0];

    // Use isolated temp directory
    let temp_dir = TempDir::new().unwrap();

    // create practice screen
    let editor_config = EditorConfig::default();
    let _practice = kata_sr::tui::practice::PracticeScreen::new_with_temp_dir(
        kata.clone(),
        editor_config,
        Some(temp_dir.path().to_path_buf()),
    )
    .unwrap();
    let template_path = temp_dir.path().join(format!("kata_{}.py", kata.id));
    assert!(template_path.exists());

    // simulate user filling in the code by copying reference
    let kata_dir = PathBuf::from("katas/exercises").join(&kata.name);
    let reference_path = kata_dir.join("reference.py");

    if reference_path.exists() {
        fs::copy(&reference_path, &template_path).unwrap();

        // run tests
        let results = run_python_tests(&kata.name, &template_path);
        assert!(results.passed);

        // create session
        let session = kata_sr::db::repo::NewSession {
            kata_id: kata.id,
            started_at: Utc::now(),
            completed_at: Some(Utc::now()),
            test_results_json: Some(format!(
                "{{\"passed\": {}, \"num_tests\": {}}}",
                results.passed, results.num_passed
            )),
            num_passed: Some(results.num_passed),
            num_failed: Some(results.num_failed),
            num_skipped: Some(results.num_skipped),
            duration_ms: Some(results.duration_ms),
            quality_rating: Some(3), // Good (FSRS)
            code_attempt: None,
        };
        repo.create_session(&session).unwrap();

        // update kata state
        let mut card = kata.fsrs_card();
        let params = kata_sr::core::fsrs::FsrsParams::default();
        card.schedule(kata_sr::core::fsrs::Rating::Good, &params, Utc::now());
        let next_review = Utc::now() + Duration::days(card.scheduled_days as i64);
        repo.update_kata_after_fsrs_review(kata.id, &card, next_review, Utc::now())
            .unwrap();

        // verify kata is no longer due
        let dashboard = Dashboard::load(&repo, 90).unwrap();
        assert_eq!(dashboard.katas_due.len(), 2);
        assert_eq!(dashboard.stats.total_reviews_today, 1);

        // tempdir automatically cleans up
    }
}
