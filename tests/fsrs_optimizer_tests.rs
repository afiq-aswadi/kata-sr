//! FSRS optimizer edge case and numerical stability tests.
//!
//! Tests optimizer behavior with insufficient data, extreme parameters, and edge cases.

use chrono::Utc;
use kata_sr::core::fsrs::{FsrsCard, FsrsParams, Rating};
use kata_sr::core::fsrs_optimizer::FsrsOptimizer;
use kata_sr::db::repo::{KataRepository, NewKata, NewSession};

fn setup_test_repo_with_sessions(num_sessions: usize, ratings: &[Rating]) -> KataRepository {
    let repo = KataRepository::new_in_memory().unwrap();
    repo.run_migrations().unwrap();

    let new_kata = NewKata {
        name: "optimizer_test".to_string(),
        category: "test".to_string(),
        description: "For optimizer testing".to_string(),
        base_difficulty: 3,
        parent_kata_id: None,
        variation_params: None,
    };

    let kata_id = repo.create_kata(&new_kata, Utc::now()).unwrap();

    // Create sessions with specified ratings
    let mut card = FsrsCard::new();
    let params = FsrsParams::default();

    for i in 0..num_sessions {
        let rating = ratings[i % ratings.len()];

        // Create session
        let session = NewSession {
            kata_id,
            started_at: Utc::now(),
            completed_at: Some(Utc::now()),
            test_results_json: None,
            num_passed: Some(if (rating as i32) >= (Rating::Hard as i32) { 5 } else { 0 }),
            num_failed: Some(if (rating as i32) >= (Rating::Hard as i32) { 0 } else { 5 }),
            num_skipped: Some(0),
            duration_ms: Some(1000),
            quality_rating: Some(rating as i32),
            code_attempt: None,
        };
        repo.create_session(&session).unwrap();

        // Update FSRS state
        card.schedule(rating, &params, Utc::now());
        let next_review = Utc::now() + chrono::Duration::days(card.scheduled_days as i64);
        repo.update_kata_after_fsrs_review(kata_id, &card, next_review, Utc::now())
            .unwrap();
    }

    repo
}

#[test]
fn test_optimizer_rejects_insufficient_data() {
    let optimizer = FsrsOptimizer::new();

    // Create repo with only 10 sessions (less than min_reviews of 50)
    let repo = setup_test_repo_with_sessions(10, &[Rating::Good, Rating::Good]);

    let result = optimizer.optimize(&repo);

    // Should fail due to insufficient data
    assert!(result.is_err());
    let error = result.unwrap_err();
    assert!(error.to_string().contains("insufficient") || error.to_string().contains("not enough"));
}

#[test]
fn test_optimizer_with_minimum_required_data() {
    let optimizer = FsrsOptimizer::with_config(0.01, 10, 20); // Lower min_reviews for testing

    // Create repo with exactly minimum sessions
    let repo = setup_test_repo_with_sessions(20, &[Rating::Good, Rating::Hard, Rating::Easy]);

    let result = optimizer.optimize(&repo);

    // Should succeed with minimum data
    assert!(result.is_ok());

    let optimized_params = result.unwrap();
    assert_eq!(optimized_params.w.len(), 19);

    // Parameters should be finite (no NaN or Infinity)
    for (i, &w) in optimized_params.w.iter().enumerate() {
        assert!(w.is_finite(), "Parameter w[{}] is not finite: {}", i, w);
    }
}

#[test]
fn test_optimizer_with_all_failing_reviews() {
    let optimizer = FsrsOptimizer::with_config(0.01, 10, 20);

    // All reviews fail (Rating::Again)
    let repo = setup_test_repo_with_sessions(25, &[Rating::Again]);

    let result = optimizer.optimize(&repo);

    if let Ok(params) = result {
        // Should still produce valid parameters
        assert_eq!(params.w.len(), 19);

        for &w in &params.w {
            assert!(w.is_finite());
        }

        // Parameters should reflect the poor performance
        // (difficulty parameters might be higher)
    }
}

#[test]
fn test_optimizer_with_all_passing_reviews() {
    let optimizer = FsrsOptimizer::with_config(0.01, 10, 20);

    // All reviews pass perfectly (Rating::Easy)
    let repo = setup_test_repo_with_sessions(25, &[Rating::Easy]);

    let result = optimizer.optimize(&repo);

    if let Ok(params) = result {
        // Should produce valid parameters
        assert_eq!(params.w.len(), 19);

        for &w in &params.w {
            assert!(w.is_finite());
        }
    }
}

#[test]
fn test_optimizer_with_mixed_performance() {
    let optimizer = FsrsOptimizer::with_config(0.01, 20, 30);

    // Realistic mixed performance
    let ratings = vec![
        Rating::Good,
        Rating::Good,
        Rating::Hard,
        Rating::Easy,
        Rating::Again,
        Rating::Good,
        Rating::Good,
        Rating::Easy,
    ];

    let repo = setup_test_repo_with_sessions(40, &ratings);

    let result = optimizer.optimize(&repo);
    assert!(result.is_ok());

    let params = result.unwrap();

    // Verify all parameters are reasonable
    for (i, &w) in params.w.iter().enumerate() {
        assert!(w.is_finite(), "Parameter w[{}] is not finite", i);
        // FSRS parameters typically in range [0, 100]
        assert!(w >= -10.0 && w <= 100.0, "Parameter w[{}] out of reasonable range: {}", i, w);
    }
}

#[test]
fn test_optimizer_detects_nan_in_parameters() {
    // This test ensures the optimizer doesn't produce NaN values
    let optimizer = FsrsOptimizer::with_config(0.01, 10, 20);

    let repo = setup_test_repo_with_sessions(25, &[Rating::Good, Rating::Hard]);

    let result = optimizer.optimize(&repo);

    if let Ok(params) = result {
        let has_nan = params.w.iter().any(|&w| w.is_nan());
        assert!(!has_nan, "Optimizer produced NaN parameters");
    }
}

#[test]
fn test_optimizer_detects_infinity_in_parameters() {
    // Ensures no infinite values in optimized parameters
    let optimizer = FsrsOptimizer::with_config(0.01, 10, 20);

    let repo = setup_test_repo_with_sessions(25, &[Rating::Good, Rating::Easy]);

    let result = optimizer.optimize(&repo);

    if let Ok(params) = result {
        let has_infinity = params.w.iter().any(|&w| w.is_infinite());
        assert!(!has_infinity, "Optimizer produced infinite parameters");
    }
}

#[test]
fn test_optimizer_with_custom_learning_rate() {
    let high_lr = FsrsOptimizer::with_config(0.1, 10, 20);
    let low_lr = FsrsOptimizer::with_config(0.001, 10, 20);

    let repo = setup_test_repo_with_sessions(30, &[Rating::Good, Rating::Hard]);

    let result_high = high_lr.optimize(&repo);
    let result_low = low_lr.optimize(&repo);

    // Both should succeed
    assert!(result_high.is_ok());
    assert!(result_low.is_ok());

    // Parameters might differ due to learning rate
    let params_high = result_high.unwrap();
    let params_low = result_low.unwrap();

    assert_eq!(params_high.w.len(), 19);
    assert_eq!(params_low.w.len(), 19);
}

#[test]
fn test_optimizer_with_many_epochs() {
    let many_epochs = FsrsOptimizer::with_config(0.01, 200, 20);

    let repo = setup_test_repo_with_sessions(30, &[Rating::Good, Rating::Hard, Rating::Easy]);

    let result = many_epochs.optimize(&repo);

    assert!(result.is_ok());

    let params = result.unwrap();

    // After many epochs, parameters should be stable
    for &w in &params.w {
        assert!(w.is_finite());
    }
}

#[test]
fn test_optimizer_with_few_epochs() {
    let few_epochs = FsrsOptimizer::with_config(0.01, 5, 20);

    let repo = setup_test_repo_with_sessions(25, &[Rating::Good, Rating::Hard]);

    let result = few_epochs.optimize(&repo);

    assert!(result.is_ok());

    // Even with few epochs, should produce valid parameters
    let params = result.unwrap();
    for &w in &params.w {
        assert!(w.is_finite());
    }
}

#[test]
fn test_optimizer_handles_very_large_dataset() {
    let optimizer = FsrsOptimizer::with_config(0.01, 50, 50);

    // Create a large dataset
    let repo = setup_test_repo_with_sessions(
        500,
        &[
            Rating::Good,
            Rating::Hard,
            Rating::Easy,
            Rating::Again,
            Rating::Good,
        ],
    );

    let result = optimizer.optimize(&repo);

    // Should handle large datasets without issues
    assert!(result.is_ok());

    let params = result.unwrap();
    for &w in &params.w {
        assert!(w.is_finite());
    }
}

#[test]
fn test_optimizer_with_alternating_success_failure() {
    let optimizer = FsrsOptimizer::with_config(0.01, 20, 25);

    // Alternating pattern: success, fail, success, fail
    let ratings = vec![Rating::Easy, Rating::Again, Rating::Good, Rating::Again];

    let repo = setup_test_repo_with_sessions(40, &ratings);

    let result = optimizer.optimize(&repo);
    assert!(result.is_ok());

    let params = result.unwrap();

    // Should adapt to alternating pattern
    for &w in &params.w {
        assert!(w.is_finite());
    }
}

#[test]
fn test_optimizer_default_config() {
    let optimizer = FsrsOptimizer::new();

    // Verify default configuration
    // Default: learning_rate=0.01, epochs=100, min_reviews=50

    let repo = setup_test_repo_with_sessions(
        60,
        &[Rating::Good, Rating::Hard, Rating::Easy],
    );

    let result = optimizer.optimize(&repo);
    assert!(result.is_ok());
}

#[test]
fn test_optimizer_parameter_stability() {
    let optimizer = FsrsOptimizer::with_config(0.01, 50, 30);

    let repo = setup_test_repo_with_sessions(50, &[Rating::Good, Rating::Hard, Rating::Easy]);

    // Optimize twice with same data
    let result1 = optimizer.optimize(&repo);
    let result2 = optimizer.optimize(&repo);

    assert!(result1.is_ok());
    assert!(result2.is_ok());

    let params1 = result1.unwrap();
    let params2 = result2.unwrap();

    // Results should be deterministic (same input -> same output)
    for i in 0..19 {
        let diff = (params1.w[i] - params2.w[i]).abs();
        assert!(
            diff < 0.001,
            "Parameter w[{}] differs between runs: {} vs {}",
            i,
            params1.w[i],
            params2.w[i]
        );
    }
}

#[test]
fn test_optimizer_preserves_parameter_count() {
    let optimizer = FsrsOptimizer::new();

    let repo = setup_test_repo_with_sessions(60, &[Rating::Good, Rating::Hard]);

    let result = optimizer.optimize(&repo);

    if let Ok(params) = result {
        // FSRS-5 always has exactly 19 parameters
        assert_eq!(params.w.len(), 19, "FSRS-5 must have exactly 19 parameters");
    }
}
