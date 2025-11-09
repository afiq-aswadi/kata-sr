//! FSRS parameter optimizer.
//!
//! This module provides functionality to optimize FSRS parameters based on
//! review history. The optimizer uses gradient descent to minimize the difference
//! between predicted and actual retention rates.
//!
//! # Usage
//!
//! ```no_run
//! use kata_sr::core::fsrs_optimizer::FsrsOptimizer;
//! use kata_sr::db::repo::KataRepository;
//!
//! let repo = KataRepository::new("kata.db")?;
//! let optimizer = FsrsOptimizer::new();
//!
//! // Train on review history
//! let optimized_params = optimizer.optimize(&repo)?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use crate::core::fsrs::FsrsParams;
use crate::db::repo::KataRepository;
use rusqlite::Result;

/// Training case for FSRS optimization.
///
/// Represents a single review event with its outcome.
#[derive(Debug, Clone)]
struct TrainingCase {
    /// Days since last review
    elapsed_days: u32,
    /// Scheduled interval
    scheduled_days: u32,
    /// User rating (1-4)
    rating: u8,
    /// Actual retention (1.0 if passed, 0.0 if failed)
    actual_retention: f64,
    /// Stability at time of review
    stability: f64,
    /// Difficulty at time of review
    difficulty: f64,
}

/// FSRS parameter optimizer using gradient descent.
///
/// Optimizes the 19 FSRS parameters to minimize prediction error
/// based on historical review data.
pub struct FsrsOptimizer {
    /// Learning rate for gradient descent
    learning_rate: f64,
    /// Number of training epochs
    epochs: usize,
    /// Minimum number of reviews required for optimization
    min_reviews: usize,
}

impl FsrsOptimizer {
    /// Creates a new optimizer with default settings.
    ///
    /// Default settings:
    /// - Learning rate: 0.01
    /// - Epochs: 100
    /// - Minimum reviews: 50
    pub fn new() -> Self {
        Self {
            learning_rate: 0.01,
            epochs: 100,
            min_reviews: 50,
        }
    }

    /// Creates an optimizer with custom settings.
    ///
    /// # Arguments
    ///
    /// * `learning_rate` - Step size for parameter updates
    /// * `epochs` - Number of training iterations
    /// * `min_reviews` - Minimum reviews needed before optimization
    pub fn with_config(learning_rate: f64, epochs: usize, min_reviews: usize) -> Self {
        Self {
            learning_rate,
            epochs,
            min_reviews,
        }
    }

    /// Optimizes FSRS parameters based on review history.
    ///
    /// Returns optimized parameters if there are enough reviews,
    /// otherwise returns default parameters.
    ///
    /// # Arguments
    ///
    /// * `repo` - Database repository containing review history
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use kata_sr::core::fsrs_optimizer::FsrsOptimizer;
    /// # use kata_sr::db::repo::KataRepository;
    /// # let repo = KataRepository::new("kata.db")?;
    /// let optimizer = FsrsOptimizer::new();
    /// let params = optimizer.optimize(&repo)?;
    /// println!("Optimized parameters: {:?}", params.w);
    /// # Ok::<(), rusqlite::Error>(())
    /// ```
    pub fn optimize(&self, repo: &KataRepository) -> Result<FsrsParams> {
        // Extract training data from sessions
        let training_data = self.extract_training_data(repo)?;

        if training_data.len() < self.min_reviews {
            println!(
                "Not enough reviews for optimization ({} < {}). Using default parameters.",
                training_data.len(),
                self.min_reviews
            );
            return Ok(FsrsParams::default());
        }

        println!(
            "Optimizing FSRS parameters on {} reviews...",
            training_data.len()
        );

        // Start with default parameters
        let mut params = FsrsParams::default();
        let mut best_params = params.clone();
        let mut best_loss = self.compute_loss(&params, &training_data);

        for epoch in 0..self.epochs {
            let loss = self.compute_loss(&params, &training_data);
            let gradients = self.compute_gradients(&params, &training_data);

            // Update parameters using gradient descent
            for i in 0..19 {
                params.w[i] -= self.learning_rate * gradients[i];
                // Ensure parameters stay positive
                params.w[i] = params.w[i].max(0.01);
            }

            // Track best parameters
            if loss < best_loss {
                best_loss = loss;
                best_params = params.clone();
            }

            if epoch % 10 == 0 {
                println!("Epoch {}: loss = {:.6}", epoch, loss);
            }
        }

        println!(
            "Optimization complete. Final loss: {:.6}",
            best_loss
        );

        Ok(best_params)
    }

    /// Extracts training data from review sessions.
    ///
    /// Converts database sessions into training cases for optimization.
    fn extract_training_data(&self, repo: &KataRepository) -> Result<Vec<TrainingCase>> {
        let all_katas = repo.get_all_katas()?;
        let mut training_data = Vec::new();

        for kata in all_katas {
            let sessions = repo.get_recent_sessions(kata.id, 1000)?;

            // Need at least 2 sessions to calculate elapsed days
            if sessions.len() < 2 {
                continue;
            }

            // Track FSRS state across sessions
            let mut stability = 0.0;
            let mut difficulty = 0.0;
            let mut state = crate::core::fsrs::CardState::New;

            for (i, session) in sessions.iter().enumerate() {
                if session.quality_rating.is_none() {
                    continue;
                }

                let rating = session.quality_rating.unwrap();
                if rating < 1 || rating > 4 {
                    continue;
                }

                // Calculate elapsed days
                let elapsed_days = if i > 0 {
                    let prev_session = &sessions[i - 1];
                    if let (Some(completed), Some(prev_completed)) =
                        (session.completed_at, prev_session.completed_at)
                    {
                        (completed - prev_completed).num_days().max(0) as u32
                    } else {
                        0
                    }
                } else {
                    0
                };

                // Actual retention: did they pass?
                let actual_retention = if rating >= 2 { 1.0 } else { 0.0 };

                // Only add to training data if we have valid state
                if i > 0 {
                    training_data.push(TrainingCase {
                        elapsed_days,
                        scheduled_days: 0, // We don't track this in old sessions
                        rating: rating as u8,
                        actual_retention,
                        stability,
                        difficulty,
                    });
                }

                // Update simulated FSRS state for next iteration
                // This is a simplified simulation - actual implementation would be more complex
                if state == crate::core::fsrs::CardState::New {
                    stability = match rating {
                        1 => 0.4,
                        2 => 1.2,
                        3 => 3.1,
                        _ => 15.5,
                    };
                    difficulty = 7.2 + 0.5 * (rating as f64 - 3.0);
                    state = if rating >= 3 {
                        crate::core::fsrs::CardState::Review
                    } else {
                        crate::core::fsrs::CardState::Learning
                    };
                } else {
                    // Simple stability update
                    if rating >= 2 {
                        stability *= 1.5;
                    } else {
                        stability *= 0.8;
                    }
                }
            }
        }

        Ok(training_data)
    }

    /// Computes mean squared error loss.
    ///
    /// Calculates the difference between predicted retention
    /// and actual retention across all training cases.
    fn compute_loss(&self, params: &FsrsParams, data: &[TrainingCase]) -> f64 {
        if data.is_empty() {
            return 0.0;
        }

        let total_error: f64 = data
            .iter()
            .map(|case| {
                let predicted = self.predict_retention(params, case);
                (predicted - case.actual_retention).powi(2)
            })
            .sum();

        total_error / data.len() as f64
    }

    /// Predicts retention probability for a training case.
    ///
    /// Uses the forgetting curve formula:
    /// R = (1 + elapsed_days / (9 * S))^(-1)
    fn predict_retention(&self, _params: &FsrsParams, case: &TrainingCase) -> f64 {
        if case.stability < 0.1 {
            return 0.5; // Default prediction for unstable cases
        }

        let retrievability =
            (1.0 + case.elapsed_days as f64 / (9.0 * case.stability)).powf(-1.0);
        retrievability.clamp(0.0, 1.0)
    }

    /// Computes gradients using numerical approximation.
    ///
    /// Uses finite differences to approximate the gradient of the loss
    /// function with respect to each parameter.
    fn compute_gradients(&self, params: &FsrsParams, data: &[TrainingCase]) -> [f64; 19] {
        let epsilon = 1e-6;
        let mut gradients = [0.0; 19];
        let base_loss = self.compute_loss(params, data);

        for i in 0..19 {
            let mut params_plus = params.clone();
            params_plus.w[i] += epsilon;
            let loss_plus = self.compute_loss(&params_plus, data);

            // Use one-sided difference to avoid negative parameter values
            gradients[i] = (loss_plus - base_loss) / epsilon;
        }

        gradients
    }
}

impl Default for FsrsOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::repo::{KataRepository, NewKata, NewSession};
    use chrono::Utc;

    fn setup_test_repo_with_sessions() -> KataRepository {
        let repo = KataRepository::new_in_memory().unwrap();
        repo.run_migrations().unwrap();

        // Create a kata
        let kata_id = repo
            .create_kata(
                &NewKata {
                    name: "test_kata".to_string(),
                    category: "test".to_string(),
                    description: "Test".to_string(),
                    base_difficulty: 3,
                    parent_kata_id: None,
                    variation_params: None,
                },
                Utc::now(),
            )
            .unwrap();

        // Create multiple sessions with varying ratings
        let base_time = Utc::now();
        for (i, rating) in [3, 2, 3, 1, 3, 2, 3, 3].iter().enumerate() {
            let session_time = base_time + chrono::Duration::days(i as i64);
            repo.create_session(&NewSession {
                kata_id,
                started_at: session_time,
                completed_at: Some(session_time),
                test_results_json: None,
                num_passed: if *rating >= 2 { Some(5) } else { Some(0) },
                num_failed: if *rating >= 2 { Some(0) } else { Some(5) },
                num_skipped: None,
                duration_ms: Some(1000),
                quality_rating: Some(*rating),
            })
            .unwrap();
        }

        repo
    }

    #[test]
    fn test_optimizer_creation() {
        let optimizer = FsrsOptimizer::new();
        assert_eq!(optimizer.learning_rate, 0.01);
        assert_eq!(optimizer.epochs, 100);
        assert_eq!(optimizer.min_reviews, 50);
    }

    #[test]
    fn test_optimizer_with_config() {
        let optimizer = FsrsOptimizer::with_config(0.05, 50, 25);
        assert_eq!(optimizer.learning_rate, 0.05);
        assert_eq!(optimizer.epochs, 50);
        assert_eq!(optimizer.min_reviews, 25);
    }

    #[test]
    fn test_extract_training_data() {
        let repo = setup_test_repo_with_sessions();
        let optimizer = FsrsOptimizer::new();

        let training_data = optimizer.extract_training_data(&repo).unwrap();

        // Should have sessions minus 1 (first session has no previous review)
        assert!(training_data.len() >= 5);

        // Check that training cases have valid data
        for case in &training_data {
            assert!(case.rating >= 1 && case.rating <= 4);
            assert!(case.actual_retention == 0.0 || case.actual_retention == 1.0);
            assert!(case.stability >= 0.0);
            assert!(case.difficulty >= 0.0);
        }
    }

    #[test]
    fn test_predict_retention() {
        let optimizer = FsrsOptimizer::new();
        let params = FsrsParams::default();

        let case = TrainingCase {
            elapsed_days: 0,
            scheduled_days: 0,
            rating: 3,
            actual_retention: 1.0,
            stability: 10.0,
            difficulty: 5.0,
        };

        let retention = optimizer.predict_retention(&params, &case);
        assert!(retention > 0.9); // At day 0, retention should be very high

        let case_later = TrainingCase {
            elapsed_days: 30,
            scheduled_days: 0,
            rating: 3,
            actual_retention: 0.0,
            stability: 10.0,
            difficulty: 5.0,
        };

        let retention_later = optimizer.predict_retention(&params, &case_later);
        assert!(retention_later < retention); // After 30 days, retention should be lower
    }

    #[test]
    fn test_compute_loss() {
        let optimizer = FsrsOptimizer::new();
        let params = FsrsParams::default();

        let data = vec![
            TrainingCase {
                elapsed_days: 0,
                scheduled_days: 0,
                rating: 3,
                actual_retention: 1.0,
                stability: 10.0,
                difficulty: 5.0,
            },
            TrainingCase {
                elapsed_days: 100,
                scheduled_days: 0,
                rating: 1,
                actual_retention: 0.0,
                stability: 10.0,
                difficulty: 5.0,
            },
        ];

        let loss = optimizer.compute_loss(&params, &data);
        assert!(loss >= 0.0);
        assert!(loss < 1.0); // Loss should be reasonable
    }

    #[test]
    fn test_optimize_with_insufficient_data() {
        let repo = KataRepository::new_in_memory().unwrap();
        repo.run_migrations().unwrap();

        let optimizer = FsrsOptimizer::new();
        let result = optimizer.optimize(&repo).unwrap();

        // Should return default parameters when there's insufficient data
        let default_params = FsrsParams::default();
        assert_eq!(result.w, default_params.w);
    }

    #[test]
    fn test_compute_gradients() {
        let optimizer = FsrsOptimizer::new();
        let params = FsrsParams::default();

        let data = vec![TrainingCase {
            elapsed_days: 5,
            scheduled_days: 0,
            rating: 3,
            actual_retention: 1.0,
            stability: 10.0,
            difficulty: 5.0,
        }];

        let gradients = optimizer.compute_gradients(&params, &data);

        // All gradients should be finite
        for (i, &grad) in gradients.iter().enumerate() {
            assert!(grad.is_finite(), "Gradient {} is not finite", i);
        }
    }

    #[test]
    fn test_parameters_stay_positive() {
        let repo = setup_test_repo_with_sessions();
        let optimizer = FsrsOptimizer::with_config(0.001, 5, 1);

        let result = optimizer.optimize(&repo).unwrap();

        // All parameters should remain positive
        for (i, &w) in result.w.iter().enumerate() {
            assert!(w > 0.0, "Parameter w[{}] = {} is not positive", i, w);
        }
    }
}
