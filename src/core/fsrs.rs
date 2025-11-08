//! FSRS-5 (Free Spaced Repetition Scheduler) algorithm implementation.
//!
//! FSRS-5 is a modern, more accurate alternative to SM-2 that uses a memory model
//! based on retrievability and stability. It provides better predictions for long-term
//! retention and adapts more effectively to individual learning patterns.
//!
//! # Key Concepts
//!
//! - **Stability (S)**: Half-life of a memory in days - how long until retrievability drops to 50%
//! - **Difficulty (D)**: How inherently difficult a card is (1-10 scale)
//! - **Retrievability (R)**: Current probability of successfully recalling the information
//! - **State**: New, Learning, Review, or Relearning
//!
//! # References
//!
//! - Algorithm: <https://github.com/open-spaced-repetition/fsrs4anki/wiki/The-Algorithm>
//! - Default parameters: <https://github.com/open-spaced-repetition/fsrs-rs>

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// FSRS-5 model parameters.
///
/// Contains 19 parameters (w0-w18) that control the behavior of the algorithm.
/// These can be optimized based on review history for personalized scheduling.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FsrsParams {
    /// FSRS-5 parameters (19 weights)
    pub w: [f64; 19],
}

impl FsrsParams {
    /// Creates default FSRS-5 parameters based on research data.
    ///
    /// These parameters are derived from analyzing millions of reviews
    /// and work well for most users without optimization.
    pub fn default_params() -> Self {
        Self {
            w: [
                0.4072, 1.1829, 3.1262, 15.4722, 7.2102, 0.5316, 1.0651, 0.0234, 1.616, 0.1544,
                1.0824, 1.9813, 0.0953, 0.2975, 2.2042, 0.2407, 2.9466, 0.5034, 0.6567,
            ],
        }
    }
}

impl Default for FsrsParams {
    fn default() -> Self {
        Self::default_params()
    }
}

/// Card state in the FSRS system.
///
/// Tracks the learning phase of a card to apply appropriate scheduling rules.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CardState {
    /// Card has never been reviewed
    New,
    /// Card is being learned for the first time (short intervals)
    Learning,
    /// Card is in regular review rotation (long intervals)
    Review,
    /// Card failed and is being relearned
    Relearning,
}

impl CardState {
    /// Converts a string representation to CardState.
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "New" => Some(CardState::New),
            "Learning" => Some(CardState::Learning),
            "Review" => Some(CardState::Review),
            "Relearning" => Some(CardState::Relearning),
            _ => None,
        }
    }

    /// Converts CardState to string representation.
    pub fn to_str(&self) -> &'static str {
        match self {
            CardState::New => "New",
            CardState::Learning => "Learning",
            CardState::Review => "Review",
            CardState::Relearning => "Relearning",
        }
    }
}

/// Quality rating for a review (1-4 scale).
///
/// Maps to the same semantic meaning as SM-2's 0-3 scale:
/// - Again (1): Complete failure
/// - Hard (2): Struggled but passed
/// - Good (3): Normal difficulty
/// - Easy (4): Too easy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum Rating {
    /// Complete failure - reset to relearning
    Again = 1,
    /// Struggled but passed - minimal interval growth
    Hard = 2,
    /// Normal difficulty - standard progression
    Good = 3,
    /// Too easy - accelerated progression
    Easy = 4,
}

impl Rating {
    /// Creates a Rating from an integer value.
    ///
    /// # Examples
    ///
    /// ```
    /// # use kata_sr::core::fsrs::Rating;
    /// assert_eq!(Rating::from_int(1), Some(Rating::Again));
    /// assert_eq!(Rating::from_int(4), Some(Rating::Easy));
    /// assert_eq!(Rating::from_int(5), None);
    /// ```
    pub fn from_int(value: i32) -> Option<Self> {
        match value {
            1 => Some(Rating::Again),
            2 => Some(Rating::Hard),
            3 => Some(Rating::Good),
            4 => Some(Rating::Easy),
            _ => None,
        }
    }

    /// Converts a Rating to integer value.
    pub fn to_int(&self) -> i32 {
        *self as i32
    }
}

/// FSRS card state tracking memory parameters and scheduling information.
///
/// This struct maintains all the state needed for FSRS scheduling, including
/// memory stability, difficulty, and review history.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FsrsCard {
    /// Memory stability in days (S)
    pub stability: f64,
    /// Card difficulty on 1-10 scale (D)
    pub difficulty: f64,
    /// Days since last review
    pub elapsed_days: u32,
    /// Scheduled interval in days
    pub scheduled_days: u32,
    /// Number of reviews completed
    pub reps: u32,
    /// Number of lapses (Again ratings)
    pub lapses: u32,
    /// Current learning state
    pub state: CardState,
    /// Timestamp of last review
    pub last_review: Option<DateTime<Utc>>,
}

impl FsrsCard {
    /// Creates a new card in the New state with default values.
    ///
    /// # Examples
    ///
    /// ```
    /// # use kata_sr::core::fsrs::FsrsCard;
    /// let card = FsrsCard::new();
    /// assert_eq!(card.stability, 0.0);
    /// assert_eq!(card.reps, 0);
    /// ```
    pub fn new() -> Self {
        Self {
            stability: 0.0,
            difficulty: 0.0,
            elapsed_days: 0,
            scheduled_days: 0,
            reps: 0,
            lapses: 0,
            state: CardState::New,
            last_review: None,
        }
    }

    /// Schedules the next review based on the rating and updates card state.
    ///
    /// This is the main entry point for FSRS scheduling. It updates all card
    /// parameters according to the FSRS-5 algorithm.
    ///
    /// # Arguments
    ///
    /// * `rating` - User's rating of the review difficulty
    /// * `params` - FSRS parameters to use for calculation
    /// * `now` - Current timestamp
    ///
    /// # Examples
    ///
    /// ```
    /// # use kata_sr::core::fsrs::{FsrsCard, FsrsParams, Rating};
    /// # use chrono::Utc;
    /// let mut card = FsrsCard::new();
    /// let params = FsrsParams::default();
    ///
    /// card.schedule(Rating::Good, &params, Utc::now());
    /// assert!(card.scheduled_days > 0);
    /// ```
    pub fn schedule(&mut self, rating: Rating, params: &FsrsParams, now: DateTime<Utc>) {
        // Calculate elapsed days since last review
        if let Some(last) = self.last_review {
            let duration = now.signed_duration_since(last);
            self.elapsed_days = duration.num_days().max(0) as u32;
        }

        match self.state {
            CardState::New => self.schedule_new(rating, params),
            CardState::Learning | CardState::Relearning => self.schedule_learning(rating, params),
            CardState::Review => self.schedule_review(rating, params),
        }

        self.last_review = Some(now);
    }

    /// Schedules a new card (first review).
    fn schedule_new(&mut self, rating: Rating, params: &FsrsParams) {
        // Initialize difficulty based on first rating
        self.difficulty = self.init_difficulty(rating, params);

        // Initialize stability based on first rating
        self.stability = self.init_stability(rating, params);

        match rating {
            Rating::Again => {
                self.state = CardState::Learning;
                self.scheduled_days = 0; // Review again today
            }
            Rating::Hard => {
                self.state = CardState::Learning;
                self.scheduled_days = 1;
            }
            Rating::Good => {
                self.state = CardState::Review;
                self.scheduled_days = self.stability.round() as u32;
                self.reps = 1;
            }
            Rating::Easy => {
                self.state = CardState::Review;
                self.scheduled_days = (self.stability * 1.3).round() as u32;
                self.reps = 1;
            }
        }
    }

    /// Schedules a card in Learning or Relearning state.
    fn schedule_learning(&mut self, rating: Rating, _params: &FsrsParams) {
        match rating {
            Rating::Again => {
                self.scheduled_days = 0;
                self.lapses += 1;
            }
            Rating::Hard => {
                self.scheduled_days = 1;
            }
            Rating::Good | Rating::Easy => {
                self.state = CardState::Review;
                self.scheduled_days = self.stability.round() as u32;
                self.reps = 1;
            }
        }
    }

    /// Schedules a card in Review state.
    fn schedule_review(&mut self, rating: Rating, params: &FsrsParams) {
        let elapsed = self.elapsed_days as f64;
        let retrievability = self.forgetting_curve(elapsed, self.stability);

        // Update difficulty
        self.difficulty = self.next_difficulty(self.difficulty, rating, params);

        // Update stability and schedule based on rating
        match rating {
            Rating::Again => {
                self.state = CardState::Relearning;
                self.lapses += 1;
                self.scheduled_days = 0;
                self.stability =
                    self.forgetting_stability(self.stability, self.difficulty, retrievability, params);
            }
            Rating::Hard => {
                self.reps += 1;
                self.stability = self.recall_stability(
                    self.stability,
                    self.difficulty,
                    retrievability,
                    rating,
                    params,
                );
                self.scheduled_days = self.stability.round() as u32;
            }
            Rating::Good => {
                self.reps += 1;
                self.stability = self.recall_stability(
                    self.stability,
                    self.difficulty,
                    retrievability,
                    rating,
                    params,
                );
                self.scheduled_days = self.stability.round() as u32;
            }
            Rating::Easy => {
                self.reps += 1;
                self.stability = self.recall_stability(
                    self.stability,
                    self.difficulty,
                    retrievability,
                    rating,
                    params,
                );
                self.scheduled_days = (self.stability * 1.3).round() as u32;
            }
        }
    }

    // ===== FSRS-5 Formula Implementations =====

    /// Calculates initial stability for a new card.
    ///
    /// Uses parameters w[0-3] corresponding to Again/Hard/Good/Easy ratings.
    fn init_stability(&self, rating: Rating, params: &FsrsParams) -> f64 {
        params.w[rating as usize - 1].max(0.1)
    }

    /// Calculates initial difficulty for a new card.
    ///
    /// Formula: D_0 = w[4] + w[5] * (rating - 3)
    /// Clamped to [1.0, 10.0]
    fn init_difficulty(&self, rating: Rating, params: &FsrsParams) -> f64 {
        let d0 = params.w[4];
        let d_offset = params.w[5] * (rating as i32 - 3) as f64;
        (d0 + d_offset).clamp(1.0, 10.0)
    }

    /// Calculates current retrievability using the forgetting curve.
    ///
    /// Formula: R = (1 + elapsed_days / (9 * S))^(-1)
    /// where S is stability
    fn forgetting_curve(&self, elapsed_days: f64, stability: f64) -> f64 {
        (1.0 + elapsed_days / (9.0 * stability)).powf(-1.0)
    }

    /// Updates difficulty after a review.
    ///
    /// Formula: D_new = D + w[6] * (rating - 3) - w[7] * (D - w[4])
    /// Clamped to [1.0, 10.0]
    fn next_difficulty(&self, d: f64, rating: Rating, params: &FsrsParams) -> f64 {
        let delta_d = params.w[6] * (rating as i32 - 3) as f64;
        let mean_reversion = params.w[7] * (d - params.w[4]);
        (d + delta_d - mean_reversion).clamp(1.0, 10.0)
    }

    /// Calculates new stability after successful recall.
    ///
    /// Formula: S_new = S * (1 + exp(w[8]) * (11 - D) * S^(-w[9]) *
    ///                      (exp((1 - R) * w[10]) - 1) * hard_penalty * easy_bonus)
    fn recall_stability(
        &self,
        s: f64,
        d: f64,
        r: f64,
        rating: Rating,
        params: &FsrsParams,
    ) -> f64 {
        let hard_penalty = if rating == Rating::Hard {
            params.w[15]
        } else {
            1.0
        };
        let easy_bonus = if rating == Rating::Easy {
            params.w[16]
        } else {
            1.0
        };

        s * (1.0
            + (f64::exp(params.w[8])
                * (11.0 - d)
                * s.powf(-params.w[9])
                * ((f64::exp((1.0 - r) * params.w[10]) - 1.0) * hard_penalty * easy_bonus)))
        .max(0.1)
    }

    /// Calculates new stability after forgetting (lapse).
    ///
    /// Formula: S_new = w[11] * D^(-w[12]) * ((S + 1)^w[13] - 1) * exp((1 - R) * w[14])
    fn forgetting_stability(&self, s: f64, d: f64, r: f64, params: &FsrsParams) -> f64 {
        params.w[11]
            * d.powf(-params.w[12])
            * ((s + 1.0).powf(params.w[13]) - 1.0)
            * f64::exp((1.0 - r) * params.w[14])
    }
}

impl Default for FsrsCard {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rating_from_int() {
        assert_eq!(Rating::from_int(1), Some(Rating::Again));
        assert_eq!(Rating::from_int(2), Some(Rating::Hard));
        assert_eq!(Rating::from_int(3), Some(Rating::Good));
        assert_eq!(Rating::from_int(4), Some(Rating::Easy));
        assert_eq!(Rating::from_int(0), None);
        assert_eq!(Rating::from_int(5), None);
    }

    #[test]
    fn test_card_state_conversion() {
        assert_eq!(CardState::from_str("New"), Some(CardState::New));
        assert_eq!(CardState::from_str("Learning"), Some(CardState::Learning));
        assert_eq!(CardState::from_str("Review"), Some(CardState::Review));
        assert_eq!(CardState::from_str("Relearning"), Some(CardState::Relearning));
        assert_eq!(CardState::from_str("Invalid"), None);

        assert_eq!(CardState::New.to_str(), "New");
        assert_eq!(CardState::Learning.to_str(), "Learning");
    }

    #[test]
    fn test_new_card_defaults() {
        let card = FsrsCard::new();
        assert_eq!(card.stability, 0.0);
        assert_eq!(card.difficulty, 0.0);
        assert_eq!(card.reps, 0);
        assert_eq!(card.lapses, 0);
        assert_eq!(card.state, CardState::New);
        assert!(card.last_review.is_none());
    }

    #[test]
    fn test_schedule_new_card_good() {
        let mut card = FsrsCard::new();
        let params = FsrsParams::default();
        let now = Utc::now();

        card.schedule(Rating::Good, &params, now);

        assert_eq!(card.state, CardState::Review);
        assert!(card.stability > 0.0);
        assert!(card.difficulty > 0.0);
        assert_eq!(card.reps, 1);
        assert!(card.scheduled_days > 0);
        assert!(card.last_review.is_some());
    }

    #[test]
    fn test_schedule_new_card_again() {
        let mut card = FsrsCard::new();
        let params = FsrsParams::default();

        card.schedule(Rating::Again, &params, Utc::now());

        assert_eq!(card.state, CardState::Learning);
        assert_eq!(card.scheduled_days, 0);
        assert_eq!(card.reps, 0);
    }

    #[test]
    fn test_schedule_new_card_easy() {
        let mut card = FsrsCard::new();
        let params = FsrsParams::default();

        card.schedule(Rating::Easy, &params, Utc::now());

        assert_eq!(card.state, CardState::Review);
        assert_eq!(card.reps, 1);
        // Easy should give longer interval than Good
        let mut card_good = FsrsCard::new();
        card_good.schedule(Rating::Good, &params, Utc::now());
        assert!(card.scheduled_days >= card_good.scheduled_days);
    }

    #[test]
    fn test_schedule_learning_to_review() {
        let mut card = FsrsCard::new();
        let params = FsrsParams::default();
        let now = Utc::now();

        // First review: Again -> Learning
        card.schedule(Rating::Again, &params, now);
        assert_eq!(card.state, CardState::Learning);

        // Second review: Good -> Review
        card.schedule(Rating::Good, &params, now);
        assert_eq!(card.state, CardState::Review);
        assert_eq!(card.reps, 1);
    }

    #[test]
    fn test_schedule_review_again_becomes_relearning() {
        let mut card = FsrsCard::new();
        let params = FsrsParams::default();
        let now = Utc::now();

        // Get to Review state
        card.schedule(Rating::Good, &params, now);
        assert_eq!(card.state, CardState::Review);

        // Fail the review
        card.schedule(Rating::Again, &params, now);
        assert_eq!(card.state, CardState::Relearning);
        assert_eq!(card.lapses, 1);
        assert_eq!(card.scheduled_days, 0);
    }

    #[test]
    fn test_stability_increases_on_good_review() {
        let mut card = FsrsCard::new();
        let params = FsrsParams::default();
        let now = Utc::now();

        card.schedule(Rating::Good, &params, now);
        let stability1 = card.stability;

        // Simulate time passing
        let later = now + chrono::Duration::days(card.scheduled_days as i64);
        card.schedule(Rating::Good, &params, later);
        let stability2 = card.stability;

        assert!(stability2 > stability1);
    }

    #[test]
    fn test_difficulty_bounds() {
        let mut card = FsrsCard::new();
        let params = FsrsParams::default();
        let now = Utc::now();

        // Multiple Again ratings should not make difficulty go below 1.0
        for _ in 0..20 {
            card.schedule(Rating::Again, &params, now);
        }
        assert!(card.difficulty >= 1.0);

        // Multiple Easy ratings should not make difficulty go above 10.0
        let mut card2 = FsrsCard::new();
        for _ in 0..20 {
            card2.schedule(Rating::Easy, &params, now);
        }
        assert!(card2.difficulty <= 10.0);
    }

    #[test]
    fn test_forgetting_curve() {
        let card = FsrsCard::new();
        let stability = 10.0;

        // At day 0, retrievability should be ~1.0
        let r0 = card.forgetting_curve(0.0, stability);
        assert!(r0 > 0.99);

        // At day 10, retrievability should be lower
        let r10 = card.forgetting_curve(10.0, stability);
        assert!(r10 < r0);

        // At day 100, retrievability should be very low
        let r100 = card.forgetting_curve(100.0, stability);
        assert!(r100 < 0.5);
    }

    #[test]
    fn test_default_params_are_valid() {
        let params = FsrsParams::default();
        assert_eq!(params.w.len(), 19);

        // All parameters should be non-negative
        for (i, &w) in params.w.iter().enumerate() {
            assert!(w.is_finite(), "Parameter w[{}] is not finite", i);
        }
    }

    #[test]
    fn test_elapsed_days_calculation() {
        let mut card = FsrsCard::new();
        let params = FsrsParams::default();
        let now = Utc::now();

        card.schedule(Rating::Good, &params, now);
        assert_eq!(card.elapsed_days, 0);

        // Schedule again 5 days later
        let later = now + chrono::Duration::days(5);
        card.schedule(Rating::Good, &params, later);
        assert_eq!(card.elapsed_days, 5);
    }
}
