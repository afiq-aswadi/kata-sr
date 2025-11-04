//! SM-2 spaced repetition scheduler implementation.
//!
//! This module implements the SM-2 algorithm used by Anki for scheduling reviews.
//! The algorithm uses a 0-3 quality rating scale and adjusts review intervals based
//! on user performance.
//!
//! # Rating Scale
//!
//! - **Again (0)**: Complete failure - resets interval to 1 day
//! - **Hard (1)**: Struggled but passed - minimal interval growth
//! - **Good (2)**: Normal difficulty - standard SM-2 progression
//! - **Easy (3)**: Too easy - accelerated interval growth

use serde::{Deserialize, Serialize};

/// Quality rating for a review session.
///
/// Used to determine the next review interval according to the SM-2 algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(i32)]
pub enum QualityRating {
    /// Complete failure - user could not complete the kata
    Again = 0,
    /// Struggled but eventually passed - needs more practice
    Hard = 1,
    /// Normal difficulty - appropriate challenge level
    Good = 2,
    /// Too easy - user completed it quickly without issues
    Easy = 3,
}

impl QualityRating {
    /// Creates a QualityRating from an integer value.
    ///
    /// # Examples
    ///
    /// ```
    /// # use kata_sr::core::scheduler::QualityRating;
    /// assert_eq!(QualityRating::from_int(0), Some(QualityRating::Again));
    /// assert_eq!(QualityRating::from_int(3), Some(QualityRating::Easy));
    /// assert_eq!(QualityRating::from_int(4), None);
    /// ```
    pub fn from_int(value: i32) -> Option<Self> {
        match value {
            0 => Some(QualityRating::Again),
            1 => Some(QualityRating::Hard),
            2 => Some(QualityRating::Good),
            3 => Some(QualityRating::Easy),
            _ => None,
        }
    }
}

/// SM-2 spaced repetition state.
///
/// Tracks the current scheduling parameters for a kata. The state is updated
/// after each review based on the quality rating provided by the user.
///
/// # Fields
///
/// - `ease_factor`: Multiplier for interval growth (min 1.3, default 2.5)
/// - `interval_days`: Current interval between reviews
/// - `repetition_count`: Number of successful reviews (resets on Again)
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SM2State {
    pub ease_factor: f64,
    pub interval_days: i64,
    pub repetition_count: i64,
}

impl SM2State {
    /// Creates a new SM-2 state with default values.
    ///
    /// # Examples
    ///
    /// ```
    /// # use kata_sr::core::scheduler::SM2State;
    /// let state = SM2State::new();
    /// assert_eq!(state.ease_factor, 2.5);
    /// assert_eq!(state.interval_days, 1);
    /// assert_eq!(state.repetition_count, 0);
    /// ```
    pub fn new() -> Self {
        Self {
            ease_factor: 2.5,
            interval_days: 1,
            repetition_count: 0,
        }
    }

    /// Updates the SM-2 state based on a quality rating and returns the next interval.
    ///
    /// The algorithm adjusts the ease factor and calculates the next review interval
    /// according to the SM-2 spaced repetition formula.
    ///
    /// # Arguments
    ///
    /// * `quality` - The quality rating from the completed review
    ///
    /// # Returns
    ///
    /// The number of days until the next review
    ///
    /// # Examples
    ///
    /// ```
    /// # use kata_sr::core::scheduler::{SM2State, QualityRating};
    /// let mut state = SM2State::new();
    ///
    /// // First Good review returns 1 day
    /// assert_eq!(state.update(QualityRating::Good), 1);
    ///
    /// // Second Good review returns 6 days
    /// assert_eq!(state.update(QualityRating::Good), 6);
    ///
    /// // Third Good review uses the ease factor (6 * 2.5 = 15)
    /// assert_eq!(state.update(QualityRating::Good), 15);
    /// ```
    pub fn update(&mut self, quality: QualityRating) -> i64 {
        match quality {
            QualityRating::Again => {
                // reset to beginning
                self.interval_days = 1;
                self.repetition_count = 0;
                self.ease_factor = (self.ease_factor - 0.2).max(1.3);
            }
            QualityRating::Hard => {
                // minimal growth, decrease ease
                self.ease_factor = (self.ease_factor - 0.15).max(1.3);
                self.interval_days = (self.interval_days as f64 * 1.2).round() as i64;
                self.repetition_count += 1;
            }
            QualityRating::Good => {
                // standard SM-2 progression
                if self.repetition_count == 0 {
                    self.interval_days = 1;
                } else if self.repetition_count == 1 {
                    self.interval_days = 6;
                } else {
                    self.interval_days =
                        (self.interval_days as f64 * self.ease_factor).round() as i64;
                }
                self.repetition_count += 1;
            }
            QualityRating::Easy => {
                // accelerated growth, increase ease
                self.ease_factor = (self.ease_factor + 0.15).min(2.5);
                if self.repetition_count == 0 {
                    self.interval_days = 1;
                } else if self.repetition_count == 1 {
                    self.interval_days = 6;
                } else {
                    self.interval_days =
                        (self.interval_days as f64 * self.ease_factor * 1.3).round() as i64;
                }
                self.repetition_count += 1;
            }
        }

        self.interval_days
    }
}

impl Default for SM2State {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quality_rating_from_int() {
        assert_eq!(QualityRating::from_int(0), Some(QualityRating::Again));
        assert_eq!(QualityRating::from_int(1), Some(QualityRating::Hard));
        assert_eq!(QualityRating::from_int(2), Some(QualityRating::Good));
        assert_eq!(QualityRating::from_int(3), Some(QualityRating::Easy));
        assert_eq!(QualityRating::from_int(4), None);
        assert_eq!(QualityRating::from_int(-1), None);
    }

    #[test]
    fn test_sm2_initial_state() {
        let state = SM2State::new();
        assert_eq!(state.ease_factor, 2.5);
        assert_eq!(state.interval_days, 1);
        assert_eq!(state.repetition_count, 0);
    }

    #[test]
    fn test_sm2_good_progression() {
        let mut state = SM2State::new();

        // first review: Good -> 1 day
        assert_eq!(state.update(QualityRating::Good), 1);
        assert_eq!(state.repetition_count, 1);
        assert_eq!(state.interval_days, 1);

        // second review: Good -> 6 days
        assert_eq!(state.update(QualityRating::Good), 6);
        assert_eq!(state.repetition_count, 2);
        assert_eq!(state.interval_days, 6);

        // third review: Good -> 6 * 2.5 = 15 days
        assert_eq!(state.update(QualityRating::Good), 15);
        assert_eq!(state.repetition_count, 3);
        assert_eq!(state.interval_days, 15);

        // fourth review: Good -> 15 * 2.5 = 37.5 -> 38 days
        assert_eq!(state.update(QualityRating::Good), 38);
        assert_eq!(state.repetition_count, 4);
    }

    #[test]
    fn test_sm2_reset_on_again() {
        let mut state = SM2State::new();
        state.update(QualityRating::Good);
        state.update(QualityRating::Good);
        state.update(QualityRating::Good);

        assert_eq!(state.interval_days, 15);
        assert_eq!(state.repetition_count, 3);
        let ease_before = state.ease_factor;

        // rating Again should reset interval and count
        state.update(QualityRating::Again);
        assert_eq!(state.interval_days, 1);
        assert_eq!(state.repetition_count, 0);
        assert_eq!(state.ease_factor, ease_before - 0.2);
    }

    #[test]
    fn test_sm2_hard_minimal_growth() {
        let mut state = SM2State::new();
        state.update(QualityRating::Good); // 1 day
        state.update(QualityRating::Good); // 6 days

        let ease_before = state.ease_factor;
        let interval_before = state.interval_days;

        // Hard should grow interval by 1.2x and decrease ease
        state.update(QualityRating::Hard);
        assert_eq!(
            state.interval_days,
            (interval_before as f64 * 1.2).round() as i64
        );
        assert_eq!(state.ease_factor, ease_before - 0.15);
        assert_eq!(state.repetition_count, 3);
    }

    #[test]
    fn test_sm2_easy_accelerated_growth() {
        let mut state = SM2State::new();
        state.update(QualityRating::Good); // 1 day
        state.update(QualityRating::Good); // 6 days

        let ease_before = state.ease_factor;
        let interval_before = state.interval_days;

        // Easy should grow interval by ease * 1.3 and increase ease (but capped at 2.5)
        state.update(QualityRating::Easy);
        let expected_interval = (interval_before as f64 * ease_before * 1.3).round() as i64;
        assert_eq!(state.interval_days, expected_interval);
        assert_eq!(state.ease_factor, (ease_before + 0.15).min(2.5));
        assert_eq!(state.repetition_count, 3);
    }

    #[test]
    fn test_sm2_ease_factor_bounds() {
        let mut state = SM2State::new();

        // test minimum bound (1.3)
        for _ in 0..20 {
            state.update(QualityRating::Again);
        }
        assert!(state.ease_factor >= 1.3);

        // test maximum bound (2.5)
        state = SM2State::new();
        state.update(QualityRating::Good); // 1 day
        state.update(QualityRating::Good); // 6 days
        for _ in 0..20 {
            state.update(QualityRating::Easy);
        }
        assert!(state.ease_factor <= 2.5);
    }

    #[test]
    fn test_sm2_easy_first_review() {
        let mut state = SM2State::new();
        // Easy on first review should still use 1 day
        assert_eq!(state.update(QualityRating::Easy), 1);
        assert_eq!(state.repetition_count, 1);
    }

    #[test]
    fn test_sm2_hard_first_review() {
        let mut state = SM2State::new();
        // Hard on first review: 1 * 1.2 = 1.2 -> 1 day
        assert_eq!(state.update(QualityRating::Hard), 1);
        assert_eq!(state.repetition_count, 1);
    }
}
