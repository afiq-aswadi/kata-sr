//! Adaptive difficulty tracking for kata exercises.
//!
//! This module provides difficulty adjustment based on user performance over recent
//! practice sessions. Unlike SM-2 scheduling (which determines when to review),
//! difficulty tracking helps identify if a kata is too easy or too hard.
//!
//! The difficulty system is independent of SM-2 scheduling and is used primarily
//! for UI recommendations and kata variations.

/// Tracks user performance and computes difficulty adjustments.
///
/// Analyzes recent session results to determine if a kata's difficulty should
/// be increased (too easy), decreased (too hard), or maintained (appropriate).
pub struct DifficultyTracker {
    #[allow(dead_code)]
    window_size: usize,
}

impl DifficultyTracker {
    /// Creates a new difficulty tracker with the specified window size.
    ///
    /// # Arguments
    ///
    /// * `window_size` - Number of recent sessions to consider for adjustment
    ///
    /// # Examples
    ///
    /// ```
    /// # use kata_sr::core::difficulty::DifficultyTracker;
    /// let tracker = DifficultyTracker::new(5);
    /// ```
    pub fn new(window_size: usize) -> Self {
        Self { window_size }
    }

    /// Computes the recommended difficulty adjustment based on recent performance.
    ///
    /// Returns a positive value if the kata is too easy (high success rate),
    /// negative if too hard (low success rate), or zero if appropriate.
    ///
    /// Requires at least 3 sessions for meaningful analysis.
    ///
    /// # Arguments
    ///
    /// * `recent_sessions` - Vector of pass/fail results for recent attempts
    ///
    /// # Returns
    ///
    /// Adjustment value to add to current difficulty:
    /// - `+0.2` if success rate > 90% (too easy)
    /// - `-0.3` if success rate < 50% (too hard)
    /// - `0.0` otherwise (appropriate difficulty)
    ///
    /// # Examples
    ///
    /// ```
    /// # use kata_sr::core::difficulty::DifficultyTracker;
    /// let tracker = DifficultyTracker::new(5);
    ///
    /// // High success rate suggests increasing difficulty
    /// let sessions = vec![true, true, true, true, true];
    /// assert_eq!(tracker.compute_adjustment(&sessions), 0.2);
    ///
    /// // Low success rate suggests decreasing difficulty
    /// let sessions = vec![false, false, true, false, false];
    /// assert_eq!(tracker.compute_adjustment(&sessions), -0.3);
    ///
    /// // Moderate success rate is appropriate
    /// let sessions = vec![true, false, true, true, false];
    /// assert_eq!(tracker.compute_adjustment(&sessions), 0.0);
    /// ```
    pub fn compute_adjustment(&self, recent_sessions: &[bool]) -> f64 {
        if recent_sessions.len() < 3 {
            return 0.0;
        }

        let success_rate =
            recent_sessions.iter().filter(|&&x| x).count() as f64 / recent_sessions.len() as f64;

        if success_rate > 0.9 {
            0.2 // increase difficulty
        } else if success_rate < 0.5 {
            -0.3 // decrease difficulty
        } else {
            0.0 // no change
        }
    }

    /// Applies an adjustment to the current difficulty, clamping to valid range.
    ///
    /// Difficulty is constrained between 1.0 (easiest) and 5.0 (hardest).
    ///
    /// # Arguments
    ///
    /// * `current` - Current difficulty value
    /// * `adjustment` - Adjustment to apply (can be positive or negative)
    ///
    /// # Examples
    ///
    /// ```
    /// # use kata_sr::core::difficulty::DifficultyTracker;
    /// // Normal adjustment
    /// assert_eq!(DifficultyTracker::apply_adjustment(3.0, 0.2), 3.2);
    ///
    /// // Clamps at maximum (5.0)
    /// assert_eq!(DifficultyTracker::apply_adjustment(4.9, 0.5), 5.0);
    ///
    /// // Clamps at minimum (1.0)
    /// assert_eq!(DifficultyTracker::apply_adjustment(1.2, -0.5), 1.0);
    /// ```
    pub fn apply_adjustment(current: f64, adjustment: f64) -> f64 {
        (current + adjustment).clamp(1.0, 5.0)
    }
}

impl Default for DifficultyTracker {
    fn default() -> Self {
        Self::new(5)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_increase_on_high_success() {
        let tracker = DifficultyTracker::new(5);
        let sessions = vec![true, true, true, true, true];
        assert_eq!(tracker.compute_adjustment(&sessions), 0.2);
    }

    #[test]
    fn test_decrease_on_low_success() {
        let tracker = DifficultyTracker::new(5);
        let sessions = vec![false, false, true, false, false];
        assert_eq!(tracker.compute_adjustment(&sessions), -0.3);
    }

    #[test]
    fn test_no_change_on_moderate_success() {
        let tracker = DifficultyTracker::new(5);
        let sessions = vec![true, false, true, true, false];
        assert_eq!(tracker.compute_adjustment(&sessions), 0.0);
    }

    #[test]
    fn test_insufficient_data() {
        let tracker = DifficultyTracker::new(5);
        assert_eq!(tracker.compute_adjustment(&[true, false]), 0.0);
        assert_eq!(tracker.compute_adjustment(&[true]), 0.0);
        assert_eq!(tracker.compute_adjustment(&[]), 0.0);
    }

    #[test]
    fn test_exactly_threshold() {
        let tracker = DifficultyTracker::new(10);

        // exactly 90% success (9/10)
        let sessions = vec![true, true, true, true, true, true, true, true, true, false];
        assert_eq!(tracker.compute_adjustment(&sessions), 0.0);

        // just over 90% (10/10)
        let sessions = vec![true; 10];
        assert_eq!(tracker.compute_adjustment(&sessions), 0.2);

        // exactly 50% success (5/10)
        let sessions = vec![
            true, true, true, true, true, false, false, false, false, false,
        ];
        assert_eq!(tracker.compute_adjustment(&sessions), 0.0);

        // just under 50% (4/10)
        let sessions = vec![
            true, true, true, true, false, false, false, false, false, false,
        ];
        assert_eq!(tracker.compute_adjustment(&sessions), -0.3);
    }

    #[test]
    fn test_apply_adjustment_normal() {
        assert_eq!(DifficultyTracker::apply_adjustment(3.0, 0.2), 3.2);
        assert_eq!(DifficultyTracker::apply_adjustment(2.5, -0.3), 2.2);
    }

    #[test]
    fn test_apply_adjustment_clamps_minimum() {
        assert_eq!(DifficultyTracker::apply_adjustment(1.0, -0.5), 1.0);
        assert_eq!(DifficultyTracker::apply_adjustment(1.2, -0.5), 1.0);
    }

    #[test]
    fn test_apply_adjustment_clamps_maximum() {
        assert_eq!(DifficultyTracker::apply_adjustment(5.0, 0.5), 5.0);
        assert_eq!(DifficultyTracker::apply_adjustment(4.9, 0.2), 5.0);
    }

    #[test]
    fn test_window_size_ignored_for_compute() {
        // window_size is a property but compute_adjustment works on the slice given
        let tracker_small = DifficultyTracker::new(3);
        let tracker_large = DifficultyTracker::new(10);

        let sessions = vec![true, true, true, true, true];
        assert_eq!(
            tracker_small.compute_adjustment(&sessions),
            tracker_large.compute_adjustment(&sessions)
        );
    }
}
