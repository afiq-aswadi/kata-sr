//! Analytics system for computing and aggregating kata statistics.
//!
//! This module provides the Analytics struct which computes daily statistics,
//! streak calculations, success rates, and category breakdowns from session data.

use crate::db::repo::{DailyStats, KataRepository};
use chrono::{Duration, NaiveDate, Utc};
use std::collections::HashMap;

/// Analytics engine for computing kata statistics and aggregations.
///
/// Provides methods for computing daily stats, streaks, success rates,
/// and category breakdowns from session history.
pub struct Analytics<'a> {
    repo: &'a KataRepository,
}

impl<'a> Analytics<'a> {
    /// Creates a new Analytics instance.
    ///
    /// # Arguments
    ///
    /// * `repo` - Repository for accessing kata and session data
    pub fn new(repo: &'a KataRepository) -> Self {
        Self { repo }
    }

    /// Computes daily statistics for a specific date.
    ///
    /// Aggregates all sessions completed on the given date, calculating:
    /// - Total reviews and successes
    /// - Success rate (quality_rating >= 2, i.e., Hard/Good/Easy in FSRS)
    /// - Streak up to this date
    /// - Category breakdown (JSON)
    ///
    /// # Arguments
    ///
    /// * `date` - The date to compute stats for
    pub fn compute_daily_stats(&self, date: NaiveDate) -> anyhow::Result<DailyStats> {
        let date_str = date.format("%Y-%m-%d").to_string();

        // query all sessions completed on this date
        let sessions = self.repo.get_sessions_for_date(&date_str)?;

        let total_reviews = sessions.len() as i32;
        let total_successes = sessions
            .iter()
            .filter(|s| s.quality_rating.map(|r| r >= 2).unwrap_or(false))
            .count() as i32;

        let success_rate = if total_reviews > 0 {
            total_successes as f64 / total_reviews as f64
        } else {
            0.0
        };

        // group by category
        let mut categories: HashMap<String, i32> = HashMap::new();
        for session in &sessions {
            if let Some(kata) = self.repo.get_kata_by_id(session.kata_id)? {
                *categories.entry(kata.category).or_insert(0) += 1;
            }
        }

        // compute streak
        let streak_days = self.compute_streak_up_to(date)?;

        Ok(DailyStats {
            date: date_str,
            total_reviews,
            total_successes,
            success_rate,
            streak_days,
            categories_json: serde_json::to_string(&categories)?,
        })
    }

    /// Computes the streak of consecutive days with reviews up to the given date.
    ///
    /// Counts backwards from the given date, stopping at the first day with no sessions.
    ///
    /// # Arguments
    ///
    /// * `end_date` - The date to compute streak up to (inclusive)
    pub fn compute_streak_up_to(&self, end_date: NaiveDate) -> anyhow::Result<i32> {
        let mut streak = 0;
        let mut current_date = end_date;

        loop {
            let date_str = current_date.format("%Y-%m-%d").to_string();
            let sessions = self.repo.get_sessions_for_date(&date_str)?;

            if sessions.is_empty() {
                break;
            }

            streak += 1;
            current_date = current_date
                .pred_opt()
                .ok_or_else(|| anyhow::anyhow!("date underflow"))?;
        }

        Ok(streak)
    }

    /// Gets success rate over the last N days.
    ///
    /// Success is defined as quality_rating >= 2 (Good or Easy).
    ///
    /// # Arguments
    ///
    /// * `n` - Number of days to look back
    pub fn get_success_rate_last_n_days(&self, n: usize) -> anyhow::Result<f64> {
        Ok(self.repo.get_success_rate_last_n_days(n)?)
    }

    /// Updates daily statistics for today.
    ///
    /// Computes stats for the current date and upserts into the database.
    /// This should be called after each session to keep stats current.
    pub fn update_daily_stats(&self) -> anyhow::Result<()> {
        let today = Utc::now().date_naive();
        let stats = self.compute_daily_stats(today)?;
        self.repo.upsert_daily_stats(&stats)?;
        Ok(())
    }

    /// Gets category breakdown for a specific date.
    ///
    /// Returns a map of category names to review counts.
    ///
    /// # Arguments
    ///
    /// * `date` - Date to get category breakdown for
    pub fn get_category_breakdown(&self, date: NaiveDate) -> anyhow::Result<HashMap<String, i32>> {
        let date_str = date.format("%Y-%m-%d").to_string();

        if let Some(stats) = self.repo.get_daily_stats(&date_str)? {
            let categories: HashMap<String, i32> = serde_json::from_str(&stats.categories_json)?;
            Ok(categories)
        } else {
            // compute on the fly if not cached
            let sessions = self.repo.get_sessions_for_date(&date_str)?;
            let mut categories: HashMap<String, i32> = HashMap::new();

            for session in &sessions {
                if let Some(kata) = self.repo.get_kata_by_id(session.kata_id)? {
                    *categories.entry(kata.category).or_insert(0) += 1;
                }
            }

            Ok(categories)
        }
    }

    /// Gets review counts for the last N days.
    ///
    /// Returns a map of dates to review counts.
    ///
    /// # Arguments
    ///
    /// * `n` - Number of days to look back (inclusive of today)
    pub fn get_review_counts_last_n_days(&self, n: i32) -> anyhow::Result<HashMap<NaiveDate, i32>> {
        let today = Utc::now().date_naive();
        let mut counts = HashMap::new();

        for i in 0..n {
            let date = today - Duration::days(i as i64);
            let date_str = date.format("%Y-%m-%d").to_string();
            let sessions = self.repo.get_sessions_for_date(&date_str)?;
            counts.insert(date, sessions.len() as i32);
        }

        Ok(counts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::repo::{NewKata, NewSession};
    use chrono::Utc;

    #[test]
    fn test_compute_streak_single_day() -> anyhow::Result<()> {
        let repo = KataRepository::new_in_memory()?;
        repo.run_migrations()?;

        // create a kata
        let kata_id = repo.create_kata(
            &NewKata {
                name: "test".to_string(),
                category: "test".to_string(),
                description: "test".to_string(),
                base_difficulty: 3,
                parent_kata_id: None,
                variation_params: None,
            },
            Utc::now(),
        )?;

        // create a session today
        let now = Utc::now();
        repo.create_session(&NewSession {
            kata_id,
            started_at: now,
            completed_at: Some(now),
            test_results_json: None,
            num_passed: Some(5),
            num_failed: Some(0),
            num_skipped: Some(0),
            duration_ms: Some(100),
            quality_rating: Some(3), // Good (FSRS)
            code_attempt: None,
        })?;

        let analytics = Analytics::new(&repo);
        let streak = analytics.compute_streak_up_to(Utc::now().date_naive())?;

        assert_eq!(streak, 1);
        Ok(())
    }

    #[test]
    fn test_compute_streak_gap() -> anyhow::Result<()> {
        let repo = KataRepository::new_in_memory()?;
        repo.run_migrations()?;

        let kata_id = repo.create_kata(
            &NewKata {
                name: "test".to_string(),
                category: "test".to_string(),
                description: "test".to_string(),
                base_difficulty: 3,
                parent_kata_id: None,
                variation_params: None,
            },
            Utc::now(),
        )?;

        // create session 3 days ago (gap in between)
        let three_days_ago = Utc::now() - Duration::days(3);
        repo.create_session(&NewSession {
            kata_id,
            started_at: three_days_ago,
            completed_at: Some(three_days_ago),
            test_results_json: None,
            num_passed: Some(5),
            num_failed: Some(0),
            num_skipped: Some(0),
            duration_ms: Some(100),
            quality_rating: Some(3), // Good (FSRS)
            code_attempt: None,
        })?;

        let analytics = Analytics::new(&repo);
        // streak should be 0 since there's a gap
        let streak = analytics.compute_streak_up_to(Utc::now().date_naive())?;

        assert_eq!(streak, 0);
        Ok(())
    }

    #[test]
    fn test_compute_daily_stats() -> anyhow::Result<()> {
        let repo = KataRepository::new_in_memory()?;
        repo.run_migrations()?;

        let kata_id = repo.create_kata(
            &NewKata {
                name: "test".to_string(),
                category: "fundamentals".to_string(),
                description: "test".to_string(),
                base_difficulty: 3,
                parent_kata_id: None,
                variation_params: None,
            },
            Utc::now(),
        )?;

        let now = Utc::now();

        // create 3 successful sessions
        for _ in 0..3 {
            repo.create_session(&NewSession {
                kata_id,
                started_at: now,
                completed_at: Some(now),
                test_results_json: None,
                num_passed: Some(5),
                num_failed: Some(0),
                num_skipped: Some(0),
                duration_ms: Some(100),
                quality_rating: Some(3), // Good (FSRS),
            code_attempt: None,
            })?;
        }

        // create 1 failed session
        repo.create_session(&NewSession {
            kata_id,
            started_at: now,
            completed_at: Some(now),
            test_results_json: None,
            num_passed: Some(2),
            num_failed: Some(3),
            num_skipped: Some(0),
            duration_ms: Some(100),
            quality_rating: Some(1), // Again (FSRS)
            code_attempt: None,
        })?;

        let analytics = Analytics::new(&repo);
        let stats = analytics.compute_daily_stats(Utc::now().date_naive())?;

        assert_eq!(stats.total_reviews, 4);
        assert_eq!(stats.total_successes, 3);
        assert_eq!(stats.success_rate, 0.75);
        assert_eq!(stats.streak_days, 1);

        // check category breakdown
        let categories: HashMap<String, i32> = serde_json::from_str(&stats.categories_json)?;
        assert_eq!(categories.get("fundamentals"), Some(&4));

        Ok(())
    }

    #[test]
    fn test_update_daily_stats() -> anyhow::Result<()> {
        let repo = KataRepository::new_in_memory()?;
        repo.run_migrations()?;

        let kata_id = repo.create_kata(
            &NewKata {
                name: "test".to_string(),
                category: "test".to_string(),
                description: "test".to_string(),
                base_difficulty: 3,
                parent_kata_id: None,
                variation_params: None,
            },
            Utc::now(),
        )?;

        repo.create_session(&NewSession {
            kata_id,
            started_at: Utc::now(),
            completed_at: Some(Utc::now()),
            test_results_json: None,
            num_passed: Some(5),
            num_failed: Some(0),
            num_skipped: Some(0),
            duration_ms: Some(100),
            quality_rating: Some(3), // Good (FSRS)
            code_attempt: None,
        })?;

        let analytics = Analytics::new(&repo);
        analytics.update_daily_stats()?;

        // verify stats were inserted
        let today = Utc::now().date_naive().format("%Y-%m-%d").to_string();
        let stats = repo.get_daily_stats(&today)?;

        assert!(stats.is_some());
        let stats = stats.unwrap();
        assert_eq!(stats.total_reviews, 1);
        assert_eq!(stats.total_successes, 1);

        Ok(())
    }

    #[test]
    fn test_get_review_counts_last_n_days() -> anyhow::Result<()> {
        let repo = KataRepository::new_in_memory()?;
        repo.run_migrations()?;

        let kata_id = repo.create_kata(
            &NewKata {
                name: "test".to_string(),
                category: "test".to_string(),
                description: "test".to_string(),
                base_difficulty: 3,
                parent_kata_id: None,
                variation_params: None,
            },
            Utc::now(),
        )?;

        // create sessions today and yesterday
        let now = Utc::now();
        let yesterday = now - Duration::days(1);

        repo.create_session(&NewSession {
            kata_id,
            started_at: now,
            completed_at: Some(now),
            test_results_json: None,
            num_passed: Some(5),
            num_failed: Some(0),
            num_skipped: Some(0),
            duration_ms: Some(100),
            quality_rating: Some(3), // Good (FSRS)
            code_attempt: None,
        })?;

        repo.create_session(&NewSession {
            kata_id,
            started_at: yesterday,
            completed_at: Some(yesterday),
            test_results_json: None,
            num_passed: Some(5),
            num_failed: Some(0),
            num_skipped: Some(0),
            duration_ms: Some(100),
            quality_rating: Some(3), // Good (FSRS)
            code_attempt: None,
        })?;

        let analytics = Analytics::new(&repo);
        let counts = analytics.get_review_counts_last_n_days(7)?;

        assert_eq!(counts.get(&Utc::now().date_naive()), Some(&1));
        assert_eq!(
            counts.get(&(Utc::now().date_naive() - Duration::days(1))),
            Some(&1)
        );

        Ok(())
    }
}
