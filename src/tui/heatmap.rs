//! ASCII visualizations for analytics data.
//!
//! This module provides text-based visualizations for activity heatmaps
//! and category breakdowns using Unicode block characters.

use chrono::{Duration, NaiveDate, Utc};
use std::collections::HashMap;

/// Renders a weekly activity heatmap using Unicode block characters.
///
/// Shows activity for the last 7 days, with intensity represented by:
/// - ░ (light shade): 0 reviews
/// - ▒ (medium shade): 1-2 reviews
/// - ▓ (dark shade): 3-5 reviews
/// - █ (full block): 6+ reviews
///
/// # Arguments
///
/// * `review_counts` - Map of dates to review counts
///
/// # Returns
///
/// A string like "Last 7 days: ░▒▓█▓▒░"
pub fn render_weekly_heatmap(review_counts: &HashMap<NaiveDate, i32>) -> String {
    let today = Utc::now().date_naive();
    let week_start = today - Duration::days(6);

    let mut output = String::from("Last 7 days: ");

    for i in 0..7 {
        let date = week_start + Duration::days(i);
        let count = review_counts.get(&date).unwrap_or(&0);

        let symbol = match count {
            0 => '░',
            1..=2 => '▒',
            3..=5 => '▓',
            _ => '█',
        };

        output.push(symbol);
    }

    output
}

/// Renders a category breakdown as ASCII bar chart with percentages.
///
/// Shows review counts and percentages for each category, sorted by count descending.
/// Each bar uses █ characters, with 5% per character.
///
/// # Arguments
///
/// * `categories` - Map of category names to review counts
///
/// # Returns
///
/// A vector of formatted strings, one per category
///
/// # Example Output
///
/// ```text
/// transformers      10 ( 50.0%) ██████████
/// fundamentals       5 ( 25.0%) █████
/// algorithms         5 ( 25.0%) █████
/// ```
pub fn render_category_breakdown(categories: &HashMap<String, i32>) -> Vec<String> {
    let total: i32 = categories.values().sum();
    let mut lines = Vec::new();

    if total == 0 {
        return lines;
    }

    let mut sorted: Vec<_> = categories.iter().collect();
    sorted.sort_by_key(|(_, count)| -(**count));

    for (category, count) in sorted {
        let percentage = (*count as f64 / total as f64) * 100.0;
        let bar_length = (percentage / 5.0) as usize; // 5% per char
        let bar = "█".repeat(bar_length);

        lines.push(format!(
            "{:15} {:3} ({:>5.1}%) {}",
            category, count, percentage, bar
        ));
    }

    lines
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_heatmap_rendering() {
        let mut counts = HashMap::new();
        let today = Utc::now().date_naive();

        counts.insert(today, 5);
        counts.insert(today - Duration::days(1), 2);
        counts.insert(today - Duration::days(3), 0);

        let heatmap = render_weekly_heatmap(&counts);

        assert!(heatmap.contains("Last 7 days"));
        // check character count (not byte length, since Unicode symbols are multi-byte)
        assert_eq!(heatmap.chars().count(), "Last 7 days: ".len() + 7);

        // check that it contains some symbols
        assert!(
            heatmap.contains('░')
                || heatmap.contains('▒')
                || heatmap.contains('▓')
                || heatmap.contains('█')
        );
    }

    #[test]
    fn test_heatmap_all_zeros() {
        let counts = HashMap::new();
        let heatmap = render_weekly_heatmap(&counts);

        // should be all light shade (░) for empty days
        assert_eq!(heatmap, "Last 7 days: ░░░░░░░");
    }

    #[test]
    fn test_heatmap_all_high() {
        let mut counts = HashMap::new();
        let today = Utc::now().date_naive();

        for i in 0..7 {
            counts.insert(today - Duration::days(i), 10);
        }

        let heatmap = render_weekly_heatmap(&counts);

        // should be all full blocks (█)
        assert_eq!(heatmap, "Last 7 days: ███████");
    }

    #[test]
    fn test_category_breakdown() {
        let mut categories = HashMap::new();
        categories.insert("transformers".to_string(), 10);
        categories.insert("graphs".to_string(), 5);

        let lines = render_category_breakdown(&categories);

        assert_eq!(lines.len(), 2);
        assert!(lines[0].contains("transformers"));
        assert!(lines[1].contains("graphs"));
        assert!(lines[0].contains("66.7%"));
        assert!(lines[1].contains("33.3%"));
    }

    #[test]
    fn test_category_breakdown_sorted() {
        let mut categories = HashMap::new();
        categories.insert("a".to_string(), 5);
        categories.insert("b".to_string(), 10);
        categories.insert("c".to_string(), 2);

        let lines = render_category_breakdown(&categories);

        assert_eq!(lines.len(), 3);
        // should be sorted by count descending
        assert!(lines[0].contains("b"));
        assert!(lines[1].contains("a"));
        assert!(lines[2].contains("c"));
    }

    #[test]
    fn test_category_breakdown_empty() {
        let categories = HashMap::new();
        let lines = render_category_breakdown(&categories);

        assert_eq!(lines.len(), 0);
    }

    #[test]
    fn test_category_breakdown_bar_length() {
        let mut categories = HashMap::new();
        categories.insert("full".to_string(), 100);

        let lines = render_category_breakdown(&categories);

        assert_eq!(lines.len(), 1);
        // 100% should give 20 bars (100/5)
        assert_eq!(lines[0].matches('█').count(), 20);
    }
}
