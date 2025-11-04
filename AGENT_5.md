# Agent 5: Analytics & Integration

## Mission

Build the analytics system, aggregate statistics, implement ASCII visualizations, and perform end-to-end integration testing. This is the final piece that ties everything together and ensures the system works smoothly.

## Dependencies

**Requires all other agents:**
- Agent 1: Database schema, repository layer
- Agent 2: Python runner
- Agent 3: TUI application
- Agent 4: Example katas for testing

You're the last agent. Wait for others to complete before starting integration work.

## What You're Building

### 1. Statistics Aggregation
Daily stats calculation and caching

### 2. ASCII Visualizations
Heatmaps and simple charts for the dashboard

### 3. Analytics Queries
Streak calculation, success rates, category breakdowns

### 4. Integration Testing
End-to-end workflow testing

### 5. Polish & Error Handling
Edge cases, helpful error messages, installation docs

## Detailed Specifications

### Daily Statistics Aggregation

```rust
// src/core/analytics.rs

use crate::db::repo::KataRepository;
use chrono::{DateTime, Utc, Duration, Datelike};
use std::collections::HashMap;

pub struct Analytics {
    repo: KataRepository,
}

impl Analytics {
    pub fn new(repo: KataRepository) -> Self {
        Self { repo }
    }

    pub fn compute_daily_stats(&self, date: chrono::NaiveDate) -> anyhow::Result<DailyStats> {
        // Query all sessions completed on this date
        let sessions = self.repo.get_sessions_for_date(date)?;

        let total_reviews = sessions.len() as i32;
        let total_successes = sessions.iter()
            .filter(|s| s.quality_rating.map(|r| r >= 2).unwrap_or(false))
            .count() as i32;

        let success_rate = if total_reviews > 0 {
            total_successes as f64 / total_reviews as f64
        } else {
            0.0
        };

        // Group by category
        let mut categories: HashMap<String, i32> = HashMap::new();
        for session in &sessions {
            let kata = self.repo.get_kata_by_id(session.kata_id)?
                .ok_or_else(|| anyhow::anyhow!("Kata not found"))?;
            *categories.entry(kata.category).or_insert(0) += 1;
        }

        // Compute streak
        let streak_days = self.compute_streak_up_to(date)?;

        Ok(DailyStats {
            date: date.format("%Y-%m-%d").to_string(),
            total_reviews,
            total_successes,
            success_rate,
            streak_days,
            categories_json: serde_json::to_string(&categories)?,
        })
    }

    pub fn compute_streak_up_to(&self, end_date: chrono::NaiveDate) -> anyhow::Result<i32> {
        let mut streak = 0;
        let mut current_date = end_date;

        loop {
            let sessions = self.repo.get_sessions_for_date(current_date)?;
            if sessions.is_empty() {
                break;
            }

            streak += 1;
            current_date = current_date.pred_opt()
                .ok_or_else(|| anyhow::anyhow!("Date underflow"))?;
        }

        Ok(streak)
    }

    pub fn get_success_rate_last_n_days(&self, n: i32) -> anyhow::Result<f64> {
        let end = Utc::now().date_naive();
        let start = end - Duration::days(n as i64);

        let sessions = self.repo.get_sessions_between_dates(start, end)?;

        if sessions.is_empty() {
            return Ok(0.0);
        }

        let successes = sessions.iter()
            .filter(|s| s.quality_rating.map(|r| r >= 2).unwrap_or(false))
            .count();

        Ok(successes as f64 / sessions.len() as f64)
    }

    pub fn update_daily_stats(&self) -> anyhow::Result<()> {
        // Update stats for today
        let today = Utc::now().date_naive();
        let stats = self.compute_daily_stats(today)?;
        self.repo.upsert_daily_stats(&stats)?;
        Ok(())
    }
}

pub struct DailyStats {
    pub date: String,
    pub total_reviews: i32,
    pub total_successes: i32,
    pub success_rate: f64,
    pub streak_days: i32,
    pub categories_json: String,
}
```

### ASCII Heatmap Visualization

```rust
// src/tui/heatmap.rs

use chrono::{NaiveDate, Duration, Datelike};
use std::collections::HashMap;

pub fn render_weekly_heatmap(review_counts: &HashMap<NaiveDate, i32>) -> String {
    let today = chrono::Utc::now().date_naive();
    let week_start = today - Duration::days(6);

    let mut output = String::new();
    output.push_str("Last 7 days: ");

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

pub fn render_category_breakdown(categories: &HashMap<String, i32>) -> Vec<String> {
    let total: i32 = categories.values().sum();
    let mut lines = Vec::new();

    let mut sorted: Vec<_> = categories.iter().collect();
    sorted.sort_by_key(|(_, count)| -(**count));

    for (category, count) in sorted {
        let percentage = if total > 0 {
            (*count as f64 / total as f64) * 100.0
        } else {
            0.0
        };

        let bar_length = (percentage / 5.0) as usize;  // 5% per char
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
        let today = chrono::Utc::now().date_naive();

        counts.insert(today, 5);
        counts.insert(today - Duration::days(1), 2);
        counts.insert(today - Duration::days(3), 0);

        let heatmap = render_weekly_heatmap(&counts);
        assert!(heatmap.contains("Last 7 days"));
        assert!(heatmap.len() > 15);  // Should have symbols for each day
    }

    #[test]
    fn test_category_breakdown() {
        let mut categories = HashMap::new();
        categories.insert("transformers".to_string(), 10);
        categories.insert("graphs".to_string(), 5);

        let lines = render_category_breakdown(&categories);
        assert_eq!(lines.len(), 2);
        assert!(lines[0].contains("transformers"));
    }
}
```

### Enhanced Dashboard with Analytics

```rust
// src/tui/dashboard.rs (additions)

use crate::core::analytics::Analytics;
use crate::tui::heatmap::{render_weekly_heatmap, render_category_breakdown};

impl Dashboard {
    pub fn load_with_analytics(repo: &KataRepository) -> anyhow::Result<Self> {
        // Existing loading logic...

        let analytics = Analytics::new(repo.clone());

        // Get review counts for heatmap
        let today = chrono::Utc::now().date_naive();
        let mut review_counts = HashMap::new();

        for i in 0..7 {
            let date = today - chrono::Duration::days(i);
            let sessions = repo.get_sessions_for_date(date)?;
            review_counts.insert(date, sessions.len() as i32);
        }

        let heatmap = render_weekly_heatmap(&review_counts);

        // Get category breakdown from recent daily stats
        let daily_stats = repo.get_daily_stats(today)?;
        let categories: HashMap<String, i32> = if let Some(stats) = daily_stats {
            serde_json::from_str(&stats.categories_json)?
        } else {
            HashMap::new()
        };

        let category_lines = render_category_breakdown(&categories);

        // Update stats struct
        let stats = DashboardStats {
            streak_days: analytics.compute_streak_up_to(today)?,
            total_reviews_today: review_counts.get(&today).unwrap_or(&0).clone(),
            success_rate_7d: analytics.get_success_rate_last_n_days(7)?,
            heatmap,
            category_breakdown: category_lines,
        };

        Ok(Self {
            katas_due,
            locked_katas,
            selected_index: 0,
            stats,
        })
    }

    pub fn render_with_analytics(&self, frame: &mut Frame) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3),   // Header
                Constraint::Min(8),      // Main kata list
                Constraint::Length(3),   // Heatmap
                Constraint::Length(8),   // Category breakdown
                Constraint::Length(3),   // Stats summary
            ])
            .split(frame.size());

        // ... existing header and list rendering ...

        // Heatmap
        let heatmap = Paragraph::new(self.stats.heatmap.clone())
            .block(Block::default().borders(Borders::ALL).title("Activity"));
        frame.render_widget(heatmap, chunks[2]);

        // Category breakdown
        let category_text = self.stats.category_breakdown.join("\n");
        let categories = Paragraph::new(category_text)
            .block(Block::default().borders(Borders::ALL).title("Categories"));
        frame.render_widget(categories, chunks[3]);

        // Stats summary (existing)
        // ...
    }
}
```

### Integration Tests

```rust
// tests/integration_test.rs

use kata_sr::{KataRepository, Analytics, SM2State, QualityRating};
use std::path::PathBuf;
use tempfile::tempdir;

#[test]
fn test_end_to_end_workflow() {
    // Create temp database
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");

    let repo = KataRepository::new(db_path.to_str().unwrap()).unwrap();
    repo.run_migrations().unwrap();

    // Create a test kata
    let kata = NewKata {
        name: "test_kata".to_string(),
        category: "test".to_string(),
        description: "Test kata".to_string(),
        base_difficulty: 3,
        parent_kata_id: None,
        variation_params: None,
    };

    let kata_id = repo.create_kata(&kata).unwrap();

    // Simulate a practice session
    let session = Session {
        kata_id,
        started_at: chrono::Utc::now(),
        completed_at: Some(chrono::Utc::now()),
        test_results_json: Some(r#"{"passed": true, "num_passed": 5}"#.to_string()),
        num_passed: Some(5),
        num_failed: Some(0),
        num_skipped: Some(0),
        duration_ms: Some(100),
        quality_rating: Some(2),  // Good
    };

    repo.create_session(&session).unwrap();

    // Update SM-2 state
    let mut state = SM2State::new();
    let interval = state.update(QualityRating::Good);
    let next_review = chrono::Utc::now() + chrono::Duration::days(interval);

    repo.update_kata_after_review(kata_id, &state, next_review).unwrap();

    // Verify kata is not due yet
    let katas_due = repo.get_katas_due(chrono::Utc::now()).unwrap();
    assert_eq!(katas_due.len(), 0);

    // Verify kata will be due in the future
    let katas_due_future = repo.get_katas_due(next_review + chrono::Duration::hours(1)).unwrap();
    assert_eq!(katas_due_future.len(), 1);

    // Test analytics
    let analytics = Analytics::new(repo);
    let streak = analytics.compute_streak_up_to(chrono::Utc::now().date_naive()).unwrap();
    assert_eq!(streak, 1);
}

#[test]
fn test_dependency_unlocking() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");

    let repo = KataRepository::new(db_path.to_str().unwrap()).unwrap();
    repo.run_migrations().unwrap();

    // Create two katas with dependency
    let kata1_id = repo.create_kata(&NewKata {
        name: "basic".to_string(),
        category: "test".to_string(),
        description: "Basic kata".to_string(),
        base_difficulty: 1,
        parent_kata_id: None,
        variation_params: None,
    }).unwrap();

    let kata2_id = repo.create_kata(&NewKata {
        name: "advanced".to_string(),
        category: "test".to_string(),
        description: "Advanced kata".to_string(),
        base_difficulty: 3,
        parent_kata_id: None,
        variation_params: None,
    }).unwrap();

    // Add dependency: kata2 depends on kata1
    repo.add_dependency(kata2_id, kata1_id, 1).unwrap();

    // Load dependency graph
    let graph = repo.load_dependency_graph().unwrap();
    let success_counts = repo.get_success_counts().unwrap();

    // kata2 should be locked
    assert!(!graph.is_unlocked(kata2_id, &success_counts));

    // Complete kata1 successfully
    repo.create_session(&Session {
        kata_id: kata1_id,
        started_at: chrono::Utc::now(),
        completed_at: Some(chrono::Utc::now()),
        test_results_json: None,
        num_passed: Some(5),
        num_failed: Some(0),
        num_skipped: Some(0),
        duration_ms: Some(100),
        quality_rating: Some(2),
    }).unwrap();

    // Reload success counts
    let success_counts = repo.get_success_counts().unwrap();

    // kata2 should now be unlocked
    assert!(graph.is_unlocked(kata2_id, &success_counts));
}
```

### CLI Enhancements

```rust
// src/main.rs (additions)

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "kata-sr")]
#[command(about = "Spaced repetition for coding patterns", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Launch TUI application (default)
    Practice,

    /// Add a new kata from manifest
    Add {
        #[arg(value_name = "PATH")]
        kata_path: PathBuf,
    },

    /// List all katas
    List,

    /// Show statistics
    Stats,

    /// Update daily stats (normally runs automatically)
    UpdateStats,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    // Setup Python environment
    let python_env = PythonEnv::setup()?;

    // Open database
    let db_path = get_db_path()?;
    let repo = KataRepository::new(db_path)?;
    repo.run_migrations()?;

    match cli.command {
        None | Some(Commands::Practice) => {
            // Launch TUI
            let mut app = App::new(repo);
            app.run()?;
        }
        Some(Commands::Add { kata_path }) => {
            add_kata(&repo, &kata_path)?;
        }
        Some(Commands::List) => {
            list_katas(&repo)?;
        }
        Some(Commands::Stats) => {
            show_stats(&repo)?;
        }
        Some(Commands::UpdateStats) => {
            let analytics = Analytics::new(repo);
            analytics.update_daily_stats()?;
            println!("Daily stats updated");
        }
    }

    Ok(())
}

fn get_db_path() -> anyhow::Result<String> {
    let home = std::env::var("HOME")?;
    let db_dir = PathBuf::from(home).join(".local/share/kata-sr");
    std::fs::create_dir_all(&db_dir)?;
    Ok(db_dir.join("kata.db").to_string_lossy().to_string())
}

fn add_kata(repo: &KataRepository, kata_path: &PathBuf) -> anyhow::Result<()> {
    // Parse manifest
    let manifest_path = kata_path.join("manifest.toml");
    // ... use toml parsing or call Python manifest_parser ...

    println!("✓ Added kata: {}", "kata_name");
    Ok(())
}

fn list_katas(repo: &KataRepository) -> anyhow::Result<()> {
    let katas = repo.get_all_katas()?;
    println!("Total katas: {}", katas.len());
    for kata in katas {
        println!("  - {} (difficulty: {})", kata.name, kata.current_difficulty);
    }
    Ok(())
}

fn show_stats(repo: &KataRepository) -> anyhow::Result<()> {
    let analytics = Analytics::new(repo);
    let streak = analytics.compute_streak_up_to(chrono::Utc::now().date_naive())?;
    let success_rate = analytics.get_success_rate_last_n_days(7)?;

    println!("Current streak: {} days", streak);
    println!("7-day success rate: {:.1}%", success_rate * 100.0);

    Ok(())
}
```

## File Structure You'll Create

```
src/
├── core/
│   └── analytics.rs        # Statistics and aggregation
└── tui/
    └── heatmap.rs          # ASCII visualizations

tests/
└── integration_test.rs     # End-to-end tests
```

## Testing Requirements

Run all integration tests:

```bash
cargo test --test integration_test
```

Manual testing workflow:
1. Install with `cargo install --path .`
2. Run `kata-sr` and verify TUI launches
3. Complete a kata and verify SM-2 scheduling
4. Check analytics update correctly
5. Test dependency unlocking
6. Verify edge cases (empty database, no katas due, etc.)

## Acceptance Criteria

- [ ] Analytics compute daily stats correctly
- [ ] Streak calculation works (handles gaps correctly)
- [ ] ASCII heatmap renders activity
- [ ] Category breakdown shows percentages
- [ ] Integration tests pass
- [ ] CLI subcommands work (add, list, stats)
- [ ] Error messages are helpful
- [ ] README with installation and usage instructions
- [ ] System handles edge cases gracefully

## Documentation

Create README.md:

```markdown
# Kata Spaced Repetition

Personal TUI tool for practicing coding patterns using spaced repetition.

## Installation

1. Install Rust and uv
2. Clone this repo
3. Run: cargo install --path .

## Usage

kata-sr              # Launch TUI
kata-sr add <path>   # Add new kata
kata-sr list         # List all katas
kata-sr stats        # Show statistics

## Adding Katas

See katas/exercises/ for examples.
Each kata needs: manifest.toml, template.py, test_kata.py, reference.py
```

## Final Integration Checklist

- [ ] All components work together
- [ ] No panics or unwraps in production code paths
- [ ] Database migrations are idempotent
- [ ] Python environment bootstraps automatically
- [ ] TUI is responsive (no blocking)
- [ ] Test results display correctly
- [ ] Rating updates schedule properly
- [ ] Analytics refresh on dashboard reload
- [ ] Dependencies unlock correctly
- [ ] Variations appear in UI

## Notes

- Focus on polish and user experience
- Error messages should guide users to solutions
- Test with empty database (first-run experience)
- Test with large number of katas (performance)
- Ensure streak doesn't break across dates
- Handle timezones correctly (use UTC internally)
