use crate::core::analytics::Analytics;
use crate::db::repo::{Kata, KataRepository};
use crate::tui::heatmap::render_category_breakdown;
use crate::tui::heatmap_calendar::HeatmapCalendar;
use chrono::Utc;
use crossterm::event::KeyCode;
use rand::seq::SliceRandom;
use ratatui::{
    layout::{Constraint, Direction, Layout},
    widgets::{Block, Borders, List, ListItem, Paragraph},
    Frame,
};

/// Sort mode for review dashboard
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ReviewSortMode {
    DueDate,         // Default: prioritize most overdue
    EasiestFirst,    // Sort by difficulty ascending
    HardestFirst,    // Sort by difficulty descending
    Category,        // Group by category
    LeastReviewed,   // Sort by last_reviewed_at (oldest first)
    Shuffle,         // Random order
}

impl ReviewSortMode {
    fn next(&self) -> Self {
        match self {
            Self::DueDate => Self::EasiestFirst,
            Self::EasiestFirst => Self::HardestFirst,
            Self::HardestFirst => Self::Category,
            Self::Category => Self::LeastReviewed,
            Self::LeastReviewed => Self::Shuffle,
            Self::Shuffle => Self::DueDate,
        }
    }

    fn as_str(&self) -> &str {
        match self {
            Self::DueDate => "Due Date",
            Self::EasiestFirst => "Easiest First",
            Self::HardestFirst => "Hardest First",
            Self::Category => "Category",
            Self::LeastReviewed => "Least Recently Reviewed",
            Self::Shuffle => "Shuffle",
        }
    }
}

pub struct Dashboard {
    pub katas_due: Vec<Kata>,
    pub locked_katas: Vec<(Kata, String)>,
    pub selected_index: usize,
    pub stats: DashboardStats,
    pub heatmap_calendar: HeatmapCalendar,
    pub sort_mode: ReviewSortMode,
}

pub struct DashboardStats {
    pub streak_days: i32,
    pub total_reviews_today: i32,
    pub success_rate_7d: f64,
    pub category_breakdown: Vec<String>,
}

impl Dashboard {
    pub fn load(repo: &KataRepository) -> anyhow::Result<Self> {
        let now = Utc::now();
        let katas_due = repo.get_katas_due(now)?;

        let dep_graph = repo.load_dependency_graph()?;
        let success_counts = repo.get_success_counts()?;
        let all_katas = repo.get_all_katas()?;

        let locked_katas = all_katas
            .into_iter()
            .filter_map(|k| {
                if !dep_graph.is_unlocked(k.id, &success_counts) {
                    let blocking = dep_graph.get_blocking_dependencies(k.id, &success_counts);
                    let reason = if !blocking.is_empty() {
                        format!(
                            "Requires: kata {} ({} more needed)",
                            blocking[0].0,
                            blocking[0].1 - blocking[0].2
                        )
                    } else {
                        "Locked".to_string()
                    };
                    Some((k, reason))
                } else {
                    None
                }
            })
            .collect();

        // compute analytics
        let analytics = Analytics::new(repo);

        // get category breakdown from today
        let today = Utc::now().date_naive();
        let categories = analytics.get_category_breakdown(today)?;
        let category_breakdown = render_category_breakdown(&categories);

        let stats = DashboardStats {
            streak_days: repo.get_current_streak()?,
            total_reviews_today: repo.get_reviews_count_today()?,
            success_rate_7d: repo.get_success_rate_last_n_days(7)?,
            category_breakdown,
        };

        // Create GitHub-style heatmap calendar
        let heatmap_calendar = HeatmapCalendar::new(repo)?;

        Ok(Self {
            katas_due,
            locked_katas,
            selected_index: 0,
            stats,
            heatmap_calendar,
            sort_mode: ReviewSortMode::DueDate, // Default sort mode
        })
    }

    /// Apply current sort mode to the katas_due list
    pub fn apply_sort(&mut self) {
        use rand::thread_rng;

        match self.sort_mode {
            ReviewSortMode::DueDate => {
                // Sort by next_review_at (most overdue first)
                // NULL values (never reviewed) should come first
                self.katas_due.sort_by(|a, b| {
                    match (a.next_review_at, b.next_review_at) {
                        (None, None) => std::cmp::Ordering::Equal,
                        (None, Some(_)) => std::cmp::Ordering::Less,
                        (Some(_), None) => std::cmp::Ordering::Greater,
                        (Some(a_time), Some(b_time)) => a_time.cmp(&b_time),
                    }
                });
            }
            ReviewSortMode::EasiestFirst => {
                // Sort by current_difficulty ascending
                self.katas_due.sort_by(|a, b| {
                    a.current_difficulty
                        .partial_cmp(&b.current_difficulty)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            }
            ReviewSortMode::HardestFirst => {
                // Sort by current_difficulty descending
                self.katas_due.sort_by(|a, b| {
                    b.current_difficulty
                        .partial_cmp(&a.current_difficulty)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            }
            ReviewSortMode::Category => {
                // Sort by category, then by name within each category
                self.katas_due.sort_by(|a, b| {
                    a.category
                        .cmp(&b.category)
                        .then_with(|| a.name.cmp(&b.name))
                });
            }
            ReviewSortMode::LeastReviewed => {
                // Sort by last_reviewed_at (oldest first, never reviewed last)
                self.katas_due.sort_by(|a, b| {
                    match (a.last_reviewed_at, b.last_reviewed_at) {
                        (None, None) => std::cmp::Ordering::Equal,
                        (None, Some(_)) => std::cmp::Ordering::Greater,
                        (Some(_), None) => std::cmp::Ordering::Less,
                        (Some(a_time), Some(b_time)) => a_time.cmp(&b_time),
                    }
                });
            }
            ReviewSortMode::Shuffle => {
                // Randomly shuffle the list
                let mut rng = thread_rng();
                self.katas_due.shuffle(&mut rng);
            }
        }

        // Reset selection to first item after sorting
        self.selected_index = 0;
    }

    /// Cycle to the next sort mode and apply it
    pub fn cycle_sort_mode(&mut self) {
        self.sort_mode = self.sort_mode.next();
        self.apply_sort();
    }

    pub fn render(&self, frame: &mut Frame) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3), // Header
                Constraint::Min(8),    // Main kata list
                Constraint::Length(12), // GitHub-style heatmap calendar (7 days + header + legend + borders)
                Constraint::Length(std::cmp::max(
                    3,
                    self.stats.category_breakdown.len() as u16 + 2,
                )), // Category breakdown (dynamic)
                Constraint::Length(4), // Stats summary
            ])
            .split(frame.size());

        let header = Paragraph::new(format!(
            "Kata Spaced Repetition - {} katas due today | Sort: {} (press 's' to change)",
            self.katas_due.len(),
            self.sort_mode.as_str()
        ))
        .block(Block::default().borders(Borders::ALL));
        frame.render_widget(header, chunks[0]);

        let items: Vec<ListItem> = self
            .katas_due
            .iter()
            .enumerate()
            .map(|(i, kata)| {
                let marker = if i == self.selected_index { ">" } else { " " };
                let flag_indicator = if kata.is_problematic { "⚠️ " } else { "" };
                let text = format!(
                    "{} {}{}  (difficulty: {:.1})",
                    marker, flag_indicator, kata.name, kata.current_difficulty
                );
                ListItem::new(text)
            })
            .collect();

        let list =
            List::new(items).block(Block::default().borders(Borders::ALL).title("Due Today"));
        frame.render_widget(list, chunks[1]);

        // GitHub-style heatmap calendar
        self.heatmap_calendar.render(frame, chunks[2]);

        // category breakdown
        let category_text = if self.stats.category_breakdown.is_empty() {
            "No reviews today".to_string()
        } else {
            self.stats.category_breakdown.join("\n")
        };
        let categories = Paragraph::new(category_text)
            .block(Block::default().borders(Borders::ALL).title("Categories"));
        frame.render_widget(categories, chunks[3]);

        // stats summary
        let stats_text = format!(
            "Streak: {} days | Reviews today: {} | 7-day success rate: {:.1}%\nPress 'l' to browse library | Press 'd' to remove | Press 'e' to edit | Press 'f' to flag | Press 's' to change sort order",
            self.stats.streak_days,
            self.stats.total_reviews_today,
            self.stats.success_rate_7d * 100.0
        );
        let stats =
            Paragraph::new(stats_text).block(Block::default().borders(Borders::ALL).title("Stats"));
        frame.render_widget(stats, chunks[4]);
    }

    pub fn handle_input(&mut self, code: KeyCode) -> DashboardAction {
        match code {
            KeyCode::Down | KeyCode::Char('j') => {
                if self.selected_index < self.katas_due.len().saturating_sub(1) {
                    self.selected_index += 1;
                }
                DashboardAction::None
            }
            KeyCode::Up | KeyCode::Char('k') => {
                self.selected_index = self.selected_index.saturating_sub(1);
                DashboardAction::None
            }
            KeyCode::Enter => {
                if let Some(kata) = self.katas_due.get(self.selected_index) {
                    DashboardAction::SelectKata(kata.clone())
                } else {
                    DashboardAction::None
                }
            }
            KeyCode::Char('d') => {
                if let Some(kata) = self.katas_due.get(self.selected_index) {
                    DashboardAction::RemoveKata(kata.clone())
                } else {
                    DashboardAction::None
                }
            }
            KeyCode::Char('e') => {
                if let Some(kata) = self.katas_due.get(self.selected_index) {
                    DashboardAction::EditKata(kata.clone())
                } else {
                    DashboardAction::None
                }
            }
            KeyCode::Char('f') => {
                if let Some(kata) = self.katas_due.get(self.selected_index) {
                    DashboardAction::ToggleFlagKata(kata.clone())
                } else {
                    DashboardAction::None
                }
            }
            KeyCode::Char('s') => {
                self.cycle_sort_mode();
                DashboardAction::None
            }
            _ => DashboardAction::None,
        }
    }
}

pub enum DashboardAction {
    None,
    SelectKata(Kata),
    RemoveKata(Kata),
    EditKata(Kata),
    ToggleFlagKata(Kata),
}
