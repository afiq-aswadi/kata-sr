use crate::db::repo::{Kata, KataRepository};
use chrono::Utc;
use crossterm::event::KeyCode;
use ratatui::{
    layout::{Constraint, Direction, Layout},
    widgets::{Block, Borders, List, ListItem, Paragraph},
    Frame,
};

pub struct Dashboard {
    pub katas_due: Vec<Kata>,
    pub locked_katas: Vec<(Kata, String)>,
    pub selected_index: usize,
    pub stats: DashboardStats,
}

pub struct DashboardStats {
    pub streak_days: i32,
    pub total_reviews_today: i32,
    pub success_rate_7d: f64,
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

        let stats = DashboardStats {
            streak_days: repo.get_current_streak()?,
            total_reviews_today: repo.get_reviews_count_today()?,
            success_rate_7d: repo.get_success_rate_last_n_days(7)?,
        };

        Ok(Self {
            katas_due,
            locked_katas,
            selected_index: 0,
            stats,
        })
    }

    pub fn render(&self, frame: &mut Frame) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3),
                Constraint::Min(10),
                Constraint::Length(5),
            ])
            .split(frame.size());

        let header = Paragraph::new(format!(
            "Kata Spaced Repetition - {} katas due today",
            self.katas_due.len()
        ))
        .block(Block::default().borders(Borders::ALL));
        frame.render_widget(header, chunks[0]);

        let items: Vec<ListItem> = self
            .katas_due
            .iter()
            .enumerate()
            .map(|(i, kata)| {
                let marker = if i == self.selected_index { ">" } else { " " };
                let text = format!(
                    "{} {} (difficulty: {:.1})",
                    marker, kata.name, kata.current_difficulty
                );
                ListItem::new(text)
            })
            .collect();

        let list =
            List::new(items).block(Block::default().borders(Borders::ALL).title("Due Today"));
        frame.render_widget(list, chunks[1]);

        let stats_text = format!(
            "Streak: {} days | Reviews today: {} | 7-day success rate: {:.1}%\nPress 'l' to browse library",
            self.stats.streak_days,
            self.stats.total_reviews_today,
            self.stats.success_rate_7d * 100.0
        );
        let stats =
            Paragraph::new(stats_text).block(Block::default().borders(Borders::ALL).title("Stats"));
        frame.render_widget(stats, chunks[2]);
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
            _ => DashboardAction::None,
        }
    }
}

pub enum DashboardAction {
    None,
    SelectKata(Kata),
}
