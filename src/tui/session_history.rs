//! Session history screen for viewing past practice sessions.
//!
//! This module provides a table view showing all past sessions for a kata,
//! including dates, pass/fail counts, duration, rating, and ease factor.

use anyhow::Result;
use crossterm::event::KeyCode;
use ratatui::{
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    widgets::{Block, Borders, Cell, Paragraph, Row, Table},
    Frame,
};

use crate::db::repo::{Kata, KataRepository, Session};

/// Action returned by session history input handling.
#[derive(Debug)]
pub enum SessionHistoryAction {
    /// No action taken
    None,
    /// View detailed information about a session
    ViewDetails(i64), // session_id
    /// Delete a specific session
    Delete(i64), // session_id
    /// Return to previous screen
    Back,
}

/// Session history screen state.
pub struct SessionHistoryScreen {
    /// The kata whose sessions are being viewed
    pub kata: Kata,
    /// All sessions for this kata, ordered by most recent first
    pub sessions: Vec<Session>,
    /// Currently selected session index
    pub selected: usize,
    /// Scroll offset (first visible row)
    pub scroll_offset: usize,
    /// Whether we're in delete confirmation mode
    pub confirm_delete: bool,
}

impl SessionHistoryScreen {
    /// Creates a new session history screen for a kata.
    ///
    /// # Arguments
    ///
    /// * `kata` - The kata to show session history for
    /// * `repo` - Database repository to fetch sessions
    pub fn new(kata: Kata, repo: &KataRepository) -> Result<Self> {
        let sessions = repo.get_all_sessions_for_kata(kata.id)?;

        Ok(Self {
            kata,
            sessions,
            selected: 0,
            scroll_offset: 0,
            confirm_delete: false,
        })
    }

    /// Handles keyboard input.
    pub fn handle_input(&mut self, code: KeyCode) -> SessionHistoryAction {
        if self.confirm_delete {
            // In delete confirmation mode
            match code {
                KeyCode::Char('y') | KeyCode::Char('Y') => {
                    if let Some(session) = self.sessions.get(self.selected) {
                        let session_id = session.id;
                        self.confirm_delete = false;
                        return SessionHistoryAction::Delete(session_id);
                    }
                    self.confirm_delete = false;
                    SessionHistoryAction::None
                }
                KeyCode::Char('n') | KeyCode::Char('N') | KeyCode::Esc => {
                    self.confirm_delete = false;
                    SessionHistoryAction::None
                }
                _ => SessionHistoryAction::None,
            }
        } else {
            // Normal navigation mode
            match code {
                KeyCode::Up | KeyCode::Char('k') => {
                    if self.selected > 0 {
                        self.selected -= 1;
                        self.adjust_scroll_offset();
                    }
                    SessionHistoryAction::None
                }
                KeyCode::Down | KeyCode::Char('j') => {
                    if self.selected + 1 < self.sessions.len() {
                        self.selected += 1;
                        self.adjust_scroll_offset();
                    }
                    SessionHistoryAction::None
                }
                KeyCode::Enter => {
                    if let Some(session) = self.sessions.get(self.selected) {
                        SessionHistoryAction::ViewDetails(session.id)
                    } else {
                        SessionHistoryAction::None
                    }
                }
                KeyCode::Char('d') => {
                    if !self.sessions.is_empty() {
                        self.confirm_delete = true;
                    }
                    SessionHistoryAction::None
                }
                KeyCode::Char('q') | KeyCode::Esc => SessionHistoryAction::Back,
                _ => SessionHistoryAction::None,
            }
        }
    }

    /// Adjusts scroll offset to keep selected item visible.
    fn adjust_scroll_offset(&mut self) {
        // Ensure we always show at least 10 rows (or all rows if fewer)
        let visible_rows = 10;

        if self.selected < self.scroll_offset {
            self.scroll_offset = self.selected;
        } else if self.selected >= self.scroll_offset + visible_rows {
            self.scroll_offset = self.selected.saturating_sub(visible_rows - 1);
        }
    }

    /// Renders the session history screen.
    pub fn render(&self, frame: &mut Frame) {
        let area = frame.size();

        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3), // Title
                Constraint::Min(0),    // Table
                Constraint::Length(3), // Instructions
            ])
            .split(area);

        // Title
        let title = Paragraph::new(format!(
            "Session History: {} ({} sessions)",
            self.kata.name,
            self.sessions.len()
        ))
        .style(Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))
        .alignment(Alignment::Center)
        .block(Block::default().borders(Borders::ALL));
        frame.render_widget(title, chunks[0]);

        // Table
        if self.sessions.is_empty() {
            let empty_msg = Paragraph::new("No practice sessions recorded yet for this kata.")
                .style(Style::default().fg(Color::Gray))
                .alignment(Alignment::Center)
                .block(Block::default().borders(Borders::ALL).title("Sessions"));
            frame.render_widget(empty_msg, chunks[1]);
        } else {
            self.render_table(frame, chunks[1]);
        }

        // Instructions
        let instructions = if self.confirm_delete {
            Paragraph::new("Delete this session? [Y]es / [N]o")
                .style(Style::default().fg(Color::Red).add_modifier(Modifier::BOLD))
                .alignment(Alignment::Center)
                .block(Block::default().borders(Borders::ALL))
        } else {
            Paragraph::new("↑/↓: Navigate  Enter: View Details  d: Delete  q/Esc: Back")
                .style(Style::default().fg(Color::Gray))
                .alignment(Alignment::Center)
                .block(Block::default().borders(Borders::ALL))
        };
        frame.render_widget(instructions, chunks[2]);
    }

    fn render_table(&self, frame: &mut Frame, area: Rect) {
        let header = Row::new(vec![
            Cell::from("Date").style(Style::default().add_modifier(Modifier::BOLD)),
            Cell::from("Pass").style(Style::default().add_modifier(Modifier::BOLD)),
            Cell::from("Fail").style(Style::default().add_modifier(Modifier::BOLD)),
            Cell::from("Duration").style(Style::default().add_modifier(Modifier::BOLD)),
            Cell::from("Rating").style(Style::default().add_modifier(Modifier::BOLD)),
            Cell::from("Ease").style(Style::default().add_modifier(Modifier::BOLD)),
        ])
        .height(1);

        // Calculate how many rows we can show
        let available_height = area.height.saturating_sub(3); // subtract borders and header
        let visible_rows = available_height as usize;

        let visible_sessions: Vec<&Session> = self
            .sessions
            .iter()
            .skip(self.scroll_offset)
            .take(visible_rows)
            .collect();

        let rows: Vec<Row> = visible_sessions
            .iter()
            .enumerate()
            .map(|(display_idx, session)| {
                let actual_idx = self.scroll_offset + display_idx;
                let is_selected = actual_idx == self.selected;

                let date = if let Some(completed_at) = session.completed_at {
                    completed_at.format("%Y-%m-%d %H:%M").to_string()
                } else {
                    session.started_at.format("%Y-%m-%d %H:%M").to_string()
                };

                let passed = session.num_passed.map_or("-".to_string(), |n| n.to_string());
                let failed = session.num_failed.map_or("-".to_string(), |n| n.to_string());

                let duration = session.duration_ms.map_or("-".to_string(), |ms| {
                    if ms < 1000 {
                        format!("{}ms", ms)
                    } else {
                        format!("{:.1}s", ms as f64 / 1000.0)
                    }
                });

                let rating_str = session.quality_rating.map_or("-".to_string(), |r| {
                    match r {
                        1 => "Again".to_string(),
                        2 => "Hard".to_string(),
                        3 => "Good".to_string(),
                        4 => "Easy".to_string(),
                        _ => format!("{}", r),
                    }
                });

                // Note: ease factor would need to be tracked per session
                // For now, we'll show a placeholder
                let ease = "-".to_string();

                let style = if is_selected {
                    Style::default()
                        .bg(Color::DarkGray)
                        .fg(Color::White)
                        .add_modifier(Modifier::BOLD)
                } else {
                    Style::default()
                };

                Row::new(vec![
                    Cell::from(date),
                    Cell::from(passed),
                    Cell::from(failed),
                    Cell::from(duration),
                    Cell::from(rating_str),
                    Cell::from(ease),
                ])
                .style(style)
            })
            .collect();

        let widths = [
            Constraint::Length(18), // Date
            Constraint::Length(6),  // Pass
            Constraint::Length(6),  // Fail
            Constraint::Length(10), // Duration
            Constraint::Length(8),  // Rating
            Constraint::Length(6),  // Ease
        ];

        let table = Table::new(rows, widths)
            .header(header)
            .block(Block::default().borders(Borders::ALL).title("Sessions"));

        frame.render_widget(table, area);
    }
}
