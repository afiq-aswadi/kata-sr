//! Session detail screen for viewing full test results from a past session.
//!
//! This module provides a detailed view of a single practice session,
//! including all test results and their output.

use anyhow::Result;
use crossterm::event::KeyCode;
use ratatui::{
    layout::{Alignment, Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, Paragraph, Wrap},
    Frame,
};
use serde::Deserialize;

use crate::db::repo::{KataRepository, Session};

/// Action returned by session detail input handling.
#[derive(Debug)]
pub enum SessionDetailAction {
    /// No action taken
    None,
    /// Return to session history
    Back,
}

/// Test result from pytest JSON output
#[derive(Debug, Clone, Deserialize)]
struct TestResult {
    test_name: String,
    status: String,
    output: String,
}

/// Session detail screen state.
pub struct SessionDetailScreen {
    /// The session being viewed
    pub session: Session,
    /// Kata name for display
    pub kata_name: String,
    /// Parsed test results (if available)
    pub test_results: Option<Vec<TestResult>>,
    /// Scroll position for test output
    pub scroll_offset: usize,
}

impl SessionDetailScreen {
    /// Creates a new session detail screen.
    ///
    /// # Arguments
    ///
    /// * `session_id` - The session ID to display
    /// * `repo` - Database repository to fetch session data
    pub fn new(session_id: i64, repo: &KataRepository) -> Result<Self> {
        let session = repo
            .get_session_by_id(session_id)?
            .ok_or_else(|| anyhow::anyhow!("Session not found: {}", session_id))?;

        let kata = repo
            .get_kata_by_id(session.kata_id)?
            .ok_or_else(|| anyhow::anyhow!("Kata not found for session: {}", session.kata_id))?;

        // Try to parse test results JSON
        let test_results = if let Some(ref json_str) = session.test_results_json {
            serde_json::from_str::<Vec<TestResult>>(json_str).ok()
        } else {
            None
        };

        Ok(Self {
            session,
            kata_name: kata.name,
            test_results,
            scroll_offset: 0,
        })
    }

    /// Handles keyboard input.
    pub fn handle_input(&mut self, code: KeyCode) -> SessionDetailAction {
        match code {
            KeyCode::Up | KeyCode::Char('k') => {
                self.scroll_offset = self.scroll_offset.saturating_sub(1);
                SessionDetailAction::None
            }
            KeyCode::Down | KeyCode::Char('j') => {
                self.scroll_offset = self.scroll_offset.saturating_add(1);
                SessionDetailAction::None
            }
            KeyCode::Char('q') | KeyCode::Esc => SessionDetailAction::Back,
            _ => SessionDetailAction::None,
        }
    }

    /// Renders the session detail screen.
    pub fn render(&self, frame: &mut Frame) {
        let area = frame.size();

        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3), // Title
                Constraint::Length(6), // Summary
                Constraint::Min(0),    // Test results
                Constraint::Length(2), // Instructions
            ])
            .split(area);

        // Title
        let title = Paragraph::new(format!("Session Details: {}", self.kata_name))
            .style(Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))
            .alignment(Alignment::Center)
            .block(Block::default().borders(Borders::ALL));
        frame.render_widget(title, chunks[0]);

        // Summary
        self.render_summary(frame, chunks[1]);

        // Test results
        self.render_test_results(frame, chunks[2]);

        // Instructions
        let instructions = Paragraph::new("↑/↓: Scroll  q/Esc: Back")
            .style(Style::default().fg(Color::Gray))
            .alignment(Alignment::Center);
        frame.render_widget(instructions, chunks[3]);
    }

    fn render_summary(&self, frame: &mut Frame, area: ratatui::layout::Rect) {
        let date = if let Some(completed_at) = self.session.completed_at {
            completed_at.format("%Y-%m-%d %H:%M:%S").to_string()
        } else {
            self.session.started_at.format("%Y-%m-%d %H:%M:%S").to_string()
        };

        let duration = self.session.duration_ms.map_or("-".to_string(), |ms| {
            if ms < 1000 {
                format!("{}ms", ms)
            } else {
                format!("{:.2}s", ms as f64 / 1000.0)
            }
        });

        let rating = self.session.quality_rating.map_or("-".to_string(), |r| {
            match r {
                1 => "Again (1)".to_string(),
                2 => "Hard (2)".to_string(),
                3 => "Good (3)".to_string(),
                4 => "Easy (4)".to_string(),
                _ => format!("{}", r),
            }
        });

        let passed = self.session.num_passed.unwrap_or(0);
        let failed = self.session.num_failed.unwrap_or(0);
        let total = passed + failed;

        let lines = vec![
            Line::from(vec![
                Span::styled("Date: ", Style::default().add_modifier(Modifier::BOLD)),
                Span::raw(date),
                Span::raw("    "),
                Span::styled("Duration: ", Style::default().add_modifier(Modifier::BOLD)),
                Span::raw(duration),
            ]),
            Line::from(vec![
                Span::styled("Rating: ", Style::default().add_modifier(Modifier::BOLD)),
                Span::raw(rating),
            ]),
            Line::from(""),
            Line::from(vec![
                Span::styled("Tests: ", Style::default().add_modifier(Modifier::BOLD)),
                Span::styled(
                    format!("{} passed", passed),
                    Style::default().fg(Color::Green),
                ),
                Span::raw(", "),
                Span::styled(
                    format!("{} failed", failed),
                    Style::default().fg(Color::Red),
                ),
                Span::raw(format!(" (total: {})", total)),
            ]),
        ];

        let summary = Paragraph::new(lines)
            .block(Block::default().borders(Borders::ALL).title("Summary"));
        frame.render_widget(summary, area);
    }

    fn render_test_results(&self, frame: &mut Frame, area: ratatui::layout::Rect) {
        if let Some(ref results) = self.test_results {
            let items: Vec<ListItem> = results
                .iter()
                .flat_map(|test| {
                    let status_symbol = match test.status.as_str() {
                        "passed" => "✓",
                        "failed" => "✗",
                        "skipped" => "⊝",
                        _ => "?",
                    };

                    let status_color = match test.status.as_str() {
                        "passed" => Color::Green,
                        "failed" => Color::Red,
                        "skipped" => Color::Yellow,
                        _ => Color::Gray,
                    };

                    let mut lines = vec![ListItem::new(Line::from(vec![
                        Span::styled(
                            format!("{} ", status_symbol),
                            Style::default().fg(status_color),
                        ),
                        Span::styled(
                            test.test_name.clone(),
                            Style::default().add_modifier(Modifier::BOLD),
                        ),
                    ]))];

                    // Add output if present and test failed
                    if !test.output.is_empty() && test.status == "failed" {
                        // Indent the output
                        let output_lines: Vec<ListItem> = test
                            .output
                            .lines()
                            .take(5) // Limit output lines
                            .map(|line| {
                                ListItem::new(format!("  {}", line))
                                    .style(Style::default().fg(Color::Gray))
                            })
                            .collect();
                        lines.extend(output_lines);
                    }

                    lines
                })
                .skip(self.scroll_offset)
                .collect();

            let list = List::new(items)
                .block(Block::default().borders(Borders::ALL).title("Test Results"));
            frame.render_widget(list, area);
        } else {
            let msg = Paragraph::new("No detailed test results available for this session.")
                .style(Style::default().fg(Color::Gray))
                .alignment(Alignment::Center)
                .block(Block::default().borders(Borders::ALL).title("Test Results"))
                .wrap(Wrap { trim: true });
            frame.render_widget(msg, area);
        }
    }
}
