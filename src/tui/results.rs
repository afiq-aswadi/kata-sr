use crate::db::repo::Kata;
use crate::runner::python_runner::TestResults;
use crossterm::event::KeyCode;
use ratatui::{
    layout::{Constraint, Direction, Layout},
    style::{Color, Style},
    widgets::{Block, Borders, List, ListItem, Paragraph},
    Frame,
};

pub struct ResultsScreen {
    _kata: Kata,
    results: TestResults,
    selected_rating: usize,
}

impl ResultsScreen {
    pub fn new(kata: Kata, results: TestResults) -> Self {
        Self {
            _kata: kata,
            results,
            selected_rating: 2,
        }
    }

    pub fn render(&self, frame: &mut Frame) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3),
                Constraint::Min(10),
                Constraint::Length(7),
            ])
            .split(frame.size());

        // header
        let status = if self.results.passed {
            "All tests passed!"
        } else {
            "Some tests failed"
        };
        let total = self.results.num_passed + self.results.num_failed;
        let header = Paragraph::new(format!(
            "{} ({}/{} passed in {}ms)",
            status, self.results.num_passed, total, self.results.duration_ms
        ))
        .block(Block::default().borders(Borders::ALL));
        frame.render_widget(header, chunks[0]);

        // test results list
        let items: Vec<ListItem> = self
            .results
            .results
            .iter()
            .map(|r| {
                let (symbol, color) = match r.status.as_str() {
                    "passed" => ("✓", Color::Green),
                    "failed" => ("✗", Color::Red),
                    _ => ("○", Color::Yellow),
                };
                let text = format!("{} {}", symbol, r.test_name);
                ListItem::new(text).style(Style::default().fg(color))
            })
            .collect();

        let list =
            List::new(items).block(Block::default().borders(Borders::ALL).title("Test Results"));
        frame.render_widget(list, chunks[1]);

        // rating selection
        let ratings = ["[0] Again", "[1] Hard", "[2] Good", "[3] Easy"];
        let rating_text = ratings
            .iter()
            .enumerate()
            .map(|(i, r)| {
                if i == self.selected_rating {
                    format!("> {}", r)
                } else {
                    format!("  {}", r)
                }
            })
            .collect::<Vec<_>>()
            .join("\n");

        let rating = Paragraph::new(rating_text).block(
            Block::default()
                .borders(Borders::ALL)
                .title("Rate Difficulty"),
        );
        frame.render_widget(rating, chunks[2]);
    }

    pub fn handle_input(&mut self, code: KeyCode) -> ResultsAction {
        match code {
            KeyCode::Char('0') => {
                self.selected_rating = 0;
                ResultsAction::None
            }
            KeyCode::Char('1') => {
                self.selected_rating = 1;
                ResultsAction::None
            }
            KeyCode::Char('2') => {
                self.selected_rating = 2;
                ResultsAction::None
            }
            KeyCode::Char('3') => {
                self.selected_rating = 3;
                ResultsAction::None
            }
            KeyCode::Enter => ResultsAction::SubmitRating(self.selected_rating as u8),
            _ => ResultsAction::None,
        }
    }
}

pub enum ResultsAction {
    None,
    SubmitRating(u8),
}
