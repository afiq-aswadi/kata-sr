//! Kata details screen for viewing detailed information about a kata.
//!
//! This module provides a detailed view of a kata including its description,
//! dependencies, difficulty, and current status.

use crossterm::event::KeyCode;
use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Wrap},
    Frame,
};

use crate::core::kata_loader::AvailableKata;
use crate::tui::library::difficulty_stars;

/// Action returned by details screen input handling.
#[derive(Debug)]
pub enum DetailsAction {
    /// No action taken
    None,
    /// Add this kata to the deck
    AddKata(String),
    /// Return to library
    Back,
}

/// Details screen state for viewing kata information.
pub struct DetailsScreen {
    /// The kata being displayed
    pub kata: AvailableKata,
    /// Whether this kata is already in the deck
    pub in_deck: bool,
}

impl DetailsScreen {
    /// Creates a new details screen for the given kata.
    pub fn new(kata: AvailableKata, in_deck: bool) -> Self {
        Self { kata, in_deck }
    }

    /// Renders the details screen.
    pub fn render(&self, frame: &mut Frame) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3),
                Constraint::Min(10),
                Constraint::Length(3),
            ])
            .split(frame.size());

        self.render_header(frame, chunks[0]);
        self.render_details(frame, chunks[1]);
        self.render_footer(frame, chunks[2]);
    }

    fn render_header(&self, frame: &mut Frame, area: Rect) {
        let header = Paragraph::new(format!("Kata Details: {}", self.kata.name))
            .block(Block::default().borders(Borders::ALL));

        frame.render_widget(header, area);
    }

    fn render_details(&self, frame: &mut Frame, area: Rect) {
        let mut lines = vec![
            Line::from(vec![
                Span::styled("Name: ", Style::default().add_modifier(Modifier::BOLD)),
                Span::raw(&self.kata.name),
            ]),
            Line::from(""),
            Line::from(vec![
                Span::styled("Category: ", Style::default().add_modifier(Modifier::BOLD)),
                Span::raw(&self.kata.category),
            ]),
            Line::from(""),
            Line::from(vec![
                Span::styled(
                    "Difficulty: ",
                    Style::default().add_modifier(Modifier::BOLD),
                ),
                Span::raw(difficulty_stars(self.kata.base_difficulty)),
            ]),
            Line::from(""),
            Line::from(vec![Span::styled(
                "Description:",
                Style::default().add_modifier(Modifier::BOLD),
            )]),
            Line::from(self.kata.description.clone()),
        ];

        if !self.kata.dependencies.is_empty() {
            lines.push(Line::from(""));
            lines.push(Line::from(vec![Span::styled(
                "Dependencies:",
                Style::default().add_modifier(Modifier::BOLD),
            )]));
            for dep in &self.kata.dependencies {
                lines.push(Line::from(format!("  - {}", dep)));
            }
        }

        if self.in_deck {
            lines.push(Line::from(""));
            lines.push(Line::from(vec![Span::styled(
                "Status: Already in deck",
                Style::default().fg(Color::Green),
            )]));
        }

        let details = Paragraph::new(lines)
            .block(Block::default().borders(Borders::ALL).title("Information"))
            .wrap(Wrap { trim: true });

        frame.render_widget(details, area);
    }

    fn render_footer(&self, frame: &mut Frame, area: Rect) {
        let footer_text = if self.in_deck {
            Line::from(vec![Span::raw("[Esc] Back to Library")])
        } else {
            Line::from(vec![
                Span::raw("[a] Add to Deck  "),
                Span::raw("[Esc] Back to Library"),
            ])
        };

        let footer = Paragraph::new(footer_text).block(Block::default().borders(Borders::ALL));

        frame.render_widget(footer, area);
    }

    /// Handles keyboard input and returns the appropriate action.
    ///
    /// # Keybindings
    ///
    /// - `a`: Add kata to deck (if not already added)
    /// - `Esc`: Return to library
    pub fn handle_input(&self, code: KeyCode) -> DetailsAction {
        match code {
            KeyCode::Char('a') => {
                if self.in_deck {
                    DetailsAction::None
                } else {
                    DetailsAction::AddKata(self.kata.name.clone())
                }
            }
            KeyCode::Esc => DetailsAction::Back,
            _ => DetailsAction::None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_details_handle_input_add() {
        let kata = AvailableKata {
            name: "test_kata".to_string(),
            category: "test".to_string(),
            tags: vec![],
            base_difficulty: 3,
            description: "Test kata".to_string(),
            dependencies: vec![],
        };

        let details = DetailsScreen::new(kata, false);

        match details.handle_input(KeyCode::Char('a')) {
            DetailsAction::AddKata(name) => assert_eq!(name, "test_kata"),
            _ => panic!("Expected AddKata action"),
        }
    }

    #[test]
    fn test_details_handle_input_add_when_in_deck() {
        let kata = AvailableKata {
            name: "test_kata".to_string(),
            category: "test".to_string(),
            tags: vec![],
            base_difficulty: 3,
            description: "Test kata".to_string(),
            dependencies: vec![],
        };

        let details = DetailsScreen::new(kata, true);

        match details.handle_input(KeyCode::Char('a')) {
            DetailsAction::None => {}
            _ => panic!("Expected None action when already in deck"),
        }
    }

    #[test]
    fn test_details_handle_input_back() {
        let kata = AvailableKata {
            name: "test_kata".to_string(),
            category: "test".to_string(),
            tags: vec![],
            base_difficulty: 3,
            description: "Test kata".to_string(),
            dependencies: vec![],
        };

        let details = DetailsScreen::new(kata, false);

        match details.handle_input(KeyCode::Esc) {
            DetailsAction::Back => {}
            _ => panic!("Expected Back action"),
        }
    }
}
