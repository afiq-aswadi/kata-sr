//! Kata details screen for viewing detailed information about a kata.
//!
//! This module provides a detailed view of a kata from the library,
//! showing its description, difficulty, category, and dependencies.

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

/// Details screen state for viewing a kata's full information.
pub struct DetailsScreen {
    /// The kata being displayed
    pub kata: AvailableKata,
    /// Whether this kata is already in the deck
    pub already_in_deck: bool,
}

impl DetailsScreen {
    /// Creates a new details screen for the given kata.
    ///
    /// # Arguments
    ///
    /// * `kata` - The kata to display details for
    /// * `already_in_deck` - Whether this kata is already added to the deck
    pub fn new(kata: AvailableKata, already_in_deck: bool) -> Self {
        Self {
            kata,
            already_in_deck,
        }
    }

    /// Renders the details screen.
    ///
    /// # Layout
    ///
    /// - Title with kata name
    /// - Metadata section (category, difficulty)
    /// - Description section
    /// - Dependencies section
    /// - Footer with keybindings
    pub fn render(&self, frame: &mut Frame) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3), // Title
                Constraint::Length(5), // Metadata
                Constraint::Min(10),   // Description
                Constraint::Length(5), // Dependencies
                Constraint::Length(3), // Footer
            ])
            .split(frame.size());

        self.render_title(frame, chunks[0]);
        self.render_metadata(frame, chunks[1]);
        self.render_description(frame, chunks[2]);
        self.render_dependencies(frame, chunks[3]);
        self.render_footer(frame, chunks[4]);
    }

    fn render_title(&self, frame: &mut Frame, area: Rect) {
        let title = Paragraph::new(format!("Kata Details: {}", self.kata.name))
            .style(Style::default().add_modifier(Modifier::BOLD))
            .block(Block::default().borders(Borders::ALL));

        frame.render_widget(title, area);
    }

    fn render_metadata(&self, frame: &mut Frame, area: Rect) {
        let difficulty = difficulty_stars(self.kata.base_difficulty);
        let status = if self.already_in_deck {
            "In Deck"
        } else {
            "Not Added"
        };
        let status_style = if self.already_in_deck {
            Style::default().fg(Color::Green)
        } else {
            Style::default().fg(Color::Gray)
        };

        let lines = vec![
            Line::from(vec![
                Span::styled("Category: ", Style::default().add_modifier(Modifier::BOLD)),
                Span::raw(&self.kata.category),
            ]),
            Line::from(vec![
                Span::styled("Difficulty: ", Style::default().add_modifier(Modifier::BOLD)),
                Span::raw(&difficulty),
            ]),
            Line::from(vec![
                Span::styled("Status: ", Style::default().add_modifier(Modifier::BOLD)),
                Span::styled(status, status_style),
            ]),
        ];

        let metadata = Paragraph::new(lines).block(Block::default().borders(Borders::ALL).title("Info"));

        frame.render_widget(metadata, area);
    }

    fn render_description(&self, frame: &mut Frame, area: Rect) {
        let description = Paragraph::new(self.kata.description.as_str())
            .wrap(Wrap { trim: true })
            .block(Block::default().borders(Borders::ALL).title("Description"));

        frame.render_widget(description, area);
    }

    fn render_dependencies(&self, frame: &mut Frame, area: Rect) {
        let deps_text = if self.kata.dependencies.is_empty() {
            "No dependencies".to_string()
        } else {
            self.kata
                .dependencies
                .iter()
                .map(|d| format!("• {}", d))
                .collect::<Vec<_>>()
                .join("\n")
        };

        let dependencies = Paragraph::new(deps_text)
            .block(Block::default().borders(Borders::ALL).title("Prerequisites"));

        frame.render_widget(dependencies, area);
    }

    fn render_footer(&self, frame: &mut Frame, area: Rect) {
        let footer_text = if self.already_in_deck {
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
    /// # Arguments
    ///
    /// * `code` - The key code pressed by the user
    ///
    /// # Keybindings
    ///
    /// - `a`: Add kata to deck (if not already added)
    /// - `Esc`: Return to library
    pub fn handle_input(&mut self, code: KeyCode) -> DetailsAction {
        match code {
            KeyCode::Char('a') => {
                if self.already_in_deck {
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
