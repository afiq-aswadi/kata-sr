//! Library screen for browsing and adding katas to the deck.
//!
//! This module provides a columnar table view of available katas from
//! the exercises directory, showing their status, title, category, and difficulty.

use std::collections::HashSet;

use anyhow::Result;
use crossterm::event::KeyCode;
use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Row, Table},
    Frame,
};

use crate::core::kata_loader::{load_available_katas, AvailableKata};
use crate::db::repo::KataRepository;

/// Action returned by library input handling.
#[derive(Debug)]
pub enum LibraryAction {
    /// No action taken
    None,
    /// Add a kata to the deck
    AddKata(String),
    /// View detailed information about a kata
    ViewDetails(AvailableKata),
    /// Return to dashboard
    Back,
    /// Open create kata screen
    CreateKata,
}

/// Library screen state for browsing available katas.
pub struct Library {
    /// All katas available in the exercises directory
    pub available_katas: Vec<AvailableKata>,
    /// Names of katas already added to the deck
    pub kata_ids_in_deck: HashSet<String>,
    /// Currently selected index
    pub selected_index: usize,
}

impl Library {
    /// Loads the library by scanning the exercises directory and checking the database.
    ///
    /// # Arguments
    ///
    /// * `repo` - Database repository to check which katas are already added
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use kata_sr::db::repo::KataRepository;
    /// # use kata_sr::tui::library::Library;
    /// let repo = KataRepository::new("kata.db")?;
    /// let library = Library::load(&repo)?;
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn load(repo: &KataRepository) -> Result<Self> {
        let available_katas = load_available_katas()?;
        let existing_katas = repo.get_all_katas()?;

        let kata_ids_in_deck = existing_katas.into_iter().map(|k| k.name).collect();

        Ok(Self {
            available_katas,
            kata_ids_in_deck,
            selected_index: 0,
        })
    }

    /// Renders the library screen with a columnar table layout.
    ///
    /// # Layout
    ///
    /// - Header row with column titles
    /// - Rows for each available kata
    /// - Footer with keybindings
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
        self.render_table(frame, chunks[1]);
        self.render_footer(frame, chunks[2]);
    }

    fn render_header(&self, frame: &mut Frame, area: Rect) {
        let header = Paragraph::new(format!(
            "Kata Library - {} available, {} in deck",
            self.available_katas.len(),
            self.kata_ids_in_deck.len()
        ))
        .block(Block::default().borders(Borders::ALL));

        frame.render_widget(header, area);
    }

    fn render_table(&self, frame: &mut Frame, area: Rect) {
        let header = Row::new(vec!["Status", "Title", "Tags", "Difficulty"])
            .style(Style::default().add_modifier(Modifier::BOLD))
            .bottom_margin(1);

        let rows: Vec<Row> = self
            .available_katas
            .iter()
            .enumerate()
            .map(|(i, kata)| {
                let status = if self.kata_ids_in_deck.contains(&kata.name) {
                    "[Added]"
                } else {
                    "[ ]"
                };

                let difficulty_stars = difficulty_stars(kata.base_difficulty);

                let mut row = Row::new(vec![
                    status.to_string(),
                    kata.name.clone(),
                    kata.category.clone(),
                    difficulty_stars,
                ]);

                if i == self.selected_index {
                    row = row.style(Style::default().fg(Color::Yellow));
                }

                row
            })
            .collect();

        let table = Table::new(
            rows,
            [
                Constraint::Length(7),
                Constraint::Length(25),
                Constraint::Length(15),
                Constraint::Length(18),
            ],
        )
        .header(header)
        .block(Block::default().borders(Borders::ALL).title("Katas"));

        frame.render_widget(table, area);
    }

    fn render_footer(&self, frame: &mut Frame, area: Rect) {
        let footer_text = if self.available_katas.is_empty() {
            Line::from(vec![
                Span::raw("No katas found. "),
                Span::raw("[n] Create New  "),
                Span::raw("[Esc] Back"),
            ])
        } else {
            let selected_kata = &self.available_katas[self.selected_index];
            let can_add = !self.kata_ids_in_deck.contains(&selected_kata.name);

            if can_add {
                Line::from(vec![
                    Span::raw("[j/k] Navigate  "),
                    Span::raw("[a] Add to Deck  "),
                    Span::raw("[n] Create New  "),
                    Span::raw("[Enter] Details  "),
                    Span::raw("[Esc] Back"),
                ])
            } else {
                Line::from(vec![
                    Span::raw("[j/k] Navigate  "),
                    Span::styled("Already in deck  ", Style::default().fg(Color::Gray)),
                    Span::raw("[n] Create New  "),
                    Span::raw("[Esc] Back"),
                ])
            }
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
    /// - `j` or `Down`: Move selection down
    /// - `k` or `Up`: Move selection up
    /// - `a`: Add selected kata to deck (if not already added)
    /// - `n`: Create a new kata
    /// - `Enter`: View details of selected kata
    /// - `Esc`: Return to dashboard
    pub fn handle_input(&mut self, code: KeyCode) -> LibraryAction {
        // Global keybindings that work regardless of available katas
        match code {
            KeyCode::Char('n') => return LibraryAction::CreateKata,
            KeyCode::Esc => return LibraryAction::Back,
            _ => {}
        }

        if self.available_katas.is_empty() {
            return LibraryAction::None;
        }

        match code {
            KeyCode::Char('j') | KeyCode::Down => {
                if self.selected_index < self.available_katas.len() - 1 {
                    self.selected_index += 1;
                }
                LibraryAction::None
            }
            KeyCode::Char('k') | KeyCode::Up => {
                if self.selected_index > 0 {
                    self.selected_index -= 1;
                }
                LibraryAction::None
            }
            KeyCode::Char('a') => {
                let selected_kata = &self.available_katas[self.selected_index];
                if self.kata_ids_in_deck.contains(&selected_kata.name) {
                    LibraryAction::None
                } else {
                    LibraryAction::AddKata(selected_kata.name.clone())
                }
            }
            KeyCode::Enter => {
                let selected_kata = self.available_katas[self.selected_index].clone();
                LibraryAction::ViewDetails(selected_kata)
            }
            _ => LibraryAction::None,
        }
    }

    /// Updates the internal state after a kata is added to the deck.
    ///
    /// # Arguments
    ///
    /// * `kata_name` - Name of the kata that was added
    pub fn mark_as_added(&mut self, kata_name: &str) {
        self.kata_ids_in_deck.insert(kata_name.to_string());
    }
}

/// Converts base difficulty (1-5) to a star representation.
///
/// # Arguments
///
/// * `diff` - Base difficulty from 1 to 5
///
/// # Returns
///
/// String with filled stars, empty stars, and numeric difficulty
///
/// # Examples
///
/// ```
/// # use kata_sr::tui::library::difficulty_stars;
/// assert_eq!(difficulty_stars(3), " (3/5)");
/// assert_eq!(difficulty_stars(5), " (5/5)");
/// ```
pub fn difficulty_stars(diff: i32) -> String {
    let diff = diff.clamp(1, 5);
    let filled = "".repeat(diff as usize);
    let empty = "".repeat((5 - diff) as usize);
    format!("{}{} ({}/5)", filled, empty, diff)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_difficulty_stars() {
        assert_eq!(difficulty_stars(1), " (1/5)");
        assert_eq!(difficulty_stars(2), " (2/5)");
        assert_eq!(difficulty_stars(3), " (3/5)");
        assert_eq!(difficulty_stars(4), " (4/5)");
        assert_eq!(difficulty_stars(5), " (5/5)");
    }

    #[test]
    fn test_difficulty_stars_clamping() {
        assert_eq!(difficulty_stars(0), " (1/5)");
        assert_eq!(difficulty_stars(6), " (5/5)");
        assert_eq!(difficulty_stars(-1), " (1/5)");
    }

    #[test]
    fn test_library_handle_input_navigation() {
        let mut library = Library {
            available_katas: vec![
                AvailableKata {
                    name: "kata1".to_string(),
                    category: "test".to_string(),
                    base_difficulty: 1,
                    description: "Test".to_string(),
                    dependencies: vec![],
                },
                AvailableKata {
                    name: "kata2".to_string(),
                    category: "test".to_string(),
                    base_difficulty: 2,
                    description: "Test".to_string(),
                    dependencies: vec![],
                },
            ],
            kata_ids_in_deck: HashSet::new(),
            selected_index: 0,
        };

        library.handle_input(KeyCode::Char('j'));
        assert_eq!(library.selected_index, 1);

        library.handle_input(KeyCode::Char('k'));
        assert_eq!(library.selected_index, 0);
    }

    #[test]
    fn test_library_handle_input_add_kata() {
        let mut library = Library {
            available_katas: vec![AvailableKata {
                name: "test_kata".to_string(),
                category: "test".to_string(),
                base_difficulty: 3,
                description: "Test".to_string(),
                dependencies: vec![],
            }],
            kata_ids_in_deck: HashSet::new(),
            selected_index: 0,
        };

        match library.handle_input(KeyCode::Char('a')) {
            LibraryAction::AddKata(name) => assert_eq!(name, "test_kata"),
            _ => panic!("Expected AddKata action"),
        }
    }

    #[test]
    fn test_library_handle_input_back() {
        let mut library = Library {
            available_katas: vec![],
            kata_ids_in_deck: HashSet::new(),
            selected_index: 0,
        };

        match library.handle_input(KeyCode::Esc) {
            LibraryAction::Back => {}
            _ => panic!("Expected Back action"),
        }
    }

    #[test]
    fn test_library_mark_as_added() {
        let mut library = Library {
            available_katas: vec![],
            kata_ids_in_deck: HashSet::new(),
            selected_index: 0,
        };

        library.mark_as_added("test_kata");
        assert!(library.kata_ids_in_deck.contains("test_kata"));
    }
}
