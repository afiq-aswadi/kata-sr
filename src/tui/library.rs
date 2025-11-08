//! Library screen for browsing and adding katas to the deck.
//!
//! This module provides a tabbed table view with two tabs:
//! - My Deck: Shows katas in the user's deck with due dates and last reviewed
//! - All Katas: Shows all available katas from exercises directory

use std::collections::HashSet;

use anyhow::Result;
use chrono::Utc;
use crossterm::event::KeyCode;
use ratatui::{
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Cell, Paragraph, Row, Table},
    Frame,
};

use crate::core::kata_loader::{load_available_katas, AvailableKata};
use crate::db::repo::{Kata, KataRepository};

/// Which tab is currently active in the library view
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LibraryTab {
    /// Shows katas in the user's deck
    MyDeck,
    /// Shows all available katas from exercises directory
    AllKatas,
}

/// Action returned by library input handling.
#[derive(Debug)]
pub enum LibraryAction {
    /// No action taken
    None,
    /// Add a kata to the deck
    AddKata(String),
    /// Remove a kata from the deck
    RemoveKata(Kata),
    /// View detailed information about a kata
    ViewDetails(AvailableKata),
    /// Return to dashboard
    Back,
    /// Open create kata screen
    CreateKata,
}

/// Library screen state for browsing available katas.
pub struct Library {
    /// Current active tab
    pub active_tab: LibraryTab,

    /// Katas in the user's deck (for My Deck tab)
    pub deck_katas: Vec<Kata>,
    /// Selected index in My Deck tab
    pub deck_selected: usize,

    /// All available katas from exercises (for All Katas tab)
    pub available_katas: Vec<AvailableKata>,
    /// Selected index in All Katas tab
    pub all_selected: usize,

    /// Names of katas already added to the deck (for marking in All Katas tab)
    pub kata_ids_in_deck: HashSet<String>,
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
        let deck_katas = repo.get_all_katas()?;

        let kata_ids_in_deck = deck_katas.iter().map(|k| k.name.clone()).collect();

        Ok(Self {
            active_tab: LibraryTab::MyDeck,
            deck_katas,
            deck_selected: 0,
            available_katas,
            all_selected: 0,
            kata_ids_in_deck,
        })
    }

    /// Renders the library screen with tabs and tabbed content.
    ///
    /// # Layout
    ///
    /// - Tab bar showing both tabs with active tab highlighted
    /// - Stats bar showing tab-specific statistics
    /// - Content area showing table for active tab
    /// - Footer with tab-specific keybindings
    pub fn render(&self, frame: &mut Frame) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3),   // Tab bar
                Constraint::Length(3),   // Stats
                Constraint::Min(10),     // Content
                Constraint::Length(3),   // Footer
            ])
            .split(frame.size());

        self.render_tabs(frame, chunks[0]);
        self.render_stats(frame, chunks[1]);

        match self.active_tab {
            LibraryTab::MyDeck => self.render_my_deck(frame, chunks[2]),
            LibraryTab::AllKatas => self.render_all_katas(frame, chunks[2]),
        }

        self.render_footer(frame, chunks[3]);
    }

    fn render_tabs(&self, frame: &mut Frame, area: Rect) {
        let tab_chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Percentage(50),
                Constraint::Percentage(50),
            ])
            .split(area);

        // My Deck tab
        let my_deck_style = if self.active_tab == LibraryTab::MyDeck {
            Style::default()
                .fg(Color::Black)
                .bg(Color::Cyan)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(Color::White)
        };

        let my_deck_tab = Paragraph::new(format!(" My Deck ({}) ", self.deck_katas.len()))
            .style(my_deck_style)
            .alignment(Alignment::Center)
            .block(Block::default().borders(Borders::ALL));
        frame.render_widget(my_deck_tab, tab_chunks[0]);

        // All Katas tab
        let all_style = if self.active_tab == LibraryTab::AllKatas {
            Style::default()
                .fg(Color::Black)
                .bg(Color::Cyan)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(Color::White)
        };

        let all_tab = Paragraph::new(format!(" All Katas ({}) ", self.available_katas.len()))
            .style(all_style)
            .alignment(Alignment::Center)
            .block(Block::default().borders(Borders::ALL));
        frame.render_widget(all_tab, tab_chunks[1]);
    }

    fn render_stats(&self, frame: &mut Frame, area: Rect) {
        let stats_text = match self.active_tab {
            LibraryTab::MyDeck => {
                let due_count = self.deck_katas.iter()
                    .filter(|k| k.next_review_at.map_or(true, |t| t <= Utc::now()))
                    .count();
                format!("Due today: {} | Total in deck: {}", due_count, self.deck_katas.len())
            }
            LibraryTab::AllKatas => {
                let in_deck_count = self.available_katas.iter()
                    .filter(|k| self.kata_ids_in_deck.contains(&k.name))
                    .count();
                format!("Total katas: {} | In your deck: {}", self.available_katas.len(), in_deck_count)
            }
        };

        let paragraph = Paragraph::new(stats_text)
            .alignment(Alignment::Center)
            .block(Block::default().borders(Borders::ALL));

        frame.render_widget(paragraph, area);
    }

    fn render_my_deck(&self, frame: &mut Frame, area: Rect) {
        if self.deck_katas.is_empty() {
            let empty_msg = Paragraph::new("No katas in your deck. Press Tab to browse All Katas.")
                .alignment(Alignment::Center)
                .block(Block::default().borders(Borders::ALL).title(" My Deck "));
            frame.render_widget(empty_msg, area);
            return;
        }

        let header = Row::new(vec!["", "Name", "Tags", "Due", "Difficulty"])
            .style(Style::default().add_modifier(Modifier::BOLD))
            .bottom_margin(1);

        let rows: Vec<Row> = self.deck_katas
            .iter()
            .enumerate()
            .map(|(i, kata)| {
                let is_selected = i == self.deck_selected;
                let prefix = if is_selected { ">" } else { " " };

                let due_str = format_due_date(kata.next_review_at);
                let is_due = kata.next_review_at.map_or(true, |t| t <= Utc::now());
                let due_style = if is_due {
                    Style::default().fg(Color::Red)
                } else {
                    Style::default().fg(Color::Green)
                };

                let row_style = if is_selected {
                    Style::default().bg(Color::DarkGray)
                } else {
                    Style::default()
                };

                Row::new(vec![
                    Cell::from(prefix),
                    Cell::from(kata.name.clone()),
                    Cell::from(kata.category.clone()),
                    Cell::from(due_str).style(due_style),
                    Cell::from(format!("{:.1}", kata.current_difficulty)),
                ])
                .style(row_style)
            })
            .collect();

        let table = Table::new(rows, [
            Constraint::Length(2),
            Constraint::Percentage(40),
            Constraint::Percentage(25),
            Constraint::Percentage(20),
            Constraint::Percentage(15),
        ])
        .header(header)
        .block(Block::default().borders(Borders::ALL).title(" My Deck "));

        frame.render_widget(table, area);
    }

    fn render_all_katas(&self, frame: &mut Frame, area: Rect) {
        if self.available_katas.is_empty() {
            let empty_msg = Paragraph::new("No katas found in exercises directory. Press 'n' to create one.")
                .alignment(Alignment::Center)
                .block(Block::default().borders(Borders::ALL).title(" All Katas "));
            frame.render_widget(empty_msg, area);
            return;
        }

        let header = Row::new(vec!["", "Name", "Tags", "Difficulty", "In Deck"])
            .style(Style::default().add_modifier(Modifier::BOLD))
            .bottom_margin(1);

        let rows: Vec<Row> = self.available_katas
            .iter()
            .enumerate()
            .map(|(i, kata)| {
                let is_selected = i == self.all_selected;
                let prefix = if is_selected { ">" } else { " " };

                let in_deck = self.kata_ids_in_deck.contains(&kata.name);
                let in_deck_marker = if in_deck { "✓" } else { " " };
                let in_deck_style = if in_deck {
                    Style::default().fg(Color::Green)
                } else {
                    Style::default()
                };

                let difficulty_str = difficulty_stars(kata.base_difficulty);

                let row_style = if is_selected {
                    Style::default().bg(Color::DarkGray)
                } else {
                    Style::default()
                };

                Row::new(vec![
                    Cell::from(prefix),
                    Cell::from(kata.name.clone()),
                    Cell::from(kata.category.clone()),
                    Cell::from(difficulty_str),
                    Cell::from(in_deck_marker).style(in_deck_style),
                ])
                .style(row_style)
            })
            .collect();

        let table = Table::new(rows, [
            Constraint::Length(2),
            Constraint::Percentage(35),
            Constraint::Percentage(25),
            Constraint::Percentage(25),
            Constraint::Percentage(15),
        ])
        .header(header)
        .block(Block::default().borders(Borders::ALL).title(" All Katas "));

        frame.render_widget(table, area);
    }

    fn render_footer(&self, frame: &mut Frame, area: Rect) {
        let footer_text = match self.active_tab {
            LibraryTab::MyDeck => {
                if self.deck_katas.is_empty() {
                    Line::from(vec![
                        Span::raw("[Tab] Switch tab  "),
                        Span::raw("[n] Create New  "),
                        Span::raw("[Esc] Back"),
                    ])
                } else {
                    Line::from(vec![
                        Span::raw("[Tab] Switch tab  "),
                        Span::raw("[j/k] Navigate  "),
                        Span::raw("[d] Remove from deck  "),
                        Span::raw("[n] Create New  "),
                        Span::raw("[Esc] Back"),
                    ])
                }
            }
            LibraryTab::AllKatas => {
                if self.available_katas.is_empty() {
                    Line::from(vec![
                        Span::raw("[Tab] Switch tab  "),
                        Span::raw("[n] Create New  "),
                        Span::raw("[Esc] Back"),
                    ])
                } else {
                    let selected_kata = &self.available_katas[self.all_selected];
                    let can_add = !self.kata_ids_in_deck.contains(&selected_kata.name);

                    if can_add {
                        Line::from(vec![
                            Span::raw("[Tab] Switch tab  "),
                            Span::raw("[j/k] Navigate  "),
                            Span::raw("[a] Add to deck  "),
                            Span::raw("[Enter] Details  "),
                            Span::raw("[n] Create New  "),
                            Span::raw("[Esc] Back"),
                        ])
                    } else {
                        Line::from(vec![
                            Span::raw("[Tab] Switch tab  "),
                            Span::raw("[j/k] Navigate  "),
                            Span::styled("Already in deck  ", Style::default().fg(Color::Gray)),
                            Span::raw("[Enter] Details  "),
                            Span::raw("[n] Create New  "),
                            Span::raw("[Esc] Back"),
                        ])
                    }
                }
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
    /// - `Tab`: Switch between My Deck and All Katas tabs
    /// - `j` or `Down`: Move selection down
    /// - `k` or `Up`: Move selection up
    /// - `a`: Add selected kata to deck (All Katas tab only)
    /// - `d`: Remove selected kata from deck (My Deck tab only)
    /// - `n`: Create a new kata
    /// - `Enter`: View details of selected kata (All Katas tab only)
    /// - `Esc`: Return to dashboard
    pub fn handle_input(&mut self, code: KeyCode) -> LibraryAction {
        // Global keybindings that work in all tabs
        match code {
            KeyCode::Tab => {
                self.active_tab = match self.active_tab {
                    LibraryTab::MyDeck => LibraryTab::AllKatas,
                    LibraryTab::AllKatas => LibraryTab::MyDeck,
                };
                return LibraryAction::None;
            }
            KeyCode::Char('n') => return LibraryAction::CreateKata,
            KeyCode::Esc => return LibraryAction::Back,
            _ => {}
        }

        // Tab-specific keybindings
        match self.active_tab {
            LibraryTab::MyDeck => self.handle_my_deck_input(code),
            LibraryTab::AllKatas => self.handle_all_katas_input(code),
        }
    }

    fn handle_my_deck_input(&mut self, code: KeyCode) -> LibraryAction {
        if self.deck_katas.is_empty() {
            return LibraryAction::None;
        }

        match code {
            KeyCode::Char('j') | KeyCode::Down => {
                if self.deck_selected < self.deck_katas.len() - 1 {
                    self.deck_selected += 1;
                }
                LibraryAction::None
            }
            KeyCode::Char('k') | KeyCode::Up => {
                if self.deck_selected > 0 {
                    self.deck_selected -= 1;
                }
                LibraryAction::None
            }
            KeyCode::Char('d') => {
                let kata = self.deck_katas[self.deck_selected].clone();
                LibraryAction::RemoveKata(kata)
            }
            _ => LibraryAction::None,
        }
    }

    fn handle_all_katas_input(&mut self, code: KeyCode) -> LibraryAction {
        if self.available_katas.is_empty() {
            return LibraryAction::None;
        }

        match code {
            KeyCode::Char('j') | KeyCode::Down => {
                if self.all_selected < self.available_katas.len() - 1 {
                    self.all_selected += 1;
                }
                LibraryAction::None
            }
            KeyCode::Char('k') | KeyCode::Up => {
                if self.all_selected > 0 {
                    self.all_selected -= 1;
                }
                LibraryAction::None
            }
            KeyCode::Char('a') => {
                let selected_kata = &self.available_katas[self.all_selected];
                if self.kata_ids_in_deck.contains(&selected_kata.name) {
                    LibraryAction::None
                } else {
                    LibraryAction::AddKata(selected_kata.name.clone())
                }
            }
            KeyCode::Enter => {
                let selected_kata = self.available_katas[self.all_selected].clone();
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

    /// Updates the internal state after a kata is removed from the deck.
    ///
    /// # Arguments
    ///
    /// * `kata_name` - Name of the kata that was removed
    pub fn mark_as_removed(&mut self, kata_name: &str) {
        self.kata_ids_in_deck.remove(kata_name);
    }

    /// Refreshes the deck katas list after changes.
    ///
    /// # Arguments
    ///
    /// * `repo` - Database repository
    pub fn refresh_deck(&mut self, repo: &KataRepository) -> Result<()> {
        self.deck_katas = repo.get_all_katas()?;

        // Adjust selected index if it's out of bounds
        if self.deck_selected >= self.deck_katas.len() && self.deck_katas.len() > 0 {
            self.deck_selected = self.deck_katas.len() - 1;
        }

        Ok(())
    }
}

/// Formats a due date for display.
///
/// # Arguments
///
/// * `due` - Optional due date
///
/// # Returns
///
/// String representation: "Now", "Today", "Tomorrow", "Nd", or "Never"
fn format_due_date(due: Option<chrono::DateTime<Utc>>) -> String {
    match due {
        None => "Now".to_string(),
        Some(dt) => {
            let now = Utc::now();
            if dt <= now {
                "Now".to_string()
            } else {
                let days = (dt - now).num_days();
                if days == 0 {
                    "Today".to_string()
                } else if days == 1 {
                    "Tomorrow".to_string()
                } else {
                    format!("{}d", days)
                }
            }
        }
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
/// assert_eq!(difficulty_stars(3), "★★★☆☆ (3/5)");
/// assert_eq!(difficulty_stars(5), "★★★★★ (5/5)");
/// ```
pub fn difficulty_stars(diff: i32) -> String {
    let diff = diff.clamp(1, 5);
    let filled = "★".repeat(diff as usize);
    let empty = "☆".repeat((5 - diff) as usize);
    format!("{}{} ({}/5)", filled, empty, diff)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_difficulty_stars() {
        assert_eq!(difficulty_stars(1), "★☆☆☆☆ (1/5)");
        assert_eq!(difficulty_stars(2), "★★☆☆☆ (2/5)");
        assert_eq!(difficulty_stars(3), "★★★☆☆ (3/5)");
        assert_eq!(difficulty_stars(4), "★★★★☆ (4/5)");
        assert_eq!(difficulty_stars(5), "★★★★★ (5/5)");
    }

    #[test]
    fn test_difficulty_stars_clamping() {
        assert_eq!(difficulty_stars(0), "★☆☆☆☆ (1/5)");
        assert_eq!(difficulty_stars(6), "★★★★★ (5/5)");
        assert_eq!(difficulty_stars(-1), "★☆☆☆☆ (1/5)");
    }

    #[test]
    fn test_library_tab_switching() {
        let mut library = Library {
            active_tab: LibraryTab::MyDeck,
            deck_katas: vec![],
            deck_selected: 0,
            available_katas: vec![],
            all_selected: 0,
            kata_ids_in_deck: HashSet::new(),
        };

        library.handle_input(KeyCode::Tab);
        assert_eq!(library.active_tab, LibraryTab::AllKatas);

        library.handle_input(KeyCode::Tab);
        assert_eq!(library.active_tab, LibraryTab::MyDeck);
    }

    #[test]
    fn test_library_handle_input_navigation_my_deck() {
        use crate::db::repo::Kata;
        use chrono::Utc;

        let mut library = Library {
            active_tab: LibraryTab::MyDeck,
            deck_katas: vec![
                Kata {
                    id: 1,
                    name: "kata1".to_string(),
                    category: "test".to_string(),
                    description: "Test".to_string(),
                    base_difficulty: 1,
                    current_difficulty: 1.0,
                    parent_kata_id: None,
                    variation_params: None,
                    next_review_at: None,
                    last_reviewed_at: None,
                    current_ease_factor: 2.5,
                    current_interval_days: 1,
                    current_repetition_count: 0,
                    created_at: Utc::now(),
                },
                Kata {
                    id: 2,
                    name: "kata2".to_string(),
                    category: "test".to_string(),
                    description: "Test".to_string(),
                    base_difficulty: 2,
                    current_difficulty: 2.0,
                    parent_kata_id: None,
                    variation_params: None,
                    next_review_at: None,
                    last_reviewed_at: None,
                    current_ease_factor: 2.5,
                    current_interval_days: 1,
                    current_repetition_count: 0,
                    created_at: Utc::now(),
                },
            ],
            deck_selected: 0,
            available_katas: vec![],
            all_selected: 0,
            kata_ids_in_deck: HashSet::new(),
        };

        library.handle_input(KeyCode::Char('j'));
        assert_eq!(library.deck_selected, 1);

        library.handle_input(KeyCode::Char('k'));
        assert_eq!(library.deck_selected, 0);
    }

    #[test]
    fn test_library_handle_input_navigation_all_katas() {
        let mut library = Library {
            active_tab: LibraryTab::AllKatas,
            deck_katas: vec![],
            deck_selected: 0,
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
            all_selected: 0,
            kata_ids_in_deck: HashSet::new(),
        };

        library.handle_input(KeyCode::Char('j'));
        assert_eq!(library.all_selected, 1);

        library.handle_input(KeyCode::Char('k'));
        assert_eq!(library.all_selected, 0);
    }

    #[test]
    fn test_library_handle_input_add_kata() {
        let mut library = Library {
            active_tab: LibraryTab::AllKatas,
            deck_katas: vec![],
            deck_selected: 0,
            available_katas: vec![AvailableKata {
                name: "test_kata".to_string(),
                category: "test".to_string(),
                base_difficulty: 3,
                description: "Test".to_string(),
                dependencies: vec![],
            }],
            all_selected: 0,
            kata_ids_in_deck: HashSet::new(),
        };

        match library.handle_input(KeyCode::Char('a')) {
            LibraryAction::AddKata(name) => assert_eq!(name, "test_kata"),
            _ => panic!("Expected AddKata action"),
        }
    }

    #[test]
    fn test_library_handle_input_back() {
        let mut library = Library {
            active_tab: LibraryTab::MyDeck,
            deck_katas: vec![],
            deck_selected: 0,
            available_katas: vec![],
            all_selected: 0,
            kata_ids_in_deck: HashSet::new(),
        };

        match library.handle_input(KeyCode::Esc) {
            LibraryAction::Back => {}
            _ => panic!("Expected Back action"),
        }
    }

    #[test]
    fn test_library_mark_as_added() {
        let mut library = Library {
            active_tab: LibraryTab::MyDeck,
            deck_katas: vec![],
            deck_selected: 0,
            available_katas: vec![],
            all_selected: 0,
            kata_ids_in_deck: HashSet::new(),
        };

        library.mark_as_added("test_kata");
        assert!(library.kata_ids_in_deck.contains("test_kata"));
    }

    #[test]
    fn test_library_mark_as_removed() {
        let mut library = Library {
            active_tab: LibraryTab::MyDeck,
            deck_katas: vec![],
            deck_selected: 0,
            available_katas: vec![],
            all_selected: 0,
            kata_ids_in_deck: HashSet::from_iter(vec!["test_kata".to_string()]),
        };

        library.mark_as_removed("test_kata");
        assert!(!library.kata_ids_in_deck.contains("test_kata"));
    }

    #[test]
    fn test_format_due_date_none() {
        assert_eq!(format_due_date(None), "Now");
    }

    #[test]
    fn test_format_due_date_past() {
        use chrono::Duration;
        let past = Utc::now() - Duration::days(1);
        assert_eq!(format_due_date(Some(past)), "Now");
    }

    #[test]
    fn test_format_due_date_future() {
        use chrono::Duration;
        // Add enough hours to ensure we cross into the next day
        let tomorrow = Utc::now() + Duration::hours(25);
        let result = format_due_date(Some(tomorrow));
        assert!(result == "Tomorrow" || result == "1d", "Expected 'Tomorrow' or '1d', got '{}'", result);

        let in_5_days = Utc::now() + Duration::days(5) + Duration::hours(1);
        let result = format_due_date(Some(in_5_days));
        assert!(result == "5d" || result == "4d", "Expected '5d' or '4d', got '{}'", result);
    }
}
