//! Library screen for browsing and adding katas to the deck.
//!
//! This module provides a columnar table view of available katas from
//! the exercises directory, showing their status, title, category, and difficulty.

use std::collections::HashSet;

use anyhow::Result;
use crossterm::event::KeyCode;
use fuzzy_matcher::skim::SkimMatcherV2;
use fuzzy_matcher::FuzzyMatcher;
use ratatui::{
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, Paragraph, Row, Table},
    Frame,
};

use crate::core::kata_loader::{get_unique_categories, load_available_katas, AvailableKata};
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

/// Sort mode for library view
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SortMode {
    Name,
    Difficulty,
    DateAdded,
    Category,
}

impl SortMode {
    fn next(&self) -> Self {
        match self {
            Self::Name => Self::Difficulty,
            Self::Difficulty => Self::Category,
            Self::Category => Self::DateAdded,
            Self::DateAdded => Self::Name,
        }
    }

    fn as_str(&self) -> &str {
        match self {
            Self::Name => "Name",
            Self::Difficulty => "Difficulty",
            Self::Category => "Category",
            Self::DateAdded => "Date Added",
        }
    }
}

/// Library screen state for browsing available katas.
pub struct Library {
    /// All katas available in the exercises directory
    pub all_katas: Vec<AvailableKata>,
    /// Filtered list after applying search and category filters
    pub filtered_katas: Vec<AvailableKata>,
    /// Names of katas already added to the deck
    pub kata_ids_in_deck: HashSet<String>,
    /// Currently selected index in filtered list
    pub selected_index: usize,

    // Search
    pub search_mode: bool,
    pub search_query: String,

    // Category filtering
    pub category_filter_mode: bool,
    pub available_categories: Vec<String>,
    pub selected_categories: Vec<String>,
    pub category_selected_index: usize,

    // Sorting
    pub sort_mode: SortMode,
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
        let all_katas = load_available_katas()?;
        let existing_katas = repo.get_all_katas()?;

        let kata_ids_in_deck = existing_katas.into_iter().map(|k| k.name).collect();
        let available_categories = get_unique_categories(&all_katas);

        let mut library = Self {
            all_katas: all_katas.clone(),
            filtered_katas: all_katas,
            kata_ids_in_deck,
            selected_index: 0,
            search_mode: false,
            search_query: String::new(),
            category_filter_mode: false,
            available_categories,
            selected_categories: Vec::new(),
            category_selected_index: 0,
            sort_mode: SortMode::Name,
        };

        library.apply_filters();
        Ok(library)
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

        if self.category_filter_mode {
            self.render_category_selector(frame, chunks[1]);
        } else {
            self.render_table(frame, chunks[1]);
        }

        self.render_footer(frame, chunks[2]);
    }

    fn render_header(&self, frame: &mut Frame, area: Rect) {
        let mut title_parts = vec![Span::raw("Library ")];

        // Show active filters
        if !self.search_query.is_empty() {
            title_parts.push(Span::styled(
                format!("| Search: {} ", self.search_query),
                Style::default().fg(Color::Cyan),
            ));
        }

        if !self.selected_categories.is_empty() {
            title_parts.push(Span::styled(
                format!("| Tags: {} ", self.selected_categories.join(", ")),
                Style::default().fg(Color::Yellow),
            ));
        }

        title_parts.push(Span::styled(
            format!("| Sort: {}", self.sort_mode.as_str()),
            Style::default().fg(Color::Magenta),
        ));

        let title = Line::from(title_parts);

        let info = Line::from(format!(
            "Showing {} of {} katas ({} in deck)",
            self.filtered_katas.len(),
            self.all_katas.len(),
            self.kata_ids_in_deck.len()
        ));

        let header = Paragraph::new(vec![title, info])
            .block(Block::default().borders(Borders::ALL));

        frame.render_widget(header, area);
    }

    fn render_table(&self, frame: &mut Frame, area: Rect) {
        let header = Row::new(vec!["Status", "Title", "Tags", "Difficulty"])
            .style(Style::default().add_modifier(Modifier::BOLD))
            .bottom_margin(1);

        let rows: Vec<Row> = self
            .filtered_katas
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

    fn render_category_selector(&self, frame: &mut Frame, area: Rect) {
        let items: Vec<ListItem> = self
            .available_categories
            .iter()
            .enumerate()
            .map(|(i, category)| {
                let is_selected = self.selected_categories.contains(category);
                let checkbox = if is_selected { "[x]" } else { "[ ]" };
                let mut style = if is_selected {
                    Style::default().fg(Color::Cyan)
                } else {
                    Style::default()
                };

                if i == self.category_selected_index {
                    style = style.add_modifier(Modifier::BOLD).fg(Color::Yellow);
                }

                ListItem::new(format!("{} {}", checkbox, category)).style(style)
            })
            .collect();

        let list = List::new(items).block(
            Block::default()
                .borders(Borders::ALL)
                .title("Select Categories (j/k to navigate, Space to toggle, Enter/Esc to close)"),
        );

        frame.render_widget(list, area);
    }

    fn render_footer(&self, frame: &mut Frame, area: Rect) {
        let footer_text = if self.search_mode {
            Line::from("Type to search | Enter/Esc: Done")
        } else if self.category_filter_mode {
            Line::from("[j/k] Navigate | [Space] Toggle | [Enter/Esc] Done")
        } else if self.filtered_katas.is_empty() {
            Line::from(vec![
                Span::raw("No katas found. "),
                Span::raw("[/] Search  "),
                Span::raw("[t] Filter  "),
                Span::raw("[c] Clear filters  "),
                Span::raw("[n] Create New  "),
                Span::raw("[Esc] Back"),
            ])
        } else {
            let selected_kata = &self.filtered_katas[self.selected_index];
            let can_add = !self.kata_ids_in_deck.contains(&selected_kata.name);

            if can_add {
                Line::from(vec![
                    Span::raw("[j/k] Navigate  "),
                    Span::raw("[a] Add  "),
                    Span::raw("[/] Search  "),
                    Span::raw("[t] Filter  "),
                    Span::raw("[s] Sort  "),
                    Span::raw("[c] Clear  "),
                    Span::raw("[n] New  "),
                    Span::raw("[Enter] Details  "),
                    Span::raw("[Esc] Back"),
                ])
            } else {
                Line::from(vec![
                    Span::raw("[j/k] Navigate  "),
                    Span::styled("Already in deck  ", Style::default().fg(Color::Gray)),
                    Span::raw("[/] Search  "),
                    Span::raw("[t] Filter  "),
                    Span::raw("[s] Sort  "),
                    Span::raw("[c] Clear  "),
                    Span::raw("[n] New  "),
                    Span::raw("[Esc] Back"),
                ])
            }
        };

        let footer = Paragraph::new(footer_text)
            .block(Block::default().borders(Borders::ALL))
            .alignment(Alignment::Center);

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
    /// - `/`: Activate search mode
    /// - `t`: Open category filter selector
    /// - `s`: Cycle sort modes
    /// - `c`: Clear all filters
    /// - `n`: Create a new kata
    /// - `Enter`: View details of selected kata
    /// - `Esc`: Return to dashboard or exit current mode
    pub fn handle_input(&mut self, code: KeyCode) -> LibraryAction {
        // Handle search mode
        if self.search_mode {
            return self.handle_search_input(code);
        }

        // Handle category filter mode
        if self.category_filter_mode {
            return self.handle_category_filter_input(code);
        }

        // Global keybindings that work regardless of available katas
        match code {
            KeyCode::Char('n') => return LibraryAction::CreateKata,
            KeyCode::Char('/') => {
                self.search_mode = true;
                return LibraryAction::None;
            }
            KeyCode::Char('t') => {
                self.category_filter_mode = true;
                return LibraryAction::None;
            }
            KeyCode::Char('s') => {
                self.sort_mode = self.sort_mode.next();
                self.apply_filters();
                return LibraryAction::None;
            }
            KeyCode::Char('c') => {
                self.search_query.clear();
                self.selected_categories.clear();
                self.apply_filters();
                return LibraryAction::None;
            }
            KeyCode::Esc => return LibraryAction::Back,
            _ => {}
        }

        if self.filtered_katas.is_empty() {
            return LibraryAction::None;
        }

        match code {
            KeyCode::Char('j') | KeyCode::Down => {
                if self.selected_index < self.filtered_katas.len() - 1 {
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
                let selected_kata = &self.filtered_katas[self.selected_index];
                if self.kata_ids_in_deck.contains(&selected_kata.name) {
                    LibraryAction::None
                } else {
                    LibraryAction::AddKata(selected_kata.name.clone())
                }
            }
            KeyCode::Enter => {
                let selected_kata = self.filtered_katas[self.selected_index].clone();
                LibraryAction::ViewDetails(selected_kata)
            }
            _ => LibraryAction::None,
        }
    }

    fn handle_search_input(&mut self, code: KeyCode) -> LibraryAction {
        match code {
            KeyCode::Char(c) => {
                self.search_query.push(c);
                self.apply_filters();
                LibraryAction::None
            }
            KeyCode::Backspace => {
                self.search_query.pop();
                self.apply_filters();
                LibraryAction::None
            }
            KeyCode::Enter | KeyCode::Esc => {
                self.search_mode = false;
                LibraryAction::None
            }
            _ => LibraryAction::None,
        }
    }

    fn handle_category_filter_input(&mut self, code: KeyCode) -> LibraryAction {
        match code {
            KeyCode::Char('j') | KeyCode::Down => {
                if self.category_selected_index < self.available_categories.len() - 1 {
                    self.category_selected_index += 1;
                }
                LibraryAction::None
            }
            KeyCode::Char('k') | KeyCode::Up => {
                if self.category_selected_index > 0 {
                    self.category_selected_index -= 1;
                }
                LibraryAction::None
            }
            KeyCode::Char(' ') => {
                let category = self.available_categories[self.category_selected_index].clone();
                if let Some(pos) = self.selected_categories.iter().position(|c| c == &category) {
                    self.selected_categories.remove(pos);
                } else {
                    self.selected_categories.push(category);
                }
                self.apply_filters();
                LibraryAction::None
            }
            KeyCode::Enter | KeyCode::Esc => {
                self.category_filter_mode = false;
                LibraryAction::None
            }
            _ => LibraryAction::None,
        }
    }

    fn apply_filters(&mut self) {
        // Start with all katas
        let mut filtered = self.all_katas.clone();

        // Apply search filter with fuzzy matching
        if !self.search_query.is_empty() {
            let matcher = SkimMatcherV2::default();
            let query = &self.search_query;

            // Filter and score
            let mut scored: Vec<(AvailableKata, i64)> = filtered
                .into_iter()
                .filter_map(|kata| {
                    let name_score = matcher.fuzzy_match(&kata.name, query).unwrap_or(0);
                    let desc_score = matcher
                        .fuzzy_match(&kata.description, query)
                        .unwrap_or(0);
                    let max_score = name_score.max(desc_score);

                    if max_score > 0 {
                        Some((kata, max_score))
                    } else {
                        None
                    }
                })
                .collect();

            // Sort by score (highest first)
            scored.sort_by(|a, b| b.1.cmp(&a.1));

            filtered = scored.into_iter().map(|(kata, _)| kata).collect();
        }

        // Apply category filter
        if !self.selected_categories.is_empty() {
            filtered.retain(|kata| self.selected_categories.contains(&kata.category));
        }

        // Apply sorting
        match self.sort_mode {
            SortMode::Name => {
                filtered.sort_by(|a, b| a.name.cmp(&b.name));
            }
            SortMode::Difficulty => {
                filtered.sort_by(|a, b| b.base_difficulty.cmp(&a.base_difficulty));
            }
            SortMode::Category => {
                filtered.sort_by(|a, b| a.category.cmp(&b.category));
            }
            SortMode::DateAdded => {
                // For available katas (not in DB), we can't sort by date added
                // So we'll just keep the original order
            }
        }

        self.filtered_katas = filtered;

        // Reset selection if out of bounds
        if self.selected_index >= self.filtered_katas.len() {
            self.selected_index = self.filtered_katas.len().saturating_sub(1);
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
        let katas = vec![
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
        ];

        let mut library = Library {
            all_katas: katas.clone(),
            filtered_katas: katas,
            kata_ids_in_deck: HashSet::new(),
            selected_index: 0,
            search_mode: false,
            search_query: String::new(),
            category_filter_mode: false,
            available_categories: vec!["test".to_string()],
            selected_categories: Vec::new(),
            category_selected_index: 0,
            sort_mode: SortMode::Name,
        };

        library.handle_input(KeyCode::Char('j'));
        assert_eq!(library.selected_index, 1);

        library.handle_input(KeyCode::Char('k'));
        assert_eq!(library.selected_index, 0);
    }

    #[test]
    fn test_library_handle_input_add_kata() {
        let katas = vec![AvailableKata {
            name: "test_kata".to_string(),
            category: "test".to_string(),
            base_difficulty: 3,
            description: "Test".to_string(),
            dependencies: vec![],
        }];

        let mut library = Library {
            all_katas: katas.clone(),
            filtered_katas: katas,
            kata_ids_in_deck: HashSet::new(),
            selected_index: 0,
            search_mode: false,
            search_query: String::new(),
            category_filter_mode: false,
            available_categories: vec!["test".to_string()],
            selected_categories: Vec::new(),
            category_selected_index: 0,
            sort_mode: SortMode::Name,
        };

        match library.handle_input(KeyCode::Char('a')) {
            LibraryAction::AddKata(name) => assert_eq!(name, "test_kata"),
            _ => panic!("Expected AddKata action"),
        }
    }

    #[test]
    fn test_library_handle_input_back() {
        let mut library = Library {
            all_katas: vec![],
            filtered_katas: vec![],
            kata_ids_in_deck: HashSet::new(),
            selected_index: 0,
            search_mode: false,
            search_query: String::new(),
            category_filter_mode: false,
            available_categories: vec![],
            selected_categories: Vec::new(),
            category_selected_index: 0,
            sort_mode: SortMode::Name,
        };

        match library.handle_input(KeyCode::Esc) {
            LibraryAction::Back => {}
            _ => panic!("Expected Back action"),
        }
    }

    #[test]
    fn test_library_mark_as_added() {
        let mut library = Library {
            all_katas: vec![],
            filtered_katas: vec![],
            kata_ids_in_deck: HashSet::new(),
            selected_index: 0,
            search_mode: false,
            search_query: String::new(),
            category_filter_mode: false,
            available_categories: vec![],
            selected_categories: Vec::new(),
            category_selected_index: 0,
            sort_mode: SortMode::Name,
        };

        library.mark_as_added("test_kata");
        assert!(library.kata_ids_in_deck.contains("test_kata"));
    }
}
