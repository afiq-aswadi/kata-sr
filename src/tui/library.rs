//! Library screen for browsing and adding katas to the deck.
//!
//! This module provides a tabbed table view with two tabs:
//! - My Deck: Shows katas in the user's deck with due dates and last reviewed
//! - All Katas: Shows all available katas from exercises directory with search/filter/sort

use std::collections::{HashMap, HashSet};

use anyhow::Result;
use chrono::Utc;
use crossterm::event::KeyCode;
use fuzzy_matcher::skim::SkimMatcherV2;
use fuzzy_matcher::FuzzyMatcher;
use ratatui::{
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Cell, List, ListItem, Paragraph, Row, Table},
    Frame,
};

use crate::core::kata_loader::{get_unique_categories, load_available_katas, AvailableKata};
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
    /// Edit an existing kata (by kata ID or name to look up)
    EditKataById(i64),
    EditKataByName(String),
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
    /// Current active tab
    pub active_tab: LibraryTab,

    /// Katas in the user's deck (for My Deck tab)
    pub deck_katas: Vec<Kata>,
    /// Tags for each kata in deck (kata_id -> tags)
    pub deck_kata_tags: HashMap<i64, Vec<String>>,
    /// Selected index in My Deck tab
    pub deck_selected: usize,
    /// Scroll offset for My Deck tab (first visible row)
    pub deck_scroll_offset: usize,

    /// All available katas from exercises (unfiltered)
    pub all_available_katas: Vec<AvailableKata>,
    /// Filtered katas after applying search and category filters
    pub filtered_available_katas: Vec<AvailableKata>,
    /// Selected index in All Katas tab (applies to filtered list)
    pub all_selected: usize,
    /// Scroll offset for All Katas tab (first visible row)
    pub all_scroll_offset: usize,

    /// Names of katas already added to the deck
    pub kata_ids_in_deck: HashSet<String>,

    // Search (for All Katas tab)
    pub search_mode: bool,
    pub search_query: String,

    // Category filtering (for All Katas tab)
    pub category_filter_mode: bool,
    pub available_categories: Vec<String>,
    pub selected_categories: Vec<String>,
    pub category_selected_index: usize,

    // Sorting (for All Katas tab)
    pub sort_mode: SortMode,
    /// Sort direction: true = ascending, false = descending
    pub sort_ascending: bool,
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

        // Load tags for all katas in deck
        let mut deck_kata_tags = HashMap::new();
        for kata in &deck_katas {
            if let Ok(tags) = repo.get_kata_tags(kata.id) {
                deck_kata_tags.insert(kata.id, tags);
            }
        }

        let kata_ids_in_deck = deck_katas.iter().map(|k| k.name.clone()).collect();
        let available_categories = get_unique_categories(&available_katas);

        let mut library = Self {
            active_tab: LibraryTab::MyDeck,
            deck_katas,
            deck_kata_tags,
            deck_selected: 0,
            deck_scroll_offset: 0,
            all_available_katas: available_katas.clone(),
            filtered_available_katas: available_katas,
            all_selected: 0,
            all_scroll_offset: 0,
            kata_ids_in_deck,
            search_mode: false,
            search_query: String::new(),
            category_filter_mode: false,
            available_categories,
            selected_categories: Vec::new(),
            category_selected_index: 0,
            sort_mode: SortMode::Name,
            sort_ascending: true,
        };

        library.apply_filters();
        Ok(library)
    }

    /// Updates scroll offset to ensure the selected item is visible.
    ///
    /// # Arguments
    ///
    /// * `selected` - The currently selected index
    /// * `scroll_offset` - The current scroll offset (will be updated)
    /// * `visible_height` - The number of visible rows in the viewport
    /// * `total_items` - Total number of items in the list
    fn update_scroll_offset(
        selected: usize,
        scroll_offset: &mut usize,
        visible_height: usize,
        total_items: usize,
    ) {
        if total_items == 0 || visible_height == 0 {
            return;
        }

        // If selected item is above the visible area, scroll up
        if selected < *scroll_offset {
            *scroll_offset = selected;
        }
        // If selected item is below the visible area, scroll down
        else if selected >= *scroll_offset + visible_height {
            *scroll_offset = selected.saturating_sub(visible_height - 1);
        }

        // Ensure scroll offset doesn't exceed bounds
        let max_offset = total_items.saturating_sub(visible_height);
        if *scroll_offset > max_offset {
            *scroll_offset = max_offset;
        }
    }

    /// Renders the library screen with tabs and tabbed content.
    ///
    /// # Layout
    ///
    /// - Tab bar showing both tabs with active tab highlighted
    /// - Stats bar showing tab-specific statistics
    /// - Content area showing table for active tab
    /// - Footer with tab-specific keybindings
    pub fn render(&mut self, frame: &mut Frame) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3), // Tab bar
                Constraint::Length(3), // Stats
                Constraint::Min(10),   // Content
                Constraint::Length(3), // Footer
            ])
            .split(frame.size());

        self.render_tabs(frame, chunks[0]);
        self.render_stats(frame, chunks[1]);

        match self.active_tab {
            LibraryTab::MyDeck => self.render_my_deck(frame, chunks[2]),
            LibraryTab::AllKatas => {
                if self.category_filter_mode {
                    self.render_category_selector(frame, chunks[2]);
                } else {
                    self.render_all_katas(frame, chunks[2]);
                }
            }
        }

        self.render_footer(frame, chunks[3]);
    }

    fn render_tabs(&self, frame: &mut Frame, area: Rect) {
        let tab_chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
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

        let all_tab = Paragraph::new(format!(" All Katas ({}) ", self.all_available_katas.len()))
            .style(all_style)
            .alignment(Alignment::Center)
            .block(Block::default().borders(Borders::ALL));
        frame.render_widget(all_tab, tab_chunks[1]);
    }

    fn render_stats(&self, frame: &mut Frame, area: Rect) {
        let stats_text = match self.active_tab {
            LibraryTab::MyDeck => {
                let due_count = self
                    .deck_katas
                    .iter()
                    .filter(|k| k.next_review_at.map_or(true, |t| t <= Utc::now()))
                    .count();
                format!(
                    "Due today: {} | Total in deck: {}",
                    due_count,
                    self.deck_katas.len()
                )
            }
            LibraryTab::AllKatas => {
                let mut parts = vec![format!(
                    "Showing {} of {}",
                    self.filtered_available_katas.len(),
                    self.all_available_katas.len()
                )];

                if !self.search_query.is_empty() {
                    parts.push(format!("Search: {}", self.search_query));
                }

                if !self.selected_categories.is_empty() {
                    parts.push(format!("Tags: {}", self.selected_categories.join(", ")));
                }

                let sort_arrow = if self.sort_ascending { "↑" } else { "↓" };
                parts.push(format!("Sort: {} {}", self.sort_mode.as_str(), sort_arrow));

                parts.join(" | ")
            }
        };

        let paragraph = Paragraph::new(stats_text)
            .alignment(Alignment::Center)
            .block(Block::default().borders(Borders::ALL));

        frame.render_widget(paragraph, area);
    }

    fn render_my_deck(&mut self, frame: &mut Frame, area: Rect) {
        if self.deck_katas.is_empty() {
            let empty_msg = Paragraph::new("No katas in your deck. Press Tab to browse All Katas.")
                .alignment(Alignment::Center)
                .block(Block::default().borders(Borders::ALL).title(" My Deck "));
            frame.render_widget(empty_msg, area);
            return;
        }

        // Calculate visible height: area height - top border - bottom border - header - header bottom margin
        let visible_height = area.height.saturating_sub(4) as usize;

        // Update scroll offset to keep selection visible
        Self::update_scroll_offset(
            self.deck_selected,
            &mut self.deck_scroll_offset,
            visible_height,
            self.deck_katas.len(),
        );

        // Calculate scroll indicators
        let has_content_above = self.deck_scroll_offset > 0;
        let has_content_below = (self.deck_scroll_offset + visible_height) < self.deck_katas.len();
        let scroll_indicator = match (has_content_above, has_content_below) {
            (true, true) => " ↑↓",
            (true, false) => " ↑",
            (false, true) => " ↓",
            (false, false) => "",
        };

        // Position indicator: showing item X of Y total
        let position_info = if self.deck_katas.len() > 0 {
            format!(" [{}/{}]{} ", self.deck_selected + 1, self.deck_katas.len(), scroll_indicator)
        } else {
            String::new()
        };

        let header = Row::new(vec!["", "Name", "Tags", "Due", "Difficulty"])
            .style(Style::default().add_modifier(Modifier::BOLD))
            .bottom_margin(1);

        // Only render visible rows
        let end_index = (self.deck_scroll_offset + visible_height).min(self.deck_katas.len());
        let visible_katas = &self.deck_katas[self.deck_scroll_offset..end_index];

        let rows: Vec<Row> = visible_katas
            .iter()
            .enumerate()
            .map(|(offset_i, kata)| {
                let i = self.deck_scroll_offset + offset_i;
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

                // Display tags as comma-separated list, fallback to category if no tags
                let tags_str = if let Some(tags) = self.deck_kata_tags.get(&kata.id) {
                    if !tags.is_empty() {
                        tags.join(", ")
                    } else if !kata.category.is_empty() {
                        kata.category.clone()
                    } else {
                        "—".to_string()
                    }
                } else if !kata.category.is_empty() {
                    kata.category.clone()
                } else {
                    "—".to_string()
                };

                Row::new(vec![
                    Cell::from(prefix),
                    Cell::from(kata.name.clone()),
                    Cell::from(tags_str),
                    Cell::from(due_str).style(due_style),
                    Cell::from(format!("{:.1}", kata.current_difficulty)),
                ])
                .style(row_style)
            })
            .collect();

        let table = Table::new(
            rows,
            [
                Constraint::Length(2),
                Constraint::Percentage(40),
                Constraint::Percentage(25),
                Constraint::Percentage(20),
                Constraint::Percentage(15),
            ],
        )
        .header(header)
        .block(Block::default().borders(Borders::ALL).title(format!(" My Deck {}", position_info)));

        frame.render_widget(table, area);
    }

    fn render_all_katas(&mut self, frame: &mut Frame, area: Rect) {
        if self.filtered_available_katas.is_empty() {
            let msg = if self.all_available_katas.is_empty() {
                "No katas found in exercises directory. Press 'n' to create one."
            } else {
                "No katas match your filters. Press 'c' to clear filters."
            };
            let empty_msg = Paragraph::new(msg)
                .alignment(Alignment::Center)
                .block(Block::default().borders(Borders::ALL).title(" All Katas "));
            frame.render_widget(empty_msg, area);
            return;
        }

        // Calculate visible height: area height - top border - bottom border - header - header bottom margin
        let visible_height = area.height.saturating_sub(4) as usize;

        // Update scroll offset to keep selection visible
        Self::update_scroll_offset(
            self.all_selected,
            &mut self.all_scroll_offset,
            visible_height,
            self.filtered_available_katas.len(),
        );

        // Calculate scroll indicators
        let has_content_above = self.all_scroll_offset > 0;
        let has_content_below = (self.all_scroll_offset + visible_height) < self.filtered_available_katas.len();
        let scroll_indicator = match (has_content_above, has_content_below) {
            (true, true) => " ↑↓",
            (true, false) => " ↑",
            (false, true) => " ↓",
            (false, false) => "",
        };

        // Position indicator: showing item X of Y total
        let position_info = if self.filtered_available_katas.len() > 0 {
            format!(" [{}/{}]{} ", self.all_selected + 1, self.filtered_available_katas.len(), scroll_indicator)
        } else {
            String::new()
        };

        let header = Row::new(vec!["", "Name", "Tags", "Difficulty", "In Deck"])
            .style(Style::default().add_modifier(Modifier::BOLD))
            .bottom_margin(1);

        // Only render visible rows
        let end_index = (self.all_scroll_offset + visible_height).min(self.filtered_available_katas.len());
        let visible_katas = &self.filtered_available_katas[self.all_scroll_offset..end_index];

        let rows: Vec<Row> = visible_katas
            .iter()
            .enumerate()
            .map(|(offset_i, kata)| {
                let i = self.all_scroll_offset + offset_i;
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

        let table = Table::new(
            rows,
            [
                Constraint::Length(2),
                Constraint::Percentage(35),
                Constraint::Percentage(25),
                Constraint::Percentage(25),
                Constraint::Percentage(15),
            ],
        )
        .header(header)
        .block(Block::default().borders(Borders::ALL).title(format!(" All Katas {}", position_info)));

        frame.render_widget(table, area);
    }

    fn render_category_selector(&self, frame: &mut Frame, area: Rect) {
        let items: Vec<ListItem> = self
            .available_categories
            .iter()
            .enumerate()
            .map(|(i, category)| {
                let is_selected = self.selected_categories.contains(category);
                let is_cursor = i == self.category_selected_index;

                // Use better checkbox symbols
                let checkbox = if is_selected { "[✓]" } else { "[ ]" };

                // Create styled line with different colors for checkbox and text
                let checkbox_style = if is_selected {
                    Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)
                } else {
                    Style::default().fg(Color::DarkGray)
                };

                let text_style = if is_cursor {
                    // Cursor: highlight the whole line
                    Style::default()
                        .fg(Color::Black)
                        .bg(Color::Yellow)
                        .add_modifier(Modifier::BOLD)
                } else if is_selected {
                    // Selected but not cursor: use cyan
                    Style::default().fg(Color::Cyan)
                } else {
                    // Not selected: use white
                    Style::default().fg(Color::White)
                };

                // Build the line with styled spans
                let line = if is_cursor {
                    // When cursor is here, highlight entire line
                    Line::from(vec![
                        Span::styled(format!("{} {}", checkbox, category), text_style),
                    ])
                } else {
                    // Otherwise, style checkbox and text separately
                    Line::from(vec![
                        Span::styled(checkbox, checkbox_style),
                        Span::raw(" "),
                        Span::styled(category, text_style),
                    ])
                };

                ListItem::new(line)
            })
            .collect();

        let list = List::new(items).block(
            Block::default()
                .borders(Borders::ALL)
                .title("Filter by Category · [j/k] Navigate · [Space] Toggle · [Enter/Esc] Done"),
        );

        frame.render_widget(list, area);
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
                if self.search_mode {
                    // Show the actual search query with cursor indicator
                    Line::from(vec![
                        Span::styled("⚡ SEARCH MODE · ", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
                        Span::styled("Query: ", Style::default().fg(Color::Cyan)),
                        Span::raw(&self.search_query),
                        Span::styled("|", Style::default().fg(Color::Yellow)),
                        Span::styled("  [Enter/Esc] Done  [Ctrl+U] Clear", Style::default().fg(Color::Gray)),
                    ])
                } else if self.category_filter_mode {
                    Line::from(vec![
                        Span::styled("🏷  FILTER MODE · ", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
                        Span::raw("[j/k] Navigate | [Space] Toggle | [Enter/Esc] Done"),
                    ])
                } else if self.filtered_available_katas.is_empty() {
                    Line::from(vec![
                        Span::raw("[Tab] Switch tab  "),
                        Span::raw("[/] Search  "),
                        Span::raw("[t] Filter  "),
                        Span::raw("[c] Clear filters  "),
                        Span::raw("[n] Create New  "),
                        Span::raw("[Esc] Back"),
                    ])
                } else {
                    let selected_kata = &self.filtered_available_katas[self.all_selected];
                    let can_add = !self.kata_ids_in_deck.contains(&selected_kata.name);

                    if can_add {
                        Line::from(vec![
                            Span::raw("[Tab] Switch  "),
                            Span::raw("[j/k] Navigate  "),
                            Span::raw("[a] Add  "),
                            Span::raw("[/] Search  "),
                            Span::raw("[t] Filter  "),
                            Span::raw("[s] Sort  "),
                            Span::raw("[r] Reverse  "),
                            Span::raw("[c] Clear  "),
                            Span::raw("[Enter] Details  "),
                            Span::raw("[Esc] Back"),
                        ])
                    } else {
                        Line::from(vec![
                            Span::raw("[Tab] Switch  "),
                            Span::raw("[j/k] Navigate  "),
                            Span::styled("Already in deck  ", Style::default().fg(Color::Gray)),
                            Span::raw("[/] Search  "),
                            Span::raw("[t] Filter  "),
                            Span::raw("[s] Sort  "),
                            Span::raw("[r] Reverse  "),
                            Span::raw("[Enter] Details  "),
                            Span::raw("[Esc] Back"),
                        ])
                    }
                }
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
    /// - `Tab`: Switch between My Deck and All Katas tabs
    /// - `j` or `Down`: Move selection down
    /// - `k` or `Up`: Move selection up
    /// - `a`: Add selected kata to deck (All Katas tab only)
    /// - `d`: Remove selected kata from deck (My Deck tab only)
    /// - `/`: Activate search mode (All Katas tab only)
    /// - `t`: Open category filter selector (All Katas tab only)
    /// - `s`: Cycle sort modes (All Katas tab only)
    /// - `c`: Clear all filters (All Katas tab only)
    /// - `n`: Create a new kata
    /// - `Enter`: View details of selected kata (All Katas tab only)
    /// - `Esc`: Return to dashboard or exit current mode
    pub fn handle_input(&mut self, code: KeyCode) -> LibraryAction {
        // Handle search mode (All Katas tab)
        if self.active_tab == LibraryTab::AllKatas && self.search_mode {
            return self.handle_search_input(code);
        }

        // Handle category filter mode (All Katas tab)
        if self.active_tab == LibraryTab::AllKatas && self.category_filter_mode {
            return self.handle_category_filter_input(code);
        }

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
            KeyCode::Char('e') => {
                let kata = self.deck_katas[self.deck_selected].clone();
                LibraryAction::EditKataById(kata.id)
            }
            _ => LibraryAction::None,
        }
    }

    fn handle_all_katas_input(&mut self, code: KeyCode) -> LibraryAction {
        // Handle global All Katas keybindings (work even if list is empty)
        match code {
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
            KeyCode::Char('r') => {
                self.sort_ascending = !self.sort_ascending;
                self.apply_filters();
                return LibraryAction::None;
            }
            KeyCode::Char('c') => {
                self.search_query.clear();
                self.selected_categories.clear();
                self.apply_filters();
                return LibraryAction::None;
            }
            _ => {}
        }

        if self.filtered_available_katas.is_empty() {
            return LibraryAction::None;
        }

        match code {
            KeyCode::Char('j') | KeyCode::Down => {
                if self.all_selected < self.filtered_available_katas.len() - 1 {
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
                let selected_kata = &self.filtered_available_katas[self.all_selected];
                if self.kata_ids_in_deck.contains(&selected_kata.name) {
                    LibraryAction::None
                } else {
                    LibraryAction::AddKata(selected_kata.name.clone())
                }
            }
            KeyCode::Char('e') => {
                let selected_kata = &self.filtered_available_katas[self.all_selected];
                // Only allow editing if kata is in deck (i.e., in database)
                if self.kata_ids_in_deck.contains(&selected_kata.name) {
                    LibraryAction::EditKataByName(selected_kata.name.clone())
                } else {
                    LibraryAction::None
                }
            }
            KeyCode::Enter => {
                let selected_kata = self.filtered_available_katas[self.all_selected].clone();
                LibraryAction::ViewDetails(selected_kata)
            }
            _ => LibraryAction::None,
        }
    }

    fn handle_search_input(&mut self, code: KeyCode) -> LibraryAction {
        match code {
            KeyCode::Char(c) if c == '\u{15}' => {
                // Ctrl+U - clear input buffer (Unix convention)
                self.search_query.clear();
                self.apply_filters();
                LibraryAction::None
            }
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
            KeyCode::Enter => {
                // Keep the search query, just exit search mode
                self.search_mode = false;
                LibraryAction::None
            }
            KeyCode::Esc => {
                // Clear search and exit search mode
                self.search_query.clear();
                self.apply_filters();
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
        let mut filtered = self.all_available_katas.clone();

        // Apply search filter with fuzzy matching
        if !self.search_query.is_empty() {
            let matcher = SkimMatcherV2::default();
            let query = &self.search_query;

            // Filter and score
            let mut scored: Vec<(AvailableKata, i64)> = filtered
                .into_iter()
                .filter_map(|kata| {
                    let name_score = matcher.fuzzy_match(&kata.name, query).unwrap_or(0);
                    let desc_score = matcher.fuzzy_match(&kata.description, query).unwrap_or(0);
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
                filtered.sort_by(|a, b| {
                    if self.sort_ascending {
                        a.name.cmp(&b.name)
                    } else {
                        b.name.cmp(&a.name)
                    }
                });
            }
            SortMode::Difficulty => {
                filtered.sort_by(|a, b| {
                    if self.sort_ascending {
                        a.base_difficulty.cmp(&b.base_difficulty)
                    } else {
                        b.base_difficulty.cmp(&a.base_difficulty)
                    }
                });
            }
            SortMode::Category => {
                filtered.sort_by(|a, b| {
                    if self.sort_ascending {
                        a.category.cmp(&b.category)
                    } else {
                        b.category.cmp(&a.category)
                    }
                });
            }
            SortMode::DateAdded => {
                // For available katas (not in DB), we can't sort by date added
                // So we'll just keep the original order
            }
        }

        self.filtered_available_katas = filtered;

        // Reset selection if out of bounds
        if self.all_selected >= self.filtered_available_katas.len() {
            self.all_selected = self.filtered_available_katas.len().saturating_sub(1);
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

        // Reload tags for all katas in deck
        self.deck_kata_tags.clear();
        for kata in &self.deck_katas {
            if let Ok(tags) = repo.get_kata_tags(kata.id) {
                self.deck_kata_tags.insert(kata.id, tags);
            }
        }

        // Adjust selected index if it's out of bounds
        if self.deck_selected >= self.deck_katas.len() && !self.deck_katas.is_empty() {
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
            deck_kata_tags: HashMap::new(),
            deck_selected: 0,
            deck_scroll_offset: 0,
            all_available_katas: vec![],
            filtered_available_katas: vec![],
            all_selected: 0,
            all_scroll_offset: 0,
            kata_ids_in_deck: HashSet::new(),
            search_mode: false,
            search_query: String::new(),
            category_filter_mode: false,
            available_categories: vec![],
            selected_categories: Vec::new(),
            category_selected_index: 0,
            sort_mode: SortMode::Name,
            sort_ascending: true,
        };

        library.handle_input(KeyCode::Tab);
        assert_eq!(library.active_tab, LibraryTab::AllKatas);

        library.handle_input(KeyCode::Tab);
        assert_eq!(library.active_tab, LibraryTab::MyDeck);
    }

    #[test]
    fn test_library_handle_input_navigation_my_deck() {
        use chrono::Utc;

        let mut library = Library {
            active_tab: LibraryTab::MyDeck,
            deck_katas: vec![
                Kata {
                    id: 1,
                    name: "kata1".to_string(),
                    category: "test".to_string(),
                    description: "Test".to_string(),
                    tags: vec![],
                    base_difficulty: 1,
                    current_difficulty: 1.0,
                    parent_kata_id: None,
                    variation_params: None,
                    next_review_at: None,
                    last_reviewed_at: None,
                    current_ease_factor: 2.5,
                    current_interval_days: 1,
                    current_repetition_count: 0,
                    fsrs_stability: 1.0,
                    fsrs_difficulty: 1.0,
                    fsrs_elapsed_days: 0,
                    fsrs_scheduled_days: 0,
                    fsrs_reps: 0,
                    fsrs_lapses: 0,
                    fsrs_state: "New".to_string(),
                    scheduler_type: "SM2".to_string(),
                    created_at: Utc::now(),
                },
                Kata {
                    id: 2,
                    name: "kata2".to_string(),
                    category: "test".to_string(),
                    description: "Test".to_string(),
                    tags: vec![],
                    base_difficulty: 2,
                    current_difficulty: 2.0,
                    parent_kata_id: None,
                    variation_params: None,
                    next_review_at: None,
                    last_reviewed_at: None,
                    current_ease_factor: 2.5,
                    current_interval_days: 1,
                    current_repetition_count: 0,
                    fsrs_stability: 1.0,
                    fsrs_difficulty: 1.0,
                    fsrs_elapsed_days: 0,
                    fsrs_scheduled_days: 0,
                    fsrs_reps: 0,
                    fsrs_lapses: 0,
                    fsrs_state: "New".to_string(),
                    scheduler_type: "SM2".to_string(),
                    created_at: Utc::now(),
                },
            ],
            deck_kata_tags: HashMap::new(),
            deck_selected: 0,
            deck_scroll_offset: 0,
            all_available_katas: vec![],
            filtered_available_katas: vec![],
            all_selected: 0,
            all_scroll_offset: 0,
            kata_ids_in_deck: HashSet::new(),
            search_mode: false,
            search_query: String::new(),
            category_filter_mode: false,
            available_categories: vec![],
            selected_categories: Vec::new(),
            category_selected_index: 0,
            sort_mode: SortMode::Name,
            sort_ascending: true,
        };

        library.handle_input(KeyCode::Char('j'));
        assert_eq!(library.deck_selected, 1);

        library.handle_input(KeyCode::Char('k'));
        assert_eq!(library.deck_selected, 0);
    }

    #[test]
    fn test_library_handle_input_navigation_all_katas() {
        let katas = vec![
            AvailableKata {
                name: "kata1".to_string(),
                category: "test".to_string(),
                tags: vec![],
                base_difficulty: 1,
                description: "Test".to_string(),
                dependencies: vec![],
            },
            AvailableKata {
                name: "kata2".to_string(),
                category: "test".to_string(),
                tags: vec![],
                base_difficulty: 2,
                description: "Test".to_string(),
                dependencies: vec![],
            },
        ];

        let mut library = Library {
            active_tab: LibraryTab::AllKatas,
            deck_katas: vec![],
            deck_kata_tags: HashMap::new(),
            deck_selected: 0,
            deck_scroll_offset: 0,
            all_available_katas: katas.clone(),
            filtered_available_katas: katas,
            all_selected: 0,
            all_scroll_offset: 0,
            kata_ids_in_deck: HashSet::new(),
            search_mode: false,
            search_query: String::new(),
            category_filter_mode: false,
            available_categories: vec!["test".to_string()],
            selected_categories: Vec::new(),
            category_selected_index: 0,
            sort_mode: SortMode::Name,
            sort_ascending: true,
        };

        library.handle_input(KeyCode::Char('j'));
        assert_eq!(library.all_selected, 1);

        library.handle_input(KeyCode::Char('k'));
        assert_eq!(library.all_selected, 0);
    }

    #[test]
    fn test_library_handle_input_add_kata() {
        let katas = vec![AvailableKata {
            name: "test_kata".to_string(),
            category: "test".to_string(),
            tags: vec![],
            base_difficulty: 3,
            description: "Test".to_string(),
            dependencies: vec![],
        }];

        let mut library = Library {
            active_tab: LibraryTab::AllKatas,
            deck_katas: vec![],
            deck_kata_tags: HashMap::new(),
            deck_selected: 0,
            deck_scroll_offset: 0,
            all_available_katas: katas.clone(),
            filtered_available_katas: katas,
            all_selected: 0,
            all_scroll_offset: 0,
            kata_ids_in_deck: HashSet::new(),
            search_mode: false,
            search_query: String::new(),
            category_filter_mode: false,
            available_categories: vec!["test".to_string()],
            selected_categories: Vec::new(),
            category_selected_index: 0,
            sort_mode: SortMode::Name,
            sort_ascending: true,
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
            deck_kata_tags: HashMap::new(),
            deck_selected: 0,
            deck_scroll_offset: 0,
            all_available_katas: vec![],
            filtered_available_katas: vec![],
            all_selected: 0,
            all_scroll_offset: 0,
            kata_ids_in_deck: HashSet::new(),
            search_mode: false,
            search_query: String::new(),
            category_filter_mode: false,
            available_categories: vec![],
            selected_categories: Vec::new(),
            category_selected_index: 0,
            sort_mode: SortMode::Name,
            sort_ascending: true,
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
            deck_kata_tags: HashMap::new(),
            deck_selected: 0,
            deck_scroll_offset: 0,
            all_available_katas: vec![],
            filtered_available_katas: vec![],
            all_selected: 0,
            all_scroll_offset: 0,
            kata_ids_in_deck: HashSet::new(),
            search_mode: false,
            search_query: String::new(),
            category_filter_mode: false,
            available_categories: vec![],
            selected_categories: Vec::new(),
            category_selected_index: 0,
            sort_mode: SortMode::Name,
            sort_ascending: true,
        };

        library.mark_as_added("test_kata");
        assert!(library.kata_ids_in_deck.contains("test_kata"));
    }

    #[test]
    fn test_library_mark_as_removed() {
        let mut library = Library {
            active_tab: LibraryTab::MyDeck,
            deck_katas: vec![],
            deck_kata_tags: HashMap::new(),
            deck_selected: 0,
            deck_scroll_offset: 0,
            all_available_katas: vec![],
            filtered_available_katas: vec![],
            all_selected: 0,
            all_scroll_offset: 0,
            kata_ids_in_deck: HashSet::from_iter(vec!["test_kata".to_string()]),
            search_mode: false,
            search_query: String::new(),
            category_filter_mode: false,
            available_categories: vec![],
            selected_categories: Vec::new(),
            category_selected_index: 0,
            sort_mode: SortMode::Name,
            sort_ascending: true,
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
        assert!(
            result == "Tomorrow" || result == "1d",
            "Expected 'Tomorrow' or '1d', got '{}'",
            result
        );

        let in_5_days = Utc::now() + Duration::days(5) + Duration::hours(1);
        let result = format_due_date(Some(in_5_days));
        assert!(
            result == "5d" || result == "4d",
            "Expected '5d' or '4d', got '{}'",
            result
        );
    }
}
