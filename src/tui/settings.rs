use crate::config::AppConfig;
use crossterm::event::KeyCode;
use ratatui::{
    layout::{Alignment, Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, ListState, Paragraph},
    Frame,
};

/// Editable field in the settings screen.
#[derive(Debug, Clone, PartialEq)]
enum EditableField {
    EditorCommand,
    EditorArgs,
    DatabasePath,
    TemplatesPath,
    Theme,
    HeatmapDays,
    DateFormat,
    DailyLimit,
    DefaultRating,
    PersistSortMode,
    DefaultSort,
    DefaultSortAscending,
}

/// Settings screen for viewing and editing application configuration.
pub struct SettingsScreen {
    selected_index: usize,
    list_state: ListState,
    config: AppConfig,
    editing_field: Option<EditableField>,
    input_buffer: String,
    has_unsaved_changes: bool,
}

/// Actions available on the settings screen.
pub enum SettingsAction {
    None,
    Save(AppConfig),
    Cancel,
}

impl SettingsScreen {
    pub fn new(config: AppConfig) -> Self {
        let mut list_state = ListState::default();
        list_state.select(Some(0));

        Self {
            selected_index: 0,
            list_state,
            config,
            editing_field: None,
            input_buffer: String::new(),
            has_unsaved_changes: false,
        }
    }

    /// Gets the editable field at the given index (skips headers and separators).
    fn get_field_at_index(&self, index: usize) -> Option<EditableField> {
        match index {
            1 => Some(EditableField::EditorCommand),
            2 => Some(EditableField::EditorArgs),
            5 => Some(EditableField::DatabasePath),
            6 => Some(EditableField::TemplatesPath),
            9 => Some(EditableField::Theme),
            10 => Some(EditableField::HeatmapDays),
            11 => Some(EditableField::DateFormat),
            14 => Some(EditableField::DailyLimit),
            15 => Some(EditableField::DefaultRating),
            16 => Some(EditableField::PersistSortMode),
            19 => Some(EditableField::DefaultSort),
            20 => Some(EditableField::DefaultSortAscending),
            _ => None,
        }
    }

    /// Starts editing the currently selected field.
    fn start_editing(&mut self) {
        if let Some(field) = self.get_field_at_index(self.selected_index) {
            self.editing_field = Some(field.clone());

            // Pre-populate input buffer with current value
            self.input_buffer = match field {
                EditableField::EditorCommand => self.config.editor.command.clone(),
                EditableField::EditorArgs => self.config.editor.args.join(" "),
                EditableField::DatabasePath => self.config.paths.database.clone(),
                EditableField::TemplatesPath => self.config.paths.templates.clone(),
                EditableField::Theme => self.config.display.theme.clone(),
                EditableField::HeatmapDays => self.config.display.heatmap_days.to_string(),
                EditableField::DateFormat => self.config.display.date_format.clone(),
                EditableField::DailyLimit => self
                    .config
                    .review
                    .daily_limit
                    .map(|l| l.to_string())
                    .unwrap_or_else(|| "".to_string()),
                EditableField::DefaultRating => self.config.review.default_rating.to_string(),
                EditableField::PersistSortMode => self.config.review.persist_sort_mode.to_string(),
                EditableField::DefaultSort => self.config.library.default_sort.clone(),
                EditableField::DefaultSortAscending => {
                    self.config.library.default_sort_ascending.to_string()
                }
            };
        }
    }

    /// Commits the current edit to the config.
    fn commit_edit(&mut self) -> bool {
        if let Some(field) = &self.editing_field {
            let result = match field {
                EditableField::EditorCommand => {
                    if !self.input_buffer.trim().is_empty() {
                        self.config.editor.command = self.input_buffer.trim().to_string();
                        true
                    } else {
                        false
                    }
                }
                EditableField::EditorArgs => {
                    self.config.editor.args = self
                        .input_buffer
                        .split_whitespace()
                        .map(|s| s.to_string())
                        .collect();
                    true
                }
                EditableField::DatabasePath => {
                    if !self.input_buffer.trim().is_empty() {
                        self.config.paths.database = self.input_buffer.trim().to_string();
                        true
                    } else {
                        false
                    }
                }
                EditableField::TemplatesPath => {
                    if !self.input_buffer.trim().is_empty() {
                        self.config.paths.templates = self.input_buffer.trim().to_string();
                        true
                    } else {
                        false
                    }
                }
                EditableField::Theme => {
                    if !self.input_buffer.trim().is_empty() {
                        self.config.display.theme = self.input_buffer.trim().to_string();
                        true
                    } else {
                        false
                    }
                }
                EditableField::HeatmapDays => {
                    if let Ok(days) = self.input_buffer.trim().parse::<usize>() {
                        if days > 0 && days <= 365 {
                            self.config.display.heatmap_days = days;
                            true
                        } else {
                            false
                        }
                    } else {
                        false
                    }
                }
                EditableField::DateFormat => {
                    if !self.input_buffer.trim().is_empty() {
                        self.config.display.date_format = self.input_buffer.trim().to_string();
                        true
                    } else {
                        false
                    }
                }
                EditableField::DailyLimit => {
                    let trimmed = self.input_buffer.trim();
                    if trimmed.is_empty() || trimmed.eq_ignore_ascii_case("unlimited") {
                        self.config.review.daily_limit = None;
                        true
                    } else if let Ok(limit) = trimmed.parse::<usize>() {
                        if limit > 0 {
                            self.config.review.daily_limit = Some(limit);
                            true
                        } else {
                            false
                        }
                    } else {
                        false
                    }
                }
                EditableField::DefaultRating => {
                    if let Ok(rating) = self.input_buffer.trim().parse::<u8>() {
                        if (1..=4).contains(&rating) {
                            self.config.review.default_rating = rating;
                            true
                        } else {
                            false
                        }
                    } else {
                        false
                    }
                }
                EditableField::PersistSortMode => {
                    let trimmed = self.input_buffer.trim().to_lowercase();
                    match trimmed.as_str() {
                        "true" | "yes" | "1" => {
                            self.config.review.persist_sort_mode = true;
                            true
                        }
                        "false" | "no" | "0" => {
                            self.config.review.persist_sort_mode = false;
                            true
                        }
                        _ => false,
                    }
                }
                EditableField::DefaultSort => {
                    let trimmed = self.input_buffer.trim();
                    // Valid sort modes (case-insensitive)
                    let valid_sorts = ["Name", "Difficulty", "Category", "Recent"];
                    if let Some(valid) =
                        valid_sorts.iter().find(|s| s.eq_ignore_ascii_case(trimmed))
                    {
                        self.config.library.default_sort = valid.to_string();
                        true
                    } else {
                        false
                    }
                }
                EditableField::DefaultSortAscending => {
                    let trimmed = self.input_buffer.trim().to_lowercase();
                    match trimmed.as_str() {
                        "true" | "yes" | "1" => {
                            self.config.library.default_sort_ascending = true;
                            true
                        }
                        "false" | "no" | "0" => {
                            self.config.library.default_sort_ascending = false;
                            true
                        }
                        _ => false,
                    }
                }
            };

            if result {
                self.has_unsaved_changes = true;
            }

            self.editing_field = None;
            self.input_buffer.clear();
            result
        } else {
            false
        }
    }

    /// Cancels the current edit.
    fn cancel_edit(&mut self) {
        self.editing_field = None;
        self.input_buffer.clear();
    }

    pub fn render(&mut self, frame: &mut Frame) {
        let area = frame.size();

        // Create layout
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .margin(2)
            .constraints([
                Constraint::Length(3),
                Constraint::Min(10),
                Constraint::Length(if self.editing_field.is_some() { 5 } else { 3 }),
            ])
            .split(area);

        // Header with unsaved changes indicator
        let header_text = if self.has_unsaved_changes {
            "Settings * (unsaved changes)"
        } else {
            "Settings"
        };
        let header = Paragraph::new(vec![Line::from(vec![Span::styled(
            header_text,
            Style::default()
                .fg(if self.has_unsaved_changes {
                    Color::Red
                } else {
                    Color::Yellow
                })
                .add_modifier(Modifier::BOLD),
        )])])
        .block(Block::default().borders(Borders::ALL))
        .alignment(Alignment::Center);
        frame.render_widget(header, chunks[0]);

        // Settings list
        // Clone config to avoid borrow checker issues
        let config = self.config.clone();
        let items = Self::build_setting_items(&config);
        let list = List::new(items)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("Configuration"),
            )
            .highlight_style(
                Style::default()
                    .bg(Color::DarkGray)
                    .add_modifier(Modifier::BOLD),
            )
            .highlight_symbol("➤ ");

        frame.render_stateful_widget(list, chunks[1], &mut self.list_state);

        // Footer with keybindings or edit input
        if let Some(field) = &self.editing_field {
            let field_name = match field {
                EditableField::EditorCommand => "Editor Command",
                EditableField::EditorArgs => "Editor Args (space-separated)",
                EditableField::DatabasePath => "Database Path",
                EditableField::TemplatesPath => "Templates Path",
                EditableField::Theme => "Theme",
                EditableField::HeatmapDays => "Heatmap Days (1-365)",
                EditableField::DateFormat => "Date Format",
                EditableField::DailyLimit => "Daily Limit (empty = unlimited)",
                EditableField::DefaultRating => "Default Rating (1-4)",
                EditableField::PersistSortMode => "Persist Sort Mode (true/false)",
                EditableField::DefaultSort => "Default Sort (Name/Difficulty/Category/Recent)",
                EditableField::DefaultSortAscending => "Sort Ascending (true/false)",
            };

            let footer_text = vec![
                Line::from(vec![Span::styled(
                    format!("Editing: {}", field_name),
                    Style::default()
                        .fg(Color::Cyan)
                        .add_modifier(Modifier::BOLD),
                )]),
                Line::from(vec![
                    Span::raw("> "),
                    Span::styled(&self.input_buffer, Style::default().fg(Color::White)),
                    Span::styled("█", Style::default().fg(Color::Green)),
                ]),
                Line::from(""),
                Line::from("Enter • Save  |  Esc • Cancel"),
            ];
            let footer = Paragraph::new(footer_text).block(Block::default().borders(Borders::ALL));
            frame.render_widget(footer, chunks[2]);
        } else {
            let save_text = if self.has_unsaved_changes {
                vec![
                    Span::styled(
                        "s",
                        Style::default()
                            .fg(Color::Green)
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::raw(" • Save  |  "),
                ]
            } else {
                vec![]
            };

            let mut footer_spans = vec![Span::raw("↑/↓ • Navigate  |  Enter • Edit  |  ")];
            footer_spans.extend(save_text);
            footer_spans.push(Span::raw("Esc/q • "));
            footer_spans.push(if self.has_unsaved_changes {
                Span::styled(
                    "Cancel (unsaved changes will be lost)",
                    Style::default().fg(Color::Red),
                )
            } else {
                Span::raw("Back")
            });

            let footer_text = vec![Line::from(""), Line::from(footer_spans)];
            let footer = Paragraph::new(footer_text)
                .block(Block::default().borders(Borders::ALL))
                .alignment(Alignment::Center);
            frame.render_widget(footer, chunks[2]);
        }
    }

    pub fn handle_input(&mut self, code: KeyCode) -> SettingsAction {
        // Handle editing mode
        if self.editing_field.is_some() {
            match code {
                KeyCode::Enter => {
                    self.commit_edit();
                    SettingsAction::None
                }
                KeyCode::Esc => {
                    self.cancel_edit();
                    SettingsAction::None
                }
                KeyCode::Backspace => {
                    self.input_buffer.pop();
                    SettingsAction::None
                }
                KeyCode::Char(c) => {
                    self.input_buffer.push(c);
                    SettingsAction::None
                }
                _ => SettingsAction::None,
            }
        } else {
            // Handle navigation mode
            match code {
                KeyCode::Up => {
                    if self.selected_index > 0 {
                        self.selected_index -= 1;
                        self.list_state.select(Some(self.selected_index));
                    }
                    SettingsAction::None
                }
                KeyCode::Down => {
                    let max_index = Self::build_setting_items(&self.config)
                        .len()
                        .saturating_sub(1);
                    if self.selected_index < max_index {
                        self.selected_index += 1;
                        self.list_state.select(Some(self.selected_index));
                    }
                    SettingsAction::None
                }
                KeyCode::Enter => {
                    self.start_editing();
                    SettingsAction::None
                }
                KeyCode::Char('s') => {
                    // 's' to save
                    if self.has_unsaved_changes {
                        SettingsAction::Save(self.config.clone())
                    } else {
                        SettingsAction::None
                    }
                }
                KeyCode::Esc | KeyCode::Char('q') => SettingsAction::Cancel,
                _ => SettingsAction::None,
            }
        }
    }

    fn build_setting_items(config: &AppConfig) -> Vec<ListItem<'_>> {
        vec![
            ListItem::new(Line::from(vec![Span::styled(
                "[Editor]",
                Style::default().fg(Color::Cyan),
            )])),
            ListItem::new(Line::from(format!("  Command: {}", config.editor.command))),
            ListItem::new(Line::from(format!(
                "  Args: {}",
                if config.editor.args.is_empty() {
                    "(none)".to_string()
                } else {
                    config.editor.args.join(" ")
                }
            ))),
            ListItem::new(Line::from("")),
            ListItem::new(Line::from(vec![Span::styled(
                "[Paths]",
                Style::default().fg(Color::Cyan),
            )])),
            ListItem::new(Line::from(format!("  Database: {}", config.paths.database))),
            ListItem::new(Line::from(format!(
                "  Templates: {}",
                config.paths.templates
            ))),
            ListItem::new(Line::from("")),
            ListItem::new(Line::from(vec![Span::styled(
                "[Display]",
                Style::default().fg(Color::Cyan),
            )])),
            ListItem::new(Line::from(format!("  Theme: {}", config.display.theme))),
            ListItem::new(Line::from(format!(
                "  Heatmap Days: {}",
                config.display.heatmap_days
            ))),
            ListItem::new(Line::from(format!(
                "  Date Format: {}",
                config.display.date_format
            ))),
            ListItem::new(Line::from("")),
            ListItem::new(Line::from(vec![Span::styled(
                "[Review]",
                Style::default().fg(Color::Cyan),
            )])),
            ListItem::new(Line::from(format!(
                "  Daily Limit: {}",
                config
                    .review
                    .daily_limit
                    .map(|l| l.to_string())
                    .unwrap_or_else(|| "Unlimited".to_string())
            ))),
            ListItem::new(Line::from(format!(
                "  Default Rating: {} ({})",
                config.review.default_rating,
                match config.review.default_rating {
                    1 => "Again",
                    2 => "Hard",
                    3 => "Good",
                    4 => "Easy",
                    _ => "Unknown",
                }
            ))),
            ListItem::new(Line::from(format!(
                "  Persist Sort Mode: {}",
                config.review.persist_sort_mode
            ))),
            ListItem::new(Line::from("")),
            ListItem::new(Line::from(vec![Span::styled(
                "[Library]",
                Style::default().fg(Color::Cyan),
            )])),
            ListItem::new(Line::from(format!(
                "  Default Sort: {}",
                config.library.default_sort
            ))),
            ListItem::new(Line::from(format!(
                "  Sort Ascending: {}",
                config.library.default_sort_ascending
            ))),
        ]
    }
}
