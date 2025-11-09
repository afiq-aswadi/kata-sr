use crate::config::AppConfig;
use crossterm::event::KeyCode;
use ratatui::{
    layout::{Alignment, Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, ListState, Paragraph},
    Frame,
};

/// Settings screen for viewing and editing application configuration.
pub struct SettingsScreen {
    selected_index: usize,
    list_state: ListState,
    config: AppConfig,
}

/// Actions available on the settings screen.
pub enum SettingsAction {
    None,
    Save,
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
        }
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
                Constraint::Length(3),
            ])
            .split(area);

        // Header
        let header = Paragraph::new(vec![Line::from(vec![Span::styled(
            "Settings",
            Style::default()
                .fg(Color::Yellow)
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

        // Footer with keybindings
        let footer_text = vec![
            Line::from(""),
            Line::from(vec![
                Span::raw("↑/↓ • Navigate  "),
                Span::raw("Esc/q • Back to Dashboard"),
            ]),
        ];
        let footer = Paragraph::new(footer_text)
            .block(Block::default().borders(Borders::ALL))
            .alignment(Alignment::Center);
        frame.render_widget(footer, chunks[2]);
    }

    pub fn handle_input(&mut self, code: KeyCode) -> SettingsAction {
        match code {
            KeyCode::Up => {
                if self.selected_index > 0 {
                    self.selected_index -= 1;
                    self.list_state.select(Some(self.selected_index));
                }
                SettingsAction::None
            }
            KeyCode::Down => {
                let max_index = Self::build_setting_items(&self.config).len().saturating_sub(1);
                if self.selected_index < max_index {
                    self.selected_index += 1;
                    self.list_state.select(Some(self.selected_index));
                }
                SettingsAction::None
            }
            KeyCode::Esc | KeyCode::Char('q') => SettingsAction::Cancel,
            _ => SettingsAction::None,
        }
    }

    fn build_setting_items(config: &AppConfig) -> Vec<ListItem> {
        vec![
            ListItem::new(Line::from(vec![
                Span::styled("[Editor]", Style::default().fg(Color::Cyan)),
            ])),
            ListItem::new(Line::from(format!(
                "  Command: {}",
                config.editor.command
            ))),
            ListItem::new(Line::from(format!(
                "  Args: {}",
                if config.editor.args.is_empty() {
                    "(none)".to_string()
                } else {
                    config.editor.args.join(" ")
                }
            ))),
            ListItem::new(Line::from("")),
            ListItem::new(Line::from(vec![
                Span::styled("[Paths]", Style::default().fg(Color::Cyan)),
            ])),
            ListItem::new(Line::from(format!(
                "  Database: {}",
                config.paths.database
            ))),
            ListItem::new(Line::from(format!(
                "  Templates: {}",
                config.paths.templates
            ))),
            ListItem::new(Line::from("")),
            ListItem::new(Line::from(vec![
                Span::styled("[Display]", Style::default().fg(Color::Cyan)),
            ])),
            ListItem::new(Line::from(format!(
                "  Theme: {}",
                config.display.theme
            ))),
            ListItem::new(Line::from(format!(
                "  Heatmap Days: {}",
                config.display.heatmap_days
            ))),
            ListItem::new(Line::from(format!(
                "  Date Format: {}",
                config.display.date_format
            ))),
            ListItem::new(Line::from("")),
            ListItem::new(Line::from(vec![
                Span::styled("[Review]", Style::default().fg(Color::Cyan)),
            ])),
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
            ListItem::new(Line::from(vec![
                Span::styled("[Library]", Style::default().fg(Color::Cyan)),
            ])),
            ListItem::new(Line::from(format!(
                "  Default Sort: {}",
                config.library.default_sort
            ))),
            ListItem::new(Line::from(format!(
                "  Sort Ascending: {}",
                config.library.default_sort_ascending
            ))),
            ListItem::new(Line::from("")),
            ListItem::new(Line::from(vec![Span::styled(
                "Note: To edit settings, modify ~/.config/kata-sr/config.toml",
                Style::default().fg(Color::Yellow).add_modifier(Modifier::ITALIC),
            )])),
        ]
    }
}
