use crate::db::repo::KataRepository;
use crate::tui::heatmap_calendar::HeatmapCalendar;
use crossterm::event::KeyCode;
use ratatui::{
    layout::{Alignment, Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Paragraph},
    Frame,
};

/// Menu items for the startup screen.
const MENU_ITEMS: &[&str] = &["Start Review", "Go to Library", "Go to Settings"];

/// Startup screen shown when the app first launches.
pub struct StartupScreen {
    /// Currently selected menu item index
    selected_index: usize,
    /// GitHub-style heatmap calendar
    heatmap_calendar: HeatmapCalendar,
}

/// Actions available on the startup screen.
pub enum StartupAction {
    None,
    StartReview,
    OpenLibrary,
    OpenSettings,
}

impl StartupScreen {
    pub fn load(repo: &KataRepository, heatmap_days: usize) -> anyhow::Result<Self> {
        let heatmap_calendar = HeatmapCalendar::new(repo, heatmap_days)?;

        Ok(Self {
            selected_index: 0,
            heatmap_calendar,
        })
    }

    pub fn render(&self, frame: &mut Frame) {
        let area = frame.size();

        // Create main layout: top padding, content, heatmap, bottom padding
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Percentage(10),
                Constraint::Min(15),      // ASCII art and menu
                Constraint::Min(12),      // Heatmap
                Constraint::Percentage(10),
            ])
            .split(area);

        // ASCII art and welcome content
        let ascii_art = r#"
    ██╗  ██╗ █████╗ ████████╗ █████╗       ███████╗██████╗
    ██║ ██╔╝██╔══██╗╚══██╔══╝██╔══██╗      ██╔════╝██╔══██╗
    █████╔╝ ███████║   ██║   ███████║█████╗███████╗██████╔╝
    ██╔═██╗ ██╔══██║   ██║   ██╔══██║╚════╝╚════██║██╔══██╗
    ██║  ██╗██║  ██║   ██║   ██║  ██║      ███████║██║  ██║
    ╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝      ╚══════╝╚═╝  ╚═╝
"#;

        let mut lines = vec![
            Line::from(""),
            Line::from(Span::styled(
                ascii_art,
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            )),
            Line::from(""),
            Line::from(Span::styled(
                "Master coding patterns through spaced repetition",
                Style::default().fg(Color::Gray),
            )),
            Line::from(""),
            Line::from(""),
        ];

        // Add menu items
        for (i, &item) in MENU_ITEMS.iter().enumerate() {
            let prefix = if i == self.selected_index { "▶ " } else { "  " };
            let style = if i == self.selected_index {
                Style::default()
                    .fg(Color::Green)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(Color::White)
            };

            lines.push(Line::from(Span::styled(
                format!("{}{}", prefix, item),
                style,
            )));
        }

        lines.push(Line::from(""));
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            "Use ↑/↓ to navigate, Enter to select",
            Style::default()
                .fg(Color::Gray)
                .add_modifier(Modifier::ITALIC),
        )));

        let paragraph = Paragraph::new(lines)
            .alignment(Alignment::Center)
            .block(Block::default());

        frame.render_widget(paragraph, chunks[1]);

        // Render heatmap calendar
        self.heatmap_calendar.render(frame, chunks[2]);
    }

    pub fn handle_input(&mut self, code: KeyCode) -> StartupAction {
        match code {
            KeyCode::Up | KeyCode::Char('k') => {
                if self.selected_index > 0 {
                    self.selected_index -= 1;
                }
                StartupAction::None
            }
            KeyCode::Down | KeyCode::Char('j') => {
                if self.selected_index < MENU_ITEMS.len() - 1 {
                    self.selected_index += 1;
                }
                StartupAction::None
            }
            KeyCode::Enter => {
                match self.selected_index {
                    0 => StartupAction::StartReview,
                    1 => StartupAction::OpenLibrary,
                    2 => StartupAction::OpenSettings,
                    _ => StartupAction::None,
                }
            }
            _ => StartupAction::None,
        }
    }
}
