use crossterm::event::KeyCode;
use ratatui::{
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph},
    Frame,
};

/// Startup screen shown when the app first launches.
pub struct StartupScreen;

/// Actions available on the startup screen.
pub enum StartupAction {
    None,
    Continue,
}

impl StartupScreen {
    pub fn new() -> Self {
        Self
    }

    pub fn render(&self, frame: &mut Frame) {
        let area = frame.size();

        // Create main layout: top padding, content, bottom padding
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Percentage(20),
                Constraint::Percentage(60),
                Constraint::Percentage(20),
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
            Line::from(Span::styled(
                "Press any key to continue...",
                Style::default()
                    .fg(Color::Green)
                    .add_modifier(Modifier::ITALIC),
            )),
        ];

        let paragraph = Paragraph::new(lines)
            .alignment(Alignment::Center)
            .block(Block::default());

        frame.render_widget(paragraph, chunks[1]);
    }

    pub fn handle_input(&self, _code: KeyCode) -> StartupAction {
        // Any key press continues to the dashboard
        StartupAction::Continue
    }
}
