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

        // Full Luffy ASCII art as background
        let luffy_art = vec![
            "⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠿⢋⣥⣶⡿⠟⠅⡿⠿⠿⠾⠿⠿⠍⠛⠋⢿⣶⣦⣍⡻⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿",
            "⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠟⠡⠾⢟⣋⣀⣠⣤⣴⣶⣶⣶⣾⣿⣷⣶⣶⣶⣬⣁⣉⣙⠻⠦⣉⠿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿",
            "⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠿⢟⣡⣴⣶⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⢿⣿⣿⣿⣿⣶⣦⣀⠙⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿",
            "⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠿⢛⣡⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠿⠿⠛⢉⣴⣿⣿⣿⣿⣿⣿⣿⣿⣿⣦⣉⠻⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿",
            "⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠟⣡⡾⠟⣻⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠛⠉⠁⠀⠀⠀⠀⠀⠴⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣦⣙⠻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿",
            "⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⢋⣤⡞⡣⢐⣾⣿⣿⣿⣿⣿⣿⡿⠟⠋⠀⠀⠀⢠⣦⠀⠀⠀⠀⡀⠀⠀⠀⠉⠻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⢟⢳⡌⠻⣿⣿⣿⣿⣿⣿⣿⣿",
            "⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⢋⣴⠟⡡⢪⢜⡯⣿⣿⢉⡭⢛⠟⠋⠀⠀⠀⣠⣾⠀⣿⣿⣆⠀⠀⠀⢱⣄⠀⠀⠀⠀⠀⠙⠿⣿⢿⡛⢿⡛⣿⢷⡘⡦⣦⢹⣦⠘⣿⣿⣿⣿⣿⣿⣿",
            "⣿⣿⣿⣿⣿⣿⣿⣿⡿⢡⣾⢋⣼⣼⡵⣡⣪⠟⢁⣤⠒⠁⠀⠀⠀⢀⣼⣿⣿⠀⣿⣿⣿⣦⠀⠀⠈⣿⣧⡀⠀⠀⠀⠀⠀⠈⠳⢝⢂⠋⢻⡁⡐⡐⣙⣦⡍⣷⡈⢿⣿⣿⣿⣿⣿",
            "⣿⣿⣿⣿⣿⣿⣿⡟⢀⣿⣿⣿⣿⣿⡟⣱⠟⣠⣿⠁⠀⠀⠀⡄⢀⣾⣿⣿⣿⡄⣿⣿⣿⣿⣷⣄⠀⢹⣿⣿⡄⠀⢠⠀⠀⠀⠀⠀⠀⠁⠡⢴⣴⣾⣿⣿⣿⣿⣿⡄⢻⣿⣿⣿⣿",
            "⣿⣿⣿⣿⣿⣿⡿⢠⣿⣿⣿⣿⣿⣿⠖⣧⣎⣾⠁⠀⠀⢀⡜⠠⠟⠛⠿⢿⣿⣿⣿⣿⣿⣿⣿⣿⣦⠈⠛⣻⣭⣄⠀⣧⠀⠀⠀⠹⣷⠳⡄⠘⠟⢻⣿⣿⣿⣿⣿⣿⡄⢻⣿⣿⣿",
            "⣿⣿⣿⣿⣿⡿⢠⡟⡸⠻⠛⠹⣿⡟⠸⣿⣿⠁⠀⠀⠀⣼⢃⣾⣿⣿⣿⣷⣶⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⣿⣿⣿⣦⢹⡆⠀⠀⠀⠹⠓⠸⡄⠜⠇⠆⠉⣿⣿⣿⣿⣿⠈⣿⣿⣿",
            "⣿⣿⣿⣿⡿⠀⡿⢀⠁⠃⠇⡆⢻⢰⢰⡆⢁⡀⠀⠀⣰⣿⣿⣿⡿⠿⢿⣿⣿⣿⣿⣧⠹⣿⣿⣿⣿⣿⡿⠟⠛⠙⠛⢿⣿⠀⠀⠀⠀⠀⢄⠰⣴⣸⡘⣴⡌⣿⣧⣿⣿⠀⣿⣿⣿",
            "⣿⣿⣿⣿⠀⢸⠀⡇⡎⡆⡆⡇⡈⡈⠚⣠⣿⣿⠀⠀⣿⣿⡿⠃⣠⣤⣤⡀⢙⣿⣿⣟⠀⢻⣿⣿⣿⣿⢠⣶⣿⣷⣄⠘⣿⡆⠃⠀⢹⣿⣾⣧⡈⠁⠃⢸⢸⣷⡍⡿⣿⠀⣿⣿⣿",
            "⣿⣿⣿⣿⡀⢸⠀⡇⡇⡇⡇⣇⠁⣶⣾⣿⣿⣿⡆⢰⣿⣿⠃⣼⣿⣿⣿⣿⣿⣿⣿⣿⢀⡞⢿⣿⣿⣿⣏⠛⠟⠻⠟⠛⠛⢿⣼⢠⣼⣿⣿⣿⣧⠀⠆⠘⠌⡏⠇⣇⣿⠆⣿⣿⣿",
            "⣿⣿⣿⣿⡇⢸⡆⡇⡇⠁⡡⢸⠁⣹⣿⣿⣿⣿⣿⠘⠟⠉⠙⣻⣿⣿⣿⡿⠿⠿⠟⣛⣛⣛⣛⣛⣛⣛⣛⣛⠷⠼⠿⣶⣶⣀⠉⢰⣿⣿⣿⣿⡟⠀⡼⠠⠠⠠⡠⡸⢿⢠⣿⣿⣿",
            "⣿⣿⣿⣿⣷⠘⣧⢹⠘⠀⣁⣡⡁⣿⣿⣿⣿⣿⣿⡆⢴⣶⠻⠟⡋⢩⣵⣶⣶⣿⣿⣿⣿⠿⠿⠿⢿⣿⣿⣿⣿⣷⣄⣬⠙⣿⠂⢍⢿⣿⣿⣿⠏⣸⡷⠠⢁⠁⠁⢠⠁⣾⣿⣿⣿",
            "⣿⣿⣿⣿⣿⠀⢿⢠⠡⠁⡹⢠⡄⢸⣿⣿⣿⣿⣿⣏⠘⣿⡆⢼⣿⣿⣿⡿⠟⠉⠉⠀⠀⠀⠀⠀⠀⠀⠉⠉⠙⠿⠿⡿⢠⣿⢠⣆⠼⠛⠉⣡⣈⠓⢰⣠⠋⠔⣱⢋⣼⣿⣿⣿⣿",
            "⣿⣿⣿⣿⣿⣧⠘⢄⢢⠁⡑⢦⡙⢸⣿⣿⣿⣿⡿⣿⡁⢹⣷⡘⠿⠟⠁⠀⢠⣂⣴⣿⣿⣶⣶⣾⣿⣦⣄⣀⠀⠀⠉⣡⣿⠃⠞⣁⣀⣴⣶⣎⠛⣷⣄⢱⣠⡾⢋⣾⣿⣿⣿⣿⣿",
            "⣿⣿⣿⣿⣿⣿⣶⡈⢾⣦⣹⣾⣷⡈⠻⣿⣿⣿⣿⣿⠟⡂⢿⣿⣆⡀⢰⡾⠿⠿⠿⣛⣛⣛⣛⣛⣛⣛⡻⠿⡧⢀⡴⠿⠿⠈⣾⣿⣿⣿⣿⣿⢣⢸⡏⠈⢋⣴⣿⣿⣿⣿⣿⣿⣿",
            "⣿⣿⣿⣿⣿⣿⣿⣿⣦⡙⠿⣿⣏⡱⠄⠻⣿⣷⢿⠽⣷⠙⠆⢻⣿⣷⣆⣀⠲⠿⣿⣿⣿⣿⣿⣿⣿⠿⠿⣃⣴⡞⢰⣶⠦⢐⣠⣿⣿⣿⣿⠏⠀⠈⣴⠀⣤⠙⣿⣿⣿⣿⣿⣿⣿",
            "⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣶⣬⡑⠿⡿⢷⣌⠻⣍⣶⠞⠗⡆⠀⠉⠻⠿⢿⣷⣶⣦⣭⣭⣭⣭⣭⣶⣶⣾⣟⠉⠰⢻⣿⣷⣿⣿⣿⡿⢿⣿⡂⠆⠀⢿⣦⠹⣃⣉⢿⣿⣿⣿⣿⣿",
            "⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣶⡮⠭⠉⠗⣈⡑⠀⠈⠁⠄⢀⣠⡀⡐⠈⠉⠙⠛⠿⠿⠿⠟⠛⡉⠉⡀⠂⠀⠘⠿⠿⠿⢛⣥⣶⣦⣉⣙⢳⡿⢆⣹⠃⣿⣿⡄⣿⣿⣿⣿⣿",
            "⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠿⣋⣩⣴⣾⣶⣿⣿⣿⣾⣿⣿⣦⣀⡈⠀⣿⣷⣄⠀⠀⠀⢀⡆⠀⠀⡆⣰⢱⡠⢀⣉⣁⣤⣾⣿⣿⣿⣿⣿⣿⣮⡳⠿⣀⣀⠿⢛⣡⣿⣿⣿⣿⣿",
            "⣿⣿⣿⣿⣿⣿⣿⣿⡿⠟⣁⣊⣙⣋⣍⣛⣻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣦⣙⠻⣿⣷⣤⣀⣿⠃⡄⡠⡇⠃⣠⣶⣿⣿⣿⣿⣿⣿⣿⣿⣿⣏⣵⣦⣩⣶⣦⡶⠞⢉⠈⣿⣿⣿⣿⣿",
            "⣿⣿⣿⣿⣿⣿⣿⠟⣡⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⣮⣙⠻⣿⣿⡆⢓⣥⣠⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣗⠈⣿⣿⣿⣿",
            "⣿⣿⣿⣿⣿⠟⣡⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠿⠓⣨⣥⣶⣿⣿⣿⣿⡿⠋⠀⢠⣿⣿⣿⣿⣿⡿⢿⡿⢿⠿⣿⣿⣿⣿⣿⣿⣿⣷⠸⣿⣿⣿",
            "⣿⣿⣿⣿⠃⣼⣿⣿⣿⣿⣿⣿⣿⣿⡿⠿⠛⠛⠛⠋⠛⠛⠿⡏⠳⢿⣦⢇⢹⠇⣸⣿⣿⣿⣿⠟⢛⣛⠁⠀⣴⣶⣿⣿⣿⠟⣡⠂⢀⣡⣞⡻⠿⣿⣿⣿⣿⣿⣩⠛⣿⠀⣿⣿⣿",
            "⣿⣿⣿⣿⡆⣿⡙⢿⡙⣟⠋⠃⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠃⠊⠦⣠⣾⣿⣿⡟⠀⢀⣿⣿⣿⣿⠟⠫⠍⠛⠁⠒⠀⠀⠊⠥⠴⠅⠒⠁⢤⡴⢟⠋⣁⠙⠥⠔⢰⣿⣿⣿",
            "⣿⣿⣿⣿⣇⠻⡙⠀⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣿⣿⣿⣿⣷⣶⣿⡟⡿⠋⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠁⠛⠋⡀⣼⣿⣿⣿",
            "⣿⣿⣿⣿⡿⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠋⠿⡘⣿⠸⠈⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠛⢿⣿⣿",
            "⣿⣿⣿⣿⠁⠀⠀⠀⠀⠀⢠⣶⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⣶⣤⣄⣀⠀⠀⠀⣀⣠⣤⣄⡀⠀⣠⣤⣤⣴⣦⣤⣶⣶⣶⣤⣤⣤⣤⣤⣤⡄⢠⣴⡦⠀⠀⠀⠀⠀⠀⠈⢻⣿",
            "⣿⣿⣿⠃⠀⠀⠀⠀⠀⠀⠘⣿⣿⣇⢸⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⢰⣿⣿⣿⣿⣿⣿⢰⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⢣⣾⡟⣀⣠⣤⣤⣤⣀⡀⠀⠈⢿",
        ];

        // Render background art
        let background_lines: Vec<Line> = luffy_art
            .iter()
            .map(|line| {
                Line::from(Span::styled(
                    *line,
                    Style::default().fg(Color::Rgb(60, 60, 80)), // dim background
                ))
            })
            .collect();

        let background = Paragraph::new(background_lines)
            .alignment(Alignment::Center)
            .block(Block::default());
        frame.render_widget(background, area);

        // Create overlay layout for menu
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Percentage(25),
                Constraint::Min(10), // Menu overlay
                Constraint::Min(12), // Heatmap
                Constraint::Percentage(15),
            ])
            .split(area);

        // Menu overlay
        let mut menu_lines = vec![
            Line::from(""),
            Line::from(Span::styled(
                "KATA-SR",
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            )),
            Line::from(Span::styled(
                "Master coding patterns through spaced repetition",
                Style::default().fg(Color::Gray),
            )),
            Line::from(""),
            Line::from(""),
        ];

        // Add menu items
        for (i, &item) in MENU_ITEMS.iter().enumerate() {
            let prefix = if i == self.selected_index {
                "▶ "
            } else {
                "  "
            };
            let style = if i == self.selected_index {
                Style::default()
                    .fg(Color::Green)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(Color::White)
            };

            menu_lines.push(Line::from(Span::styled(
                format!("{}{}", prefix, item),
                style,
            )));
        }

        menu_lines.push(Line::from(""));
        menu_lines.push(Line::from(""));
        menu_lines.push(Line::from(Span::styled(
            "Use ↑/↓ to navigate, Enter to select",
            Style::default()
                .fg(Color::Gray)
                .add_modifier(Modifier::ITALIC),
        )));

        let menu_overlay = Paragraph::new(menu_lines)
            .alignment(Alignment::Center)
            .block(Block::default());

        frame.render_widget(menu_overlay, chunks[1]);

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
            KeyCode::Enter => match self.selected_index {
                0 => StartupAction::StartReview,
                1 => StartupAction::OpenLibrary,
                2 => StartupAction::OpenSettings,
                _ => StartupAction::None,
            },
            _ => StartupAction::None,
        }
    }
}
