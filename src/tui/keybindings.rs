//! Centralized keybinding system for the TUI.
//!
//! This module defines all keyboard shortcuts used throughout the application
//! and provides a help screen for displaying them to users.
//!
//! # Keybinding Organization
//!
//! Keybindings are organized by screen:
//! - **Global**: Available in all screens (quit, help)
//! - **Dashboard**: Main menu navigation and kata selection
//! - **Practice**: Kata editing and test execution
//! - **Results**: Test results and difficulty rating
//!
//! # Usage
//!
//! ```rust
//! use kata_sr::tui::keybindings::{get_keybindings, render_help_screen};
//!
//! // Get all keybindings
//! let keybindings = get_keybindings();
//!
//! // Render help screen in TUI
//! // render_help_screen(&mut frame);
//! ```

use ratatui::{
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, Paragraph},
    Frame,
};

/// Represents a single keybinding with its key and description.
#[derive(Debug, Clone)]
pub struct Keybinding {
    pub key: &'static str,
    pub description: &'static str,
}

impl Keybinding {
    /// Creates a new keybinding.
    pub const fn new(key: &'static str, description: &'static str) -> Self {
        Self { key, description }
    }
}

/// Collection of all keybindings organized by screen.
#[derive(Debug, Clone)]
pub struct Keybindings {
    pub global: Vec<Keybinding>,
    pub dashboard: Vec<Keybinding>,
    pub practice: Vec<Keybinding>,
    pub results: Vec<Keybinding>,
}

/// Returns all keybindings for the application.
///
/// This is the single source of truth for all keyboard shortcuts.
/// Keybindings are organized by screen for easy reference.
///
/// # Examples
///
/// ```
/// use kata_sr::tui::keybindings::get_keybindings;
///
/// let keybindings = get_keybindings();
/// assert!(!keybindings.global.is_empty());
/// ```
pub fn get_keybindings() -> Keybindings {
    Keybindings {
        global: vec![
            Keybinding::new("q", "Quit application"),
            Keybinding::new("?", "Show this help screen"),
        ],
        dashboard: vec![
            Keybinding::new("j / ↓", "Move down in kata list"),
            Keybinding::new("k / ↑", "Move up in kata list"),
            Keybinding::new("Enter", "Select and start kata practice"),
            Keybinding::new("a", "Show analytics and statistics"),
            Keybinding::new("r", "Refresh kata list"),
        ],
        practice: vec![
            Keybinding::new("e", "Edit kata in nvim"),
            Keybinding::new("t", "Run tests"),
            Keybinding::new("Esc", "Return to dashboard (discard changes)"),
            Keybinding::new("s", "Save progress without rating"),
        ],
        results: vec![
            Keybinding::new("0", "Rate as Again (complete failure)"),
            Keybinding::new("1", "Rate as Hard (struggled but passed)"),
            Keybinding::new("2", "Rate as Good (normal difficulty)"),
            Keybinding::new("3", "Rate as Easy (too easy)"),
            Keybinding::new("j / ↓", "Navigate to next rating option"),
            Keybinding::new("k / ↑", "Navigate to previous rating option"),
            Keybinding::new("Enter", "Submit selected rating"),
            Keybinding::new("Esc", "Return to dashboard without rating"),
        ],
    }
}

/// Renders the help screen showing all keybindings.
///
/// The help screen is organized into sections by screen (Global, Dashboard, etc.)
/// with each keybinding displayed in a readable format.
///
/// # Arguments
///
/// * `frame` - The ratatui frame to render into
///
/// # Layout
///
/// The help screen uses a centered modal-style layout with:
/// - Title at the top
/// - Keybindings grouped by screen
/// - Instructions at the bottom
pub fn render_help_screen(frame: &mut Frame) {
    let keybindings = get_keybindings();

    // create centered area for help modal
    let area = frame.size();
    let help_area = centered_rect(80, 90, area);

    // clear background
    frame.render_widget(
        Block::default()
            .style(Style::default().bg(Color::Black))
            .borders(Borders::NONE),
        area,
    );

    // create main help container
    let help_block = Block::default()
        .title(" Help - Keybindings ")
        .title_alignment(Alignment::Center)
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Cyan))
        .style(Style::default().bg(Color::Black));

    let inner_area = help_block.inner(help_area);
    frame.render_widget(help_block, help_area);

    // split into sections
    let sections = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(4),  // Global
            Constraint::Length(7),  // Dashboard
            Constraint::Length(6),  // Practice
            Constraint::Length(10), // Results
            Constraint::Min(1),     // Spacer
            Constraint::Length(2),  // Footer
        ])
        .split(inner_area);

    // render each section
    render_keybinding_section(frame, sections[0], "Global", &keybindings.global);
    render_keybinding_section(frame, sections[1], "Dashboard", &keybindings.dashboard);
    render_keybinding_section(frame, sections[2], "Practice", &keybindings.practice);
    render_keybinding_section(frame, sections[3], "Results", &keybindings.results);

    // footer
    let footer = Paragraph::new("Press ? or Esc to close this help screen")
        .alignment(Alignment::Center)
        .style(
            Style::default()
                .fg(Color::DarkGray)
                .add_modifier(Modifier::ITALIC),
        );
    frame.render_widget(footer, sections[5]);
}

/// Renders a single section of keybindings.
///
/// # Arguments
///
/// * `frame` - The ratatui frame to render into
/// * `area` - The rectangular area to render in
/// * `title` - The section title (e.g., "Dashboard")
/// * `keybindings` - The list of keybindings to display
fn render_keybinding_section(
    frame: &mut Frame,
    area: Rect,
    title: &str,
    keybindings: &[Keybinding],
) {
    let items: Vec<ListItem> = keybindings
        .iter()
        .map(|kb| {
            let line = Line::from(vec![
                Span::styled(
                    format!("  {:12}", kb.key),
                    Style::default()
                        .fg(Color::Yellow)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::raw("  "),
                Span::styled(kb.description, Style::default().fg(Color::White)),
            ]);
            ListItem::new(line)
        })
        .collect();

    let list = List::new(items).block(
        Block::default()
            .title(format!(" {} ", title))
            .borders(Borders::TOP)
            .border_style(Style::default().fg(Color::DarkGray)),
    );

    frame.render_widget(list, area);
}

/// Creates a centered rectangular area within a larger area.
///
/// # Arguments
///
/// * `percent_x` - Percentage of width to use (0-100)
/// * `percent_y` - Percentage of height to use (0-100)
/// * `r` - The parent rectangular area
///
/// # Returns
///
/// A `Rect` centered within the parent area with the specified dimensions
fn centered_rect(percent_x: u16, percent_y: u16, r: Rect) -> Rect {
    let popup_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((100 - percent_y) / 2),
            Constraint::Percentage(percent_y),
            Constraint::Percentage((100 - percent_y) / 2),
        ])
        .split(r);

    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - percent_x) / 2),
            Constraint::Percentage(percent_x),
            Constraint::Percentage((100 - percent_x) / 2),
        ])
        .split(popup_layout[1])[1]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_keybindings_not_empty() {
        let kb = get_keybindings();
        assert!(!kb.global.is_empty());
        assert!(!kb.dashboard.is_empty());
        assert!(!kb.practice.is_empty());
        assert!(!kb.results.is_empty());
    }

    #[test]
    fn test_global_keybindings_present() {
        let kb = get_keybindings();
        // verify quit and help are present
        assert!(kb.global.iter().any(|k| k.key == "q"));
        assert!(kb.global.iter().any(|k| k.key == "?"));
    }

    #[test]
    fn test_results_rating_keybindings() {
        let kb = get_keybindings();
        // verify all 4 rating keys are present
        assert!(kb.results.iter().any(|k| k.key == "0"));
        assert!(kb.results.iter().any(|k| k.key == "1"));
        assert!(kb.results.iter().any(|k| k.key == "2"));
        assert!(kb.results.iter().any(|k| k.key == "3"));
    }

    #[test]
    fn test_keybinding_new() {
        let kb = Keybinding::new("x", "Test keybinding");
        assert_eq!(kb.key, "x");
        assert_eq!(kb.description, "Test keybinding");
    }

    #[test]
    fn test_centered_rect_dimensions() {
        let parent = Rect::new(0, 0, 100, 100);
        let centered = centered_rect(50, 50, parent);

        // should be centered
        assert_eq!(centered.width, 50);
        assert_eq!(centered.height, 50);
        assert_eq!(centered.x, 25);
        assert_eq!(centered.y, 25);
    }

    #[test]
    fn test_centered_rect_full_size() {
        let parent = Rect::new(0, 0, 100, 100);
        let centered = centered_rect(100, 100, parent);

        // should match parent
        assert_eq!(centered.width, 100);
        assert_eq!(centered.height, 100);
        assert_eq!(centered.x, 0);
        assert_eq!(centered.y, 0);
    }
}
