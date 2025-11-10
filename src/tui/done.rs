use chrono::{DateTime, Local, Utc};
use crossterm::event::KeyCode;
use ratatui::{
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Wrap},
    Frame,
};

/// Screen shown when all reviews for the day are completed.
pub struct DoneScreen {
    reviews_completed: i32,
    streak_days: i32,
    next_review_at: Option<DateTime<Utc>>,
}

/// Actions available on the done screen.
pub enum DoneAction {
    None,
    BrowseLibrary,
    ToggleHideFlagged,
}

impl DoneScreen {
    pub fn new(
        reviews_completed: i32,
        streak_days: i32,
        next_review_at: Option<DateTime<Utc>>,
    ) -> Self {
        Self {
            reviews_completed,
            streak_days,
            next_review_at,
        }
    }

    pub fn render(&self, frame: &mut Frame) {
        let area = centered_rect(60, 40, frame.size());
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Green))
            .title("All caught up!");
        let inner = block.inner(area);
        frame.render_widget(block, area);

        let lines = vec![
            Line::from(Span::styled(
                "✓ All done for today!",
                Style::default()
                    .fg(Color::Green)
                    .add_modifier(Modifier::BOLD),
            )),
            Line::from(""),
            Line::from(format!("Reviews completed: {}", self.reviews_completed)),
            Line::from(format!("Current streak: {} day(s)", self.streak_days)),
            Line::from(format!("Next review: {}", self.format_next_review())),
            Line::from(""),
            Line::from("l • Browse library"),
            Line::from("p • Practice any kata (opens library)"),
            Line::from("x • Toggle hide flagged katas"),
            Line::from("q • Quit"),
        ];

        let paragraph = Paragraph::new(lines)
            .alignment(Alignment::Center)
            .wrap(Wrap { trim: true });

        frame.render_widget(paragraph, inner);
    }

    pub fn handle_input(&self, code: KeyCode) -> DoneAction {
        match code {
            KeyCode::Char('l') | KeyCode::Char('p') => DoneAction::BrowseLibrary,
            KeyCode::Char('x') => DoneAction::ToggleHideFlagged,
            _ => DoneAction::None,
        }
    }

    fn format_next_review(&self) -> String {
        match self.next_review_at {
            Some(ts) => {
                let local = ts.with_timezone(&Local);
                format!(
                    "{} at {}",
                    local.format("%A, %b %d"),
                    local.format("%-I:%M %p")
                )
            }
            None => "No upcoming reviews scheduled".to_string(),
        }
    }
}

fn centered_rect(percent_x: u16, percent_y: u16, r: Rect) -> Rect {
    let popup_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((100 - percent_y) / 2),
            Constraint::Percentage(percent_y),
            Constraint::Percentage((100 - percent_y) / 2),
        ])
        .split(r);

    let vertical = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - percent_x) / 2),
            Constraint::Percentage(percent_x),
            Constraint::Percentage((100 - percent_x) / 2),
        ])
        .split(popup_layout[1]);

    vertical[1]
}
