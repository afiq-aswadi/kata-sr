use crate::db::repo::Kata;
use crate::runner::python_runner::{TestResult, TestResults};
use crossterm::event::KeyCode;
use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Clear, List, ListItem, ListState, Paragraph, Wrap},
    Frame,
};

const RATING_LABELS: [&str; 4] = ["Again", "Hard", "Good", "Easy"];
const DETAIL_PAGE_SIZE: u16 = 20; // Lines to scroll per page in detail mode

pub struct ResultsScreen {
    kata: Kata,
    results: TestResults,
    selected_rating: usize,
    selected_test: usize,
    test_state: ListState,
    focus: ResultsFocus,
    rating_submitted: bool,
    submitted_rating: Option<u8>,
    remaining_due_after_submit: Option<usize>,
    detail_mode: bool,
    detail_scroll: u16,
}

#[derive(Copy, Clone, PartialEq, Eq)]
enum ResultsFocus {
    Rating,
    Tests,
}

impl ResultsFocus {
    fn toggle(self) -> Self {
        match self {
            ResultsFocus::Rating => ResultsFocus::Tests,
            ResultsFocus::Tests => ResultsFocus::Rating,
        }
    }
}

impl ResultsScreen {
    pub fn new(kata: Kata, results: TestResults) -> Self {
        let mut test_state = ListState::default();
        if !results.results.is_empty() {
            test_state.select(Some(0));
        }

        Self {
            kata,
            results,
            selected_rating: 3, // Good (FSRS 1-4 scale)
            selected_test: 0,
            test_state,
            focus: ResultsFocus::Rating,
            rating_submitted: false,
            submitted_rating: None,
            remaining_due_after_submit: None,
            detail_mode: false,
            detail_scroll: 0,
        }
    }

    pub fn mark_rating_submitted(&mut self, rating: u8, remaining_due: usize) {
        self.rating_submitted = true;
        self.submitted_rating = Some(rating);
        self.remaining_due_after_submit = Some(remaining_due);
    }

    pub fn render(&mut self, frame: &mut Frame) {
        let layout = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3),
                Constraint::Min(10),
                Constraint::Length(4),
                Constraint::Length(8),
            ])
            .split(frame.size());

        self.render_header(frame, layout[0]);
        self.render_tests(frame, layout[1]);
        self.render_test_summary(frame, layout[2]);
        self.render_actions(frame, layout[3]);

        if self.detail_mode {
            self.render_detail_overlay(frame);
        }
    }

    fn render_header(&self, frame: &mut Frame, area: Rect) {
        let status = if self.results.passed {
            "All tests passed"
        } else {
            "Some tests failed"
        };
        let total = self.results.num_passed + self.results.num_failed + self.results.num_skipped;
        let header = Paragraph::new(format!(
            "{} · {} ({}/{} passed) · {} ms total",
            status, self.kata.name, self.results.num_passed, total, self.results.duration_ms
        ))
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("Results Summary"),
        );
        frame.render_widget(header, area);
    }

    fn render_tests(&mut self, frame: &mut Frame, area: Rect) {
        let items: Vec<ListItem> = if self.results.results.is_empty() {
            vec![ListItem::new("No tests reported")]
        } else {
            self.results
                .results
                .iter()
                .map(|result| {
                    let (symbol, color) = match result.status.as_str() {
                        "passed" => ("✓", Color::Green),
                        "failed" => ("✗", Color::Red),
                        _ => ("○", Color::Yellow),
                    };
                    let text = format!(
                        "{} {:<40} {:>5} ms",
                        symbol, result.test_name, result.duration_ms
                    );
                    ListItem::new(text).style(Style::default().fg(color))
                })
                .collect()
        };

        if self.results.results.is_empty() {
            self.test_state.select(None);
        } else {
            self.test_state
                .select(Some(self.selected_test.min(self.results.results.len() - 1)));
        }

        let title = match self.focus {
            ResultsFocus::Tests => "Test Results (focused · Tab to switch)",
            ResultsFocus::Rating => "Test Results (Tab to focus)",
        };
        let list = List::new(items)
            .block(Block::default().borders(Borders::ALL).title(title))
            .highlight_symbol("➜ ")
            .highlight_style(
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            );

        frame.render_stateful_widget(list, area, &mut self.test_state);
    }

    fn render_test_summary(&self, frame: &mut Frame, area: Rect) {
        let summary = if let Some(test) = self.selected_test_result() {
            format!(
                "Selected: {} · {} · {} ms\nPress o to view full output. Use Tab + ↑/↓ to navigate tests.",
                test.test_name, test.status, test.duration_ms
            )
        } else {
            "No test selected.".to_string()
        };

        let summary_widget = Paragraph::new(summary)
            .block(Block::default().borders(Borders::ALL).title("Selection"));
        frame.render_widget(summary_widget, area);
    }

    fn render_actions(&self, frame: &mut Frame, area: Rect) {
        if !self.results.passed {
            let text = "Tests failed. Fix your implementation before rating.\n[r] Retry (keep edits)    [Esc] Back to dashboard\n[o] Inspect selected test output";
            let block = Paragraph::new(text)
                .block(Block::default().borders(Borders::ALL).title("Next steps"));
            frame.render_widget(block, area);
            return;
        }

        if self.rating_submitted {
            let rating_name = self
                .submitted_rating
                .and_then(|r| RATING_LABELS.get((r - 1) as usize).copied()) // Convert 1-4 to 0-3 index
                .unwrap_or("Unknown");
            let remaining_msg = match self.remaining_due_after_submit {
                Some(0) => "No more katas due today.".to_string(),
                Some(count) => format!("{} kata(s) still due.", count),
                None => "Loading queue...".to_string(),
            };
            let lines = vec![
                format!("Saved rating: {}", rating_name),
                remaining_msg,
                "[n] Next kata (auto)   [c] Choose different kata   [Enter/d] Dashboard   [o] Inspect output"
                    .to_string(),
            ];
            let block = Paragraph::new(lines.join("\n"))
                .block(Block::default().borders(Borders::ALL).title("What next?"));
            frame.render_widget(block, area);
            return;
        }

        let mut lines: Vec<Line> = Vec::new();
        for (idx, label) in RATING_LABELS.iter().enumerate() {
            let rating_value = idx + 1; // Convert 0-3 array index to 1-4 FSRS rating
            let mut spans = vec![
                Span::styled(format!("[{}] ", rating_value), Style::default().fg(Color::Gray)),
                Span::raw(label.to_string()),
            ];

            if rating_value == self.selected_rating {
                spans = vec![Span::styled(
                    format!("[{}] {}", rating_value, label),
                    Style::default()
                        .fg(Color::Black)
                        .bg(Color::Yellow)
                        .add_modifier(Modifier::BOLD),
                )];
            }

            lines.push(Line::from(spans));
        }

        let instructions = Line::from(vec![
            Span::raw("Use ←/→ or h/l to move, ↑/↓ or j/k to change selection, numbers 1-4 to jump. Tab focuses tests."),
        ]);
        lines.push(Line::from(""));
        lines.push(instructions);
        lines.push(Line::from("Press Enter to submit rating."));

        let title = match self.focus {
            ResultsFocus::Rating => "Rate Difficulty (focused)",
            ResultsFocus::Tests => "Rate Difficulty",
        };
        let rating_block = Paragraph::new(lines)
            .wrap(Wrap { trim: true })
            .block(Block::default().borders(Borders::ALL).title(title));
        frame.render_widget(rating_block, area);
    }

    fn render_detail_overlay(&self, frame: &mut Frame) {
        let area = centered_rect(80, 70, frame.size());
        frame.render_widget(Clear, area);

        let (title, body) = if let Some(test) = self.selected_test_result() {
            let title = format!(
                "{} output · [↑↓/jk] scroll · [PgUp/PgDn/Space/b] page · [Home/End] jump · [o/Esc] close",
                test.test_name
            );
            let body = if test.output.trim().is_empty() {
                "No additional output captured for this test.".to_string()
            } else {
                test.output.clone()
            };
            (title, body)
        } else {
            (
                "No test selected".to_string(),
                "Select a test to see its detailed output.".to_string(),
            )
        };

        let detail = Paragraph::new(body)
            .wrap(Wrap { trim: false })
            .scroll((self.detail_scroll, 0))
            .block(Block::default().borders(Borders::ALL).title(title));
        frame.render_widget(detail, area);
    }

    pub fn handle_input(&mut self, code: KeyCode) -> ResultsAction {
        if self.detail_mode {
            match code {
                KeyCode::Esc | KeyCode::Char('o') => {
                    self.detail_mode = false;
                }
                KeyCode::Down | KeyCode::Char('j') => {
                    self.detail_scroll = self.detail_scroll.saturating_add(1);
                }
                KeyCode::Up | KeyCode::Char('k') => {
                    self.detail_scroll = self.detail_scroll.saturating_sub(1);
                }
                KeyCode::PageDown | KeyCode::Char(' ') => {
                    // Scroll down one viewport height
                    self.detail_scroll = self.detail_scroll.saturating_add(DETAIL_PAGE_SIZE);
                }
                KeyCode::PageUp | KeyCode::Char('b') => {
                    // Scroll up one viewport height
                    self.detail_scroll = self.detail_scroll.saturating_sub(DETAIL_PAGE_SIZE);
                }
                KeyCode::Home | KeyCode::Char('g') => {
                    // Jump to top
                    self.detail_scroll = 0;
                }
                KeyCode::End | KeyCode::Char('G') => {
                    // Jump to bottom (use large number, ratatui will clamp)
                    self.detail_scroll = u16::MAX;
                }
                _ => {}
            }
            return ResultsAction::None;
        }

        if matches!(code, KeyCode::Char('o')) && self.selected_test_result().is_some() {
            self.detail_mode = true;
            self.detail_scroll = 0;
            return ResultsAction::None;
        }

        if code == KeyCode::Tab {
            self.focus = self.focus.toggle();
            return ResultsAction::None;
        }

        if self.focus == ResultsFocus::Tests {
            match code {
                KeyCode::Up | KeyCode::Char('k') => {
                    self.move_test_selection(-1);
                    return ResultsAction::None;
                }
                KeyCode::Down | KeyCode::Char('j') => {
                    self.move_test_selection(1);
                    return ResultsAction::None;
                }
                KeyCode::PageUp => {
                    self.move_test_selection(-5);
                    return ResultsAction::None;
                }
                KeyCode::PageDown => {
                    self.move_test_selection(5);
                    return ResultsAction::None;
                }
                _ => {}
            }
        }

        if !self.results.passed {
            return match code {
                KeyCode::Char('r') => ResultsAction::Retry,
                KeyCode::Esc => ResultsAction::BackToDashboard,
                _ => ResultsAction::None,
            };
        }

        if self.rating_submitted {
            return match code {
                KeyCode::Char('n') => ResultsAction::StartNextDue,
                KeyCode::Char('c') | KeyCode::Char('r') => ResultsAction::ReviewAnother,
                KeyCode::Enter | KeyCode::Char('d') => ResultsAction::BackToDashboard,
                _ => ResultsAction::None,
            };
        }

        match code {
            KeyCode::Left | KeyCode::Char('h') => {
                self.bump_rating(-1);
                ResultsAction::None
            }
            KeyCode::Right | KeyCode::Char('l') => {
                self.bump_rating(1);
                ResultsAction::None
            }
            KeyCode::Up | KeyCode::Char('k') if self.focus == ResultsFocus::Rating => {
                self.bump_rating(-1);
                ResultsAction::None
            }
            KeyCode::Down | KeyCode::Char('j') if self.focus == ResultsFocus::Rating => {
                self.bump_rating(1);
                ResultsAction::None
            }
            KeyCode::Char('1') => {
                self.selected_rating = 1; // Again
                ResultsAction::None
            }
            KeyCode::Char('2') => {
                self.selected_rating = 2; // Hard
                ResultsAction::None
            }
            KeyCode::Char('3') => {
                self.selected_rating = 3; // Good
                ResultsAction::None
            }
            KeyCode::Char('4') => {
                self.selected_rating = 4; // Easy
                ResultsAction::None
            }
            KeyCode::Enter => ResultsAction::SubmitRating(self.selected_rating as u8), // Direct FSRS 1-4 rating
            _ => ResultsAction::None,
        }
    }

    fn selected_test_result(&self) -> Option<&TestResult> {
        self.results.results.get(self.selected_test)
    }

    fn bump_rating(&mut self, delta: isize) {
        let new_rating = (self.selected_rating as isize + delta).clamp(1, 4);
        self.selected_rating = new_rating as usize;
    }

    fn move_test_selection(&mut self, delta: isize) {
        if self.results.results.is_empty() {
            return;
        }

        let max_index = self.results.results.len().saturating_sub(1) as isize;
        let current = self.selected_test as isize;
        let next = (current + delta).clamp(0, max_index);
        self.selected_test = next as usize;
        self.test_state.select(Some(self.selected_test));
    }
}

pub enum ResultsAction {
    None,
    SubmitRating(u8),
    Retry,
    BackToDashboard,
    StartNextDue,
    ReviewAnother,
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
