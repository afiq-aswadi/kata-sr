use crate::db::repo::Kata;
use crate::runner::python_runner::{TestResult, TestResults};
use anyhow::Context;
use crossterm::event::KeyCode;
use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Clear, List, ListItem, ListState, Paragraph, Wrap},
    Frame,
};
use std::process::Command;

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
    flag_popup_active: bool,
    flag_reason: String,
    flag_cursor_position: usize,
    gave_up: bool,
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
            flag_popup_active: false,
            flag_reason: String::new(),
            flag_cursor_position: 0,
            gave_up: false,
        }
    }

    pub fn mark_rating_submitted(&mut self, rating: u8, remaining_due: usize) {
        self.rating_submitted = true;
        self.submitted_rating = Some(rating);
        self.remaining_due_after_submit = Some(remaining_due);
    }

    /// Returns a reference to the test results.
    pub fn get_results(&self) -> &TestResults {
        &self.results
    }

    /// Update the kata after it's been modified in the database.
    /// This ensures the flag popup shows the current flag status.
    pub fn update_kata(&mut self, kata: Kata) {
        self.kata = kata;
    }

    /// Check if this kata uses matplotlib or plotly for visualization.
    fn is_plot_kata(&self) -> bool {
        self.kata.category.to_lowercase().contains("matplotlib")
            || self.kata.category.to_lowercase().contains("plotly")
            || self.kata.tags.contains(&"matplotlib".to_string())
            || self.kata.tags.contains(&"plotly".to_string())
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

        if self.flag_popup_active {
            self.render_flag_popup(frame);
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
        // Check if this is preview mode (kata not in deck)
        let is_preview_mode = self.kata.id == -1;

        if is_preview_mode {
            // Preview mode: no rating, just back to library
            let is_plot_kata = self.is_plot_kata();

            let status_text = if self.results.passed {
                "Tests passed! This was a preview attempt.\nAdd this kata to your deck to track progress and schedule reviews."
            } else {
                "Tests failed. This was a preview attempt.\nFix your implementation and retry, or view the solution."
            };

            let mut actions_text = if self.results.passed {
                "\n[Esc] Back to library".to_string()
            } else {
                "\n[r] Retry (keep edits)    [g] Give up (view solution)    [Esc] Back to library"
                    .to_string()
            };

            if is_plot_kata {
                actions_text.push_str("\n[v] View plot comparison");
            }

            let text = format!("{}{}", status_text, actions_text);
            let block = Paragraph::new(text)
                .wrap(Wrap { trim: false })
                .block(Block::default().borders(Borders::ALL).title("Preview Mode"));
            frame.render_widget(block, area);
            return;
        }

        if !self.results.passed && !self.rating_submitted {
            let is_plot_kata = self.is_plot_kata();

            let mut text = "Tests failed. Fix your implementation before rating.\n[r] Retry (keep edits)    [g] Give up (view solution)    [Esc] Back to dashboard\n[o] Inspect selected test output    [s] Settings".to_string();

            if is_plot_kata {
                text.push_str("\n[v] View plot comparison");
            }

            let block = Paragraph::new(text)
                .block(Block::default().borders(Borders::ALL).title("Next steps"));
            frame.render_widget(block, area);
            return;
        }

        if !self.results.passed && self.rating_submitted {
            let is_plot_kata = self.is_plot_kata();

            let remaining_msg = match self.remaining_due_after_submit {
                Some(0) => "No more katas due today.".to_string(),
                Some(count) => format!("{} kata(s) still due.", count),
                None => "Loading queue...".to_string(),
            };

            let mut lines = vec![
                "Gave up and viewed solution. Rating saved: Again".to_string(),
                remaining_msg,
                "[Enter/d] Dashboard   [n] Next due   [r] Review picker   [s] Settings".to_string(),
            ];

            if is_plot_kata {
                lines.push("[v] View plot comparison".to_string());
            }

            let block = Paragraph::new(lines.join("\n"))
                .block(Block::default().borders(Borders::ALL).title("What next?"));
            frame.render_widget(block, area);
            return;
        }

        if self.rating_submitted {
            let is_plot_kata = self.is_plot_kata();

            let rating_name = self
                .submitted_rating
                .and_then(|r| RATING_LABELS.get((r - 1) as usize).copied()) // Convert 1-4 to 0-3 index
                .unwrap_or("Unknown");
            let remaining_msg = match self.remaining_due_after_submit {
                Some(0) => "No more katas due today.".to_string(),
                Some(count) => format!("{} kata(s) still due.", count),
                None => "Loading queue...".to_string(),
            };

            let mut lines = vec![
                format!("Saved rating: {}", rating_name),
                remaining_msg,
                "[n] Next kata (auto)   [c] Choose different kata   [Enter/d] Dashboard   [s] Settings".to_string(),
            ];

            if is_plot_kata {
                lines.push("[v] View plot comparison".to_string());
            }

            let block = Paragraph::new(lines.join("\n"))
                .block(Block::default().borders(Borders::ALL).title("What next?"));
            frame.render_widget(block, area);
            return;
        }

        let mut lines: Vec<Line> = Vec::new();
        for (idx, label) in RATING_LABELS.iter().enumerate() {
            let rating_value = idx + 1; // Convert 0-3 array index to 1-4 FSRS rating
            let mut spans = vec![
                Span::styled(
                    format!("[{}] ", rating_value),
                    Style::default().fg(Color::Gray),
                ),
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

        let is_plot_kata = self.is_plot_kata();

        let instructions = Line::from(vec![
            Span::raw("Use ←/→ or h/l to move, ↑/↓ or j/k to change selection, numbers 1-4 to jump. Tab focuses tests."),
        ]);
        lines.push(Line::from(""));
        lines.push(instructions);

        let controls_text = "Press Enter to submit rating, [b] to bury (postpone to tomorrow), [f] to flag, or [s] for Settings.";
        lines.push(Line::from(controls_text));

        if is_plot_kata {
            lines.push(Line::from("[v] View plot comparison"));
        }

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

    fn render_flag_popup(&self, frame: &mut Frame) {
        let area = centered_rect(70, 60, frame.size());
        frame.render_widget(Clear, area);

        // Split the popup into sections
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(5), // Kata info
                Constraint::Length(4), // Current status
                Constraint::Length(5), // Reason input
                Constraint::Min(1),    // Instructions
            ])
            .split(area);

        // Kata info section
        let description_preview = if self.kata.description.chars().count() > 80 {
            let truncated: String = self.kata.description.chars().take(80).collect();
            format!("{}...", truncated)
        } else {
            self.kata.description.clone()
        };
        let kata_info = format!(
            "Kata: {}\nCategory: {}\nDescription: {}",
            self.kata.name, self.kata.category, description_preview
        );
        let info_widget = Paragraph::new(kata_info).wrap(Wrap { trim: false }).block(
            Block::default()
                .borders(Borders::ALL)
                .title("Flag Kata as Problematic"),
        );
        frame.render_widget(info_widget, chunks[0]);

        // Current status section
        let current_status = if self.kata.is_problematic {
            let notes = self
                .kata
                .problematic_notes
                .as_deref()
                .unwrap_or("(no reason given)");
            format!("Current Status: FLAGGED\nReason: {}", notes)
        } else {
            "Current Status: NOT FLAGGED".to_string()
        };
        let status_widget = Paragraph::new(current_status)
            .style(Style::default().fg(if self.kata.is_problematic {
                Color::Red
            } else {
                Color::Green
            }))
            .block(Block::default().borders(Borders::ALL).title("Status"));
        frame.render_widget(status_widget, chunks[1]);

        // Reason input section
        let action = if self.kata.is_problematic {
            "Unflag"
        } else {
            "Flag"
        };
        let input_text = if self.flag_reason.is_empty() {
            Span::styled(
                "(optional - press Enter to skip)",
                Style::default().fg(Color::DarkGray),
            )
        } else {
            Span::raw(&self.flag_reason)
        };
        let reason_widget = Paragraph::new(Line::from(vec![input_text]))
            .wrap(Wrap { trim: false })
            .block(Block::default().borders(Borders::ALL).title(format!(
                "Reason (optional) - Type reason to {} this kata",
                action.to_lowercase()
            )));
        frame.render_widget(reason_widget, chunks[2]);

        // Instructions section
        let instructions = if self.kata.is_problematic {
            "[Enter] Unflag kata    [Esc] Cancel"
        } else {
            "[Enter] Flag kata    [Esc] Cancel"
        };
        let instructions_widget = Paragraph::new(instructions)
            .style(Style::default().fg(Color::Cyan))
            .block(Block::default().borders(Borders::ALL).title("Controls"));
        frame.render_widget(instructions_widget, chunks[3]);
    }

    pub fn handle_input(&mut self, code: KeyCode) -> ResultsAction {
        // Handle flag popup input first
        if self.flag_popup_active {
            match code {
                KeyCode::Esc => {
                    // Cancel flagging
                    self.flag_popup_active = false;
                    self.flag_reason.clear();
                    self.flag_cursor_position = 0;
                    return ResultsAction::None;
                }
                KeyCode::Enter => {
                    // Submit flag with reason (or None if empty)
                    self.flag_popup_active = false;
                    let reason = if self.flag_reason.is_empty() {
                        None
                    } else {
                        Some(self.flag_reason.clone())
                    };
                    self.flag_reason.clear();
                    self.flag_cursor_position = 0;
                    return ResultsAction::ToggleFlagWithReason(reason);
                }
                KeyCode::Char(c) => {
                    // Add character to reason
                    // Convert character position to byte index
                    let byte_idx =
                        self.char_pos_to_byte_idx(&self.flag_reason, self.flag_cursor_position);
                    self.flag_reason.insert(byte_idx, c);
                    self.flag_cursor_position += 1;
                    return ResultsAction::None;
                }
                KeyCode::Backspace => {
                    // Remove character before cursor
                    if self.flag_cursor_position > 0 {
                        self.flag_cursor_position -= 1;
                        let byte_idx =
                            self.char_pos_to_byte_idx(&self.flag_reason, self.flag_cursor_position);
                        self.flag_reason.remove(byte_idx);
                    }
                    return ResultsAction::None;
                }
                KeyCode::Delete => {
                    // Remove character at cursor
                    let char_count = self.flag_reason.chars().count();
                    if self.flag_cursor_position < char_count {
                        let byte_idx =
                            self.char_pos_to_byte_idx(&self.flag_reason, self.flag_cursor_position);
                        self.flag_reason.remove(byte_idx);
                    }
                    return ResultsAction::None;
                }
                KeyCode::Left => {
                    // Move cursor left
                    if self.flag_cursor_position > 0 {
                        self.flag_cursor_position -= 1;
                    }
                    return ResultsAction::None;
                }
                KeyCode::Right => {
                    // Move cursor right
                    let char_count = self.flag_reason.chars().count();
                    if self.flag_cursor_position < char_count {
                        self.flag_cursor_position += 1;
                    }
                    return ResultsAction::None;
                }
                KeyCode::Home => {
                    // Move cursor to start
                    self.flag_cursor_position = 0;
                    return ResultsAction::None;
                }
                KeyCode::End => {
                    // Move cursor to end
                    self.flag_cursor_position = self.flag_reason.chars().count();
                    return ResultsAction::None;
                }
                _ => {
                    return ResultsAction::None;
                }
            }
        }

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

        if matches!(code, KeyCode::Char('f')) {
            // Activate flag popup
            self.flag_popup_active = true;
            return ResultsAction::None;
        }

        // Handle 'v' to view plot (only for matplotlib/plotly katas)
        if matches!(code, KeyCode::Char('v'))
            && (self.kata.category.to_lowercase().contains("matplotlib")
                || self.kata.category.to_lowercase().contains("plotly")
                || self.kata.tags.contains(&"matplotlib".to_string())
                || self.kata.tags.contains(&"plotly".to_string()))
        {
            return ResultsAction::ViewPlot;
        }

        if code == KeyCode::Tab {
            self.focus = self.focus.toggle();
            return ResultsAction::None;
        }

        if code == KeyCode::Char('s') {
            return ResultsAction::OpenSettings;
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

        // Check if this is preview mode
        let is_preview_mode = self.kata.id == -1;

        if !self.results.passed && !self.rating_submitted {
            return match code {
                KeyCode::Char('r') => ResultsAction::Retry,
                KeyCode::Char('g') => ResultsAction::GiveUp,
                KeyCode::Esc => {
                    if is_preview_mode {
                        ResultsAction::BackToLibrary
                    } else {
                        ResultsAction::BackToDashboard
                    }
                }
                _ => ResultsAction::None,
            };
        }

        // In preview mode, if tests passed, just go back to library
        if is_preview_mode && self.results.passed {
            return match code {
                KeyCode::Esc => ResultsAction::BackToLibrary,
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
            KeyCode::Char('b') => ResultsAction::BuryCard,
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

    /// Convert a character position (0-indexed) to a byte index for use with String operations.
    /// This ensures we always land on valid UTF-8 character boundaries.
    fn char_pos_to_byte_idx(&self, s: &str, char_pos: usize) -> usize {
        s.char_indices()
            .nth(char_pos)
            .map(|(byte_idx, _)| byte_idx)
            .unwrap_or(s.len())
    }

    pub fn generate_and_view_plot(&self) -> anyhow::Result<()> {
        // Determine if this is a plotly kata
        let is_plotly = self.kata.category.to_lowercase().contains("plotly")
            || self.kata.tags.contains(&"plotly".to_string());

        // Get the path to the user's kata file
        // For preview mode (kata.id == -1), use kata_preview as module name
        let module_suffix = if self.kata.id == -1 {
            "preview".to_string()
        } else {
            self.kata.id.to_string()
        };
        let template_path = std::path::PathBuf::from(format!("/tmp/kata_{}.py", module_suffix));

        if !template_path.exists() {
            anyhow::bail!("Kata file not found at {}", template_path.display());
        }

        // Get the path to the reference solution
        let katas_root = std::env::var("KATA_SR_KATAS_DIR")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| std::path::PathBuf::from("katas"));
        let reference_dir = katas_root.join("exercises").join(&self.kata.name);

        // Convert reference_dir to string with forward slashes (Python accepts these on all platforms)
        // This avoids Windows backslash escape issues when interpolating into Python code
        let reference_dir_str = reference_dir
            .to_str()
            .context("Reference directory path contains invalid UTF-8")?
            .replace('\\', "/");

        // Create a Python script to generate and save both plots
        let plot_script = format!(
            r#"
import sys
sys.path.insert(0, '/tmp')
sys.path.insert(0, '{reference_dir}')

# Determine plot library based on kata type
is_plotly = {is_plotly}

if not is_plotly:
    import matplotlib
    matplotlib.use('Agg')  # Use non-GUI backend
    import matplotlib.pyplot as plt

import inspect

def generate_plot(module_name, output_path, title_prefix):
    """Generate a plot from a module and save it."""
    try:
        # Import the module
        if module_name.startswith('kata_'):
            import kata_{module_suffix} as mod
        else:
            import reference as mod

        # Find the first function in the module
        kata_functions = []
        for name, obj in inspect.getmembers(mod):
            if inspect.isfunction(obj) and not name.startswith('_'):
                kata_functions.append((name, obj))

        if not kata_functions:
            print(f"Error: No functions found in {{module_name}}")
            return False

        func_name, func = kata_functions[0]
        print(f"Calling {{module_name}}.{{func_name}}()...")

        # Get function signature to see if it needs arguments
        sig = inspect.signature(func)

        # Try to run the function based on its signature
        fig = None
        ax = None

        if len(sig.parameters) == 0:
            # No parameters - call directly
            result = func()
            if isinstance(result, tuple):
                fig = result[0]
                ax = result[1] if len(result) > 1 else None
            else:
                fig = result
        else:
            # Function has parameters - need to set up test environment
            if not is_plotly:
                # Matplotlib setup
                fig, ax = plt.subplots()
                ax.plot([0, 1], [0, 1])  # Add some dummy data

                # Try calling with common argument patterns
                params = list(sig.parameters.keys())

                if 'ax' in params or 'axes' in params:
                    # Function modifies an axes object (most common pattern)
                    args = []
                    for param in params:
                        if param in ['ax', 'axes']:
                            args.append(ax)
                        elif param in ['x', 'y', 'width', 'height']:
                            args.append(0.5)
                        elif param == 'text':
                            args.append('Example')
                        elif param == 'data':
                            import numpy as np
                            args.append(np.linspace(0, 10, 100))
                        else:
                            # Default values for unknown params
                            args.append(0.5)

                    func(*args)
                    # Function returns None, figure is modified in-place
                else:
                    # Function likely returns a figure
                    result = func()
                    if isinstance(result, tuple):
                        fig = result[0]
                        ax = result[1] if len(result) > 1 else None
                    else:
                        fig = result
            else:
                # Plotly setup - functions may need parameters
                params = list(sig.parameters.keys())

                # Try to provide common argument patterns
                args = []
                for param in params:
                    if param in ['x', 'y', 'width', 'height', 'value']:
                        args.append(0.5)
                    elif param == 'text':
                        args.append('Example')
                    elif param == 'data':
                        import numpy as np
                        args.append(np.linspace(0, 10, 100))
                    elif param == 'title':
                        args.append('Plot')
                    else:
                        # Default values for unknown params
                        args.append(0.5)

                result = func(*args)
                if isinstance(result, tuple):
                    fig = result[0]
                    ax = None
                else:
                    fig = result
                    ax = None

        if fig is None:
            print(f"Error: No figure created from {{module_name}}")
            return False

        # Add a title to distinguish user vs reference
        if not is_plotly:
            # Matplotlib title handling
            if ax is not None:
                original_title = ax.get_title()
                new_title = f"{{title_prefix}}{{original_title}}" if original_title else title_prefix.strip()
                ax.set_title(new_title)
            else:
                # Get the first axes from the figure
                axes = fig.get_axes()
                if axes:
                    original_title = axes[0].get_title()
                    new_title = f"{{title_prefix}}{{original_title}}" if original_title else title_prefix.strip()
                    axes[0].set_title(new_title)

        # Save the plot (different methods for matplotlib vs plotly)
        if not is_plotly:
            # Matplotlib
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            import matplotlib.pyplot as plt
            plt.close(fig)
        else:
            # Plotly - update title and save
            if hasattr(fig, 'update_layout'):
                current_title = fig.layout.title.text if fig.layout.title else ''
                new_title = f"{{title_prefix}}{{current_title}}" if current_title else title_prefix.strip()
                fig.update_layout(title=new_title)

            # Use write_image for plotly
            fig.write_image(output_path, width=1500, height=900)

        print(f"Plot saved to {{output_path}}")
        return True
    except Exception as e:
        print(f"Error generating plot from {{module_name}}: {{e}}")
        import traceback
        traceback.print_exc()
        return False

# Generate user's plot
user_success = generate_plot('kata_{module_suffix}', '/tmp/kata_plot_user.png', '[Your Solution] ')

# Generate reference plot
ref_success = generate_plot('reference', '/tmp/kata_plot_reference.png', '[Reference] ')

if not user_success and not ref_success:
    sys.exit(1)
"#,
            module_suffix = module_suffix,
            reference_dir = reference_dir_str,
            is_plotly = if is_plotly { "True" } else { "False" }
        );

        // Write the script to a temporary file
        let script_path = std::path::PathBuf::from("/tmp/generate_kata_plot.py");
        std::fs::write(&script_path, plot_script)
            .context("Failed to write plot generation script")?;

        // Delete any stale plot files from previous runs
        // This prevents showing outdated plots if current generation fails
        let user_plot = std::path::Path::new("/tmp/kata_plot_user.png");
        let ref_plot = std::path::Path::new("/tmp/kata_plot_reference.png");

        if user_plot.exists() {
            let _ = std::fs::remove_file(user_plot);
        }
        if ref_plot.exists() {
            let _ = std::fs::remove_file(ref_plot);
        }

        // Get Python interpreter path from environment (same logic as python_runner.rs)
        // Respects KATA_SR_PYTHON override for custom interpreter paths
        let python_path = std::env::var("KATA_SR_PYTHON")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| {
                // Fall back to default venv location
                let katas_root = std::env::var("KATA_SR_KATAS_DIR")
                    .map(std::path::PathBuf::from)
                    .unwrap_or_else(|_| std::path::PathBuf::from("katas"));
                katas_root.join(".venv/bin/python")
            });

        // Run the script to generate the plots
        let output = Command::new(&python_path)
            .arg(&script_path)
            .output()
            .context("Failed to run plot generation script")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let stdout = String::from_utf8_lossy(&output.stdout);
            anyhow::bail!(
                "Failed to generate plot:\nstdout:\n{}\nstderr:\n{}",
                stdout,
                stderr
            );
        }

        // Open both plots (side by side)
        // Opening multiple files at once will display them in separate windows
        let mut has_plots = false;

        // Helper function to open a file with the platform-specific viewer
        let open_file = |path: &str| -> anyhow::Result<()> {
            if cfg!(target_os = "macos") {
                Command::new("open")
                    .arg(path)
                    .spawn()
                    .context("Failed to open plot with 'open'")?;
            } else if cfg!(target_os = "windows") {
                // On Windows, 'start' is a shell built-in, not an executable
                // Must use cmd /C start to invoke it
                Command::new("cmd")
                    .args(["/C", "start", "", path]) // Empty string after start prevents it from interpreting path as window title
                    .spawn()
                    .context("Failed to open plot with 'cmd /C start'")?;
            } else {
                // Linux and other Unix-like systems
                Command::new("xdg-open")
                    .arg(path)
                    .spawn()
                    .context("Failed to open plot with 'xdg-open'")?;
            }
            Ok(())
        };

        if std::path::Path::new("/tmp/kata_plot_user.png").exists() {
            open_file("/tmp/kata_plot_user.png")?;
            has_plots = true;
        }

        if std::path::Path::new("/tmp/kata_plot_reference.png").exists() {
            open_file("/tmp/kata_plot_reference.png")?;
            has_plots = true;
        }

        if !has_plots {
            anyhow::bail!("No plots were generated");
        }

        Ok(())
    }

    pub fn open_solution_in_editor(&mut self, is_give_up: bool) -> anyhow::Result<()> {
        let katas_root = std::env::var("KATA_SR_KATAS_DIR")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| std::path::PathBuf::from("katas"));
        let reference_path = katas_root
            .join("exercises")
            .join(&self.kata.name)
            .join("reference.py");

        // Check if the reference file exists
        if !reference_path.exists() {
            anyhow::bail!(
                "Reference solution not found at {}",
                reference_path.display()
            );
        }

        // Determine which editor to use (respects EDITOR env var)
        let editor = std::env::var("EDITOR").unwrap_or_else(|_| "nvim".to_string());

        // Parse editor command to handle arguments (e.g., "code -w")
        let parts: Vec<&str> = editor.split_whitespace().collect();
        if parts.is_empty() {
            anyhow::bail!("EDITOR environment variable is empty");
        }

        let editor_program = parts[0];
        let editor_args = &parts[1..];

        // Determine if we should add read-only flag based on editor
        let editor_basename = std::path::Path::new(editor_program)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or(editor_program);

        let is_vim_like = matches!(
            editor_basename,
            "vim" | "nvim" | "vi" | "view" | "gvim" | "nvim-qt"
        );

        // Exit alternate screen and disable raw mode to hand control to editor
        let mut stdout = std::io::stdout();
        crossterm::execute!(
            stdout,
            crossterm::terminal::Clear(crossterm::terminal::ClearType::All),
            crossterm::cursor::Show,
            crossterm::terminal::LeaveAlternateScreen,
        )
        .context("failed to leave alternate screen before launching editor")?;

        crossterm::terminal::disable_raw_mode()
            .context("failed to disable raw mode before launching editor")?;

        // Launch editor with reference solution
        let mut cmd = Command::new(editor_program);

        // Add user's editor arguments
        cmd.args(editor_args);

        // Add read-only flag only for vim-like editors
        if is_vim_like {
            cmd.arg("-R");
        }

        cmd.arg(&reference_path);

        let status_result = cmd.status();

        // Re-enable raw mode and re-enter alternate screen to restore TUI
        crossterm::terminal::enable_raw_mode()
            .context("failed to re-enable raw mode after exiting editor")?;

        crossterm::execute!(
            stdout,
            crossterm::terminal::EnterAlternateScreen,
            crossterm::terminal::Clear(crossterm::terminal::ClearType::All),
            crossterm::cursor::Hide,
            crossterm::cursor::MoveTo(0, 0),
        )
        .context("failed to re-enter alternate screen after exiting editor")?;

        // Flush to ensure all commands are executed
        use std::io::Write;
        stdout
            .flush()
            .context("failed to flush stdout after terminal reset")?;

        let editor_status =
            status_result.with_context(|| format!("failed to launch editor: {}", editor))?;

        if !editor_status.success() {
            anyhow::bail!(
                "Editor '{}' exited with non-zero status (code {:?}). Solution not viewed.",
                editor,
                editor_status.code()
            );
        }

        // Mark that we gave up if that's the case
        if is_give_up {
            self.gave_up = true;
        }

        Ok(())
    }
}

pub enum ResultsAction {
    None,
    SubmitRating(u8),
    BuryCard,
    Retry,
    GiveUp,
    BackToDashboard,
    BackToLibrary,
    StartNextDue,
    ReviewAnother,
    OpenSettings,
    ToggleFlagWithReason(Option<String>),
    SolutionViewed, // Signal that external editor was used and terminal needs clearing
    ViewPlot,       // Generate and display plot for matplotlib katas
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
