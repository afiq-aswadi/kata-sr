//! Course Library screen for browsing and selecting guided courses.
//!
//! This module provides a list view of available courses with their progress status.

use anyhow::Result;
use crossterm::event::KeyCode;
use ratatui::{
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, Paragraph, Wrap},
    Frame,
};

use crate::core::course_loader::{load_available_courses, AvailableCourse};
use crate::db::repo::{Course, CourseProgress, KataRepository};

/// Action returned by course library input handling.
#[derive(Debug)]
pub enum CourseLibraryAction {
    /// No action taken
    None,
    /// Start or continue a course
    SelectCourse(Course),
    /// View course details before starting
    ViewDetails(Course),
    /// Return to dashboard
    Back,
}

/// Course Library screen state for browsing available courses.
pub struct CourseLibrary {
    /// All courses from database with their progress
    pub courses: Vec<(Course, Option<CourseProgress>)>,
    /// Selected index in the list
    pub selected: usize,
    /// Scroll offset (first visible row)
    pub scroll_offset: usize,
}

impl CourseLibrary {
    /// Loads the course library from the database.
    pub fn load(repo: &KataRepository) -> Result<Self> {
        let courses = repo.get_courses_with_progress()?;

        Ok(Self {
            courses,
            selected: 0,
            scroll_offset: 0,
        })
    }

    /// Handles keyboard input and returns an action.
    pub fn handle_input(&mut self, code: KeyCode) -> CourseLibraryAction {
        match code {
            KeyCode::Up | KeyCode::Char('k') => {
                if self.selected > 0 {
                    self.selected -= 1;
                    self.adjust_scroll();
                }
                CourseLibraryAction::None
            }
            KeyCode::Down | KeyCode::Char('j') => {
                if !self.courses.is_empty() && self.selected < self.courses.len() - 1 {
                    self.selected += 1;
                    self.adjust_scroll();
                }
                CourseLibraryAction::None
            }
            KeyCode::Enter | KeyCode::Char(' ') => {
                if let Some((course, _)) = self.courses.get(self.selected) {
                    CourseLibraryAction::SelectCourse(course.clone())
                } else {
                    CourseLibraryAction::None
                }
            }
            KeyCode::Char('i') | KeyCode::Char('d') => {
                // View course details
                if let Some((course, _)) = self.courses.get(self.selected) {
                    CourseLibraryAction::ViewDetails(course.clone())
                } else {
                    CourseLibraryAction::None
                }
            }
            KeyCode::Esc => CourseLibraryAction::Back,
            _ => CourseLibraryAction::None,
        }
    }

    /// Renders the course library screen.
    pub fn render(&mut self, frame: &mut Frame) {
        let area = frame.size();

        // Main layout: header + list + footer
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3), // Header
                Constraint::Min(0),    // Course list
                Constraint::Length(3), // Footer
            ])
            .split(area);

        // Render header
        self.render_header(frame, chunks[0]);

        // Render course list
        self.render_course_list(frame, chunks[1]);

        // Render footer
        self.render_footer(frame, chunks[2]);
    }

    fn render_header(&self, frame: &mut Frame, area: Rect) {
        let header = Paragraph::new("Guided Courses")
            .style(Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))
            .alignment(Alignment::Center)
            .block(Block::default().borders(Borders::ALL));

        frame.render_widget(header, area);
    }

    fn render_course_list(&mut self, frame: &mut Frame, area: Rect) {
        if self.courses.is_empty() {
            let empty_msg = Paragraph::new("No courses available.\n\nAdd courses to the 'courses/' directory and import them using the debug commands.")
                .style(Style::default().fg(Color::Gray))
                .alignment(Alignment::Center)
                .wrap(Wrap { trim: false })
                .block(Block::default().borders(Borders::ALL).title(" Courses "));

            frame.render_widget(empty_msg, area);
            return;
        }

        // Calculate visible window
        let list_height = area.height.saturating_sub(2) as usize; // Subtract borders
        let visible_start = self.scroll_offset;
        let visible_end = (visible_start + list_height).min(self.courses.len());

        let items: Vec<ListItem> = self.courses[visible_start..visible_end]
            .iter()
            .enumerate()
            .map(|(i, (course, progress))| {
                let actual_index = visible_start + i;
                let is_selected = actual_index == self.selected;

                let status_icon = if let Some(prog) = progress {
                    if prog.completed_at.is_some() {
                        "✓"
                    } else {
                        "⋯" // In progress
                    }
                } else {
                    " " // Not started
                };

                let status_color = if let Some(prog) = progress {
                    if prog.completed_at.is_some() {
                        Color::Green
                    } else {
                        Color::Yellow
                    }
                } else {
                    Color::Gray
                };

                let line = if is_selected {
                    Line::from(vec![
                        Span::styled(
                            format!(" {} ", status_icon),
                            Style::default().fg(status_color).add_modifier(Modifier::BOLD),
                        ),
                        Span::styled(
                            format!("{} ", course.title),
                            Style::default()
                                .fg(Color::White)
                                .add_modifier(Modifier::BOLD | Modifier::REVERSED),
                        ),
                    ])
                } else {
                    Line::from(vec![
                        Span::styled(
                            format!(" {} ", status_icon),
                            Style::default().fg(status_color),
                        ),
                        Span::styled(format!("{} ", course.title), Style::default().fg(Color::White)),
                    ])
                };

                ListItem::new(line)
            })
            .collect();

        let list = List::new(items)
            .block(Block::default().borders(Borders::ALL).title(format!(
                " Courses ({}/{}) ",
                self.selected + 1,
                self.courses.len()
            )));

        frame.render_widget(list, area);
    }

    fn render_footer(&self, frame: &mut Frame, area: Rect) {
        let help_text = if self.courses.is_empty() {
            "ESC: Back to Dashboard"
        } else {
            "↑/↓: Navigate | ENTER: Start Course | i: Info | ESC: Back"
        };

        let footer = Paragraph::new(help_text)
            .style(Style::default().fg(Color::DarkGray))
            .alignment(Alignment::Center)
            .block(Block::default().borders(Borders::ALL));

        frame.render_widget(footer, area);
    }

    /// Adjusts scroll offset to keep selected item visible.
    fn adjust_scroll(&mut self) {
        // Assume a reasonable list height (we don't have the exact area here)
        let list_height = 20; // Approximate visible rows

        if self.selected < self.scroll_offset {
            self.scroll_offset = self.selected;
        } else if self.selected >= self.scroll_offset + list_height {
            self.scroll_offset = self.selected.saturating_sub(list_height - 1);
        }
    }
}
