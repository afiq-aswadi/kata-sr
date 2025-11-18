//! Course Detail screen for viewing and navigating course sections.
//!
//! This module provides section navigation, HTML viewing, and progress tracking.

use anyhow::Result;
use crossterm::event::KeyCode;
use ratatui::{
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, Paragraph, Wrap},
    Frame,
};

use crate::db::repo::{Course, CourseProgress, CourseSection, KataRepository};

/// Action returned by course detail input handling.
#[derive(Debug)]
pub enum CourseDetailAction {
    /// No action taken
    None,
    /// View section HTML content in browser
    ViewSection(CourseSection),
    /// Practice the exercise kata associated with a section
    PracticeExercise(String), // kata_name
    /// Move to next section
    NextSection,
    /// Move to previous section
    PreviousSection,
    /// Return to course library
    Back,
}

/// Course Detail screen state for viewing course sections.
pub struct CourseDetail {
    /// The course being viewed
    pub course: Course,
    /// All sections for this course
    pub sections: Vec<CourseSection>,
    /// Current progress for this course
    pub progress: Option<CourseProgress>,
    /// Currently selected section index
    pub current_section: usize,
    /// Scroll offset for section list
    pub scroll_offset: usize,
}

impl CourseDetail {
    /// Loads the course detail view from the database.
    ///
    /// If the user has previous progress, resume from the last accessed section.
    pub fn load(repo: &KataRepository, course: Course) -> Result<Self> {
        let sections = repo.get_course_sections(course.id)?;
        let progress = repo.get_course_progress(course.id)?;

        // Resume from last section if available
        let current_section = if let Some(ref prog) = progress {
            if let Some(last_section_id) = prog.last_section_id {
                // Find the index of the last accessed section
                sections
                    .iter()
                    .position(|s| s.id == last_section_id)
                    .unwrap_or(0)
            } else {
                0
            }
        } else {
            0
        };

        Ok(Self {
            course,
            sections,
            progress,
            current_section,
            scroll_offset: 0,
        })
    }

    /// Handles keyboard input and returns an action.
    pub fn handle_input(&mut self, code: KeyCode) -> CourseDetailAction {
        match code {
            KeyCode::Up | KeyCode::Char('k') => {
                if self.current_section > 0 {
                    self.current_section -= 1;
                    self.adjust_scroll();
                }
                CourseDetailAction::None
            }
            KeyCode::Down | KeyCode::Char('j') => {
                if !self.sections.is_empty() && self.current_section < self.sections.len() - 1 {
                    self.current_section += 1;
                    self.adjust_scroll();
                }
                CourseDetailAction::None
            }
            KeyCode::Char('n') | KeyCode::Right => {
                // Next section
                if !self.sections.is_empty() && self.current_section < self.sections.len() - 1 {
                    self.current_section += 1;
                    self.adjust_scroll();
                }
                CourseDetailAction::NextSection
            }
            KeyCode::Char('p') | KeyCode::Left => {
                // Previous section
                if self.current_section > 0 {
                    self.current_section -= 1;
                    self.adjust_scroll();
                }
                CourseDetailAction::PreviousSection
            }
            KeyCode::Char('v') | KeyCode::Enter => {
                // View current section in browser
                if let Some(section) = self.sections.get(self.current_section) {
                    CourseDetailAction::ViewSection(section.clone())
                } else {
                    CourseDetailAction::None
                }
            }
            KeyCode::Char('e') => {
                // Practice exercise if available
                if let Some(section) = self.sections.get(self.current_section) {
                    if let Some(ref kata_name) = section.exercise_kata_name {
                        CourseDetailAction::PracticeExercise(kata_name.clone())
                    } else {
                        CourseDetailAction::None
                    }
                } else {
                    CourseDetailAction::None
                }
            }
            KeyCode::Esc => CourseDetailAction::Back,
            _ => CourseDetailAction::None,
        }
    }

    /// Renders the course detail screen.
    pub fn render(&mut self, frame: &mut Frame) {
        let area = frame.size();

        // Main layout: header + content + footer
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(5), // Header (course info)
                Constraint::Min(0),    // Section list + description
                Constraint::Length(3), // Footer
            ])
            .split(area);

        // Render header
        self.render_header(frame, chunks[0]);

        // Split content area: sections list + section description
        let content_chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Percentage(40), // Section list
                Constraint::Percentage(60), // Section description
            ])
            .split(chunks[1]);

        // Render section list
        self.render_section_list(frame, content_chunks[0]);

        // Render section description
        self.render_section_description(frame, content_chunks[1]);

        // Render footer
        self.render_footer(frame, chunks[2]);
    }

    fn render_header(&self, frame: &mut Frame, area: Rect) {
        let progress_info = if let Some(ref prog) = self.progress {
            if prog.completed_at.is_some() {
                format!(" [Completed ✓] ")
            } else {
                format!(
                    " [In Progress: {}/{}] ",
                    self.current_section + 1,
                    self.sections.len()
                )
            }
        } else {
            format!(" [Not Started] ")
        };

        let header_text = vec![
            Line::from(vec![
                Span::styled(
                    format!("Course: {}", self.course.title),
                    Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
                ),
                Span::styled(progress_info, Style::default().fg(Color::Yellow)),
            ]),
            Line::from(format!("Description: {}", self.course.description)),
        ];

        let header = Paragraph::new(header_text)
            .alignment(Alignment::Left)
            .block(Block::default().borders(Borders::ALL).title(" Course "));

        frame.render_widget(header, area);
    }

    fn render_section_list(&mut self, frame: &mut Frame, area: Rect) {
        if self.sections.is_empty() {
            let empty_msg = Paragraph::new("No sections in this course.")
                .style(Style::default().fg(Color::Gray))
                .alignment(Alignment::Center)
                .block(Block::default().borders(Borders::ALL).title(" Sections "));

            frame.render_widget(empty_msg, area);
            return;
        }

        // Calculate visible window
        let list_height = area.height.saturating_sub(2) as usize;
        let visible_start = self.scroll_offset;
        let visible_end = (visible_start + list_height).min(self.sections.len());

        let items: Vec<ListItem> = self.sections[visible_start..visible_end]
            .iter()
            .enumerate()
            .map(|(i, section)| {
                let actual_index = visible_start + i;
                let is_selected = actual_index == self.current_section;

                let prefix = format!("{}. ", actual_index + 1);
                let exercise_marker = if section.exercise_kata_name.is_some() {
                    " [E]"
                } else {
                    ""
                };

                let line = if is_selected {
                    Line::from(vec![
                        Span::styled(prefix, Style::default().fg(Color::Cyan)),
                        Span::styled(
                            format!("{}{}", section.title, exercise_marker),
                            Style::default()
                                .fg(Color::White)
                                .add_modifier(Modifier::BOLD | Modifier::REVERSED),
                        ),
                    ])
                } else {
                    Line::from(vec![
                        Span::styled(prefix, Style::default().fg(Color::DarkGray)),
                        Span::styled(
                            format!("{}{}", section.title, exercise_marker),
                            Style::default().fg(Color::White),
                        ),
                    ])
                };

                ListItem::new(line)
            })
            .collect();

        let list = List::new(items)
            .block(Block::default().borders(Borders::ALL).title(format!(
                " Sections ({}/{}) ",
                self.current_section + 1,
                self.sections.len()
            )));

        frame.render_widget(list, area);
    }

    fn render_section_description(&self, frame: &mut Frame, area: Rect) {
        if let Some(section) = self.sections.get(self.current_section) {
            let mut lines = vec![
                Line::from(vec![Span::styled(
                    &section.title,
                    Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
                )]),
                Line::from(""),
            ];

            if let Some(ref kata_name) = section.exercise_kata_name {
                lines.push(Line::from(vec![
                    Span::styled("Exercise: ", Style::default().fg(Color::Yellow)),
                    Span::styled(kata_name, Style::default().fg(Color::White)),
                ]));
                lines.push(Line::from(""));
                lines.push(Line::from(vec![Span::styled(
                    "Press 'e' to practice this exercise",
                    Style::default().fg(Color::Green),
                )]));
            }

            lines.push(Line::from(""));
            lines.push(Line::from(vec![Span::styled(
                "Press 'v' or ENTER to view this section in your browser",
                Style::default().fg(Color::Cyan),
            )]));

            let desc = Paragraph::new(lines)
                .wrap(Wrap { trim: false })
                .block(
                    Block::default()
                        .borders(Borders::ALL)
                        .title(" Section Details "),
                );

            frame.render_widget(desc, area);
        }
    }

    fn render_footer(&self, frame: &mut Frame, area: Rect) {
        let help_text = "↑/↓: Navigate | v/ENTER: View in Browser | e: Practice Exercise | ←/p: Prev | →/n: Next | ESC: Back";

        let footer = Paragraph::new(help_text)
            .style(Style::default().fg(Color::DarkGray))
            .alignment(Alignment::Center)
            .block(Block::default().borders(Borders::ALL));

        frame.render_widget(footer, area);
    }

    /// Adjusts scroll offset to keep selected section visible.
    fn adjust_scroll(&mut self) {
        let list_height = 20; // Approximate

        if self.current_section < self.scroll_offset {
            self.scroll_offset = self.current_section;
        } else if self.current_section >= self.scroll_offset + list_height {
            self.scroll_offset = self.current_section.saturating_sub(list_height - 1);
        }
    }

    /// Gets the currently selected section.
    pub fn get_current_section(&self) -> Option<&CourseSection> {
        self.sections.get(self.current_section)
    }
}
