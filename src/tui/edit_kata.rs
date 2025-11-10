//! Edit Kata screen for modifying existing katas through the TUI.
//!
//! This module provides a form-based interface for editing existing katas,
//! with pre-filled values, real-time validation, and external editor support.

use crossterm::event::KeyCode;
use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Style},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, Paragraph, Wrap},
    Frame,
};
use std::collections::HashMap;
use std::path::Path;

use crate::core::kata_generator::KataFormData;
use crate::core::kata_validation::{slugify_kata_name, validate_dependencies, validate_kata_name};
use crate::db::repo::Kata;

/// Action returned by edit kata input handling.
#[derive(Debug)]
pub enum EditKataAction {
    /// No action taken
    None,
    /// Submit the form and update the kata
    Submit {
        kata_id: i64,
        original_slug: String,
        form_data: KataFormData,
        new_slug: String,
    },
    /// Open external editor for a specific file
    OpenEditor { file_path: std::path::PathBuf },
    /// Cancel and return to previous screen
    Cancel,
}

/// Form field being edited.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FormField {
    Name,
    Category,
    Description,
    Difficulty,
    Dependencies,
}

impl FormField {
    /// Returns the next field in tab order.
    fn next(self) -> Self {
        match self {
            FormField::Name => FormField::Category,
            FormField::Category => FormField::Description,
            FormField::Description => FormField::Difficulty,
            FormField::Difficulty => FormField::Dependencies,
            FormField::Dependencies => FormField::Name,
        }
    }

    /// Returns the previous field in tab order.
    fn prev(self) -> Self {
        match self {
            FormField::Name => FormField::Dependencies,
            FormField::Category => FormField::Name,
            FormField::Description => FormField::Category,
            FormField::Difficulty => FormField::Description,
            FormField::Dependencies => FormField::Difficulty,
        }
    }

    /// Returns the display label for this field.
    fn label(self) -> &'static str {
        match self {
            FormField::Name => "Name",
            FormField::Category => "Tags (comma-separated)",
            FormField::Description => "Description",
            FormField::Difficulty => "Difficulty",
            FormField::Dependencies => "Dependencies",
        }
    }
}

/// Edit Kata screen state.
pub struct EditKataScreen {
    // Kata being edited
    pub kata_id: i64,
    pub original_name: String,
    pub original_slug: String,

    // User input fields
    pub name_input: String,
    pub category_input: String,
    pub description_input: String,
    pub difficulty: u8, // 1-5
    pub selected_dependencies: Vec<String>,

    // UI state
    pub current_field: FormField,
    pub cursor_position: usize,

    // Available katas for dependency selection
    pub available_katas: Vec<String>,
    pub dependency_scroll_offset: usize,
    pub dependency_selected_index: usize,

    // Validation state
    pub validation_errors: HashMap<FormField, String>,

    // Confirmation state
    pub showing_confirmation: bool,
    pub confirmation_slug: String,
    pub confirmation_error: Option<String>,

    // File paths for external editor
    pub exercises_dir: std::path::PathBuf,
}

impl EditKataScreen {
    /// Creates a new EditKata screen with existing kata data.
    ///
    /// # Arguments
    ///
    /// * `kata` - The kata to edit
    /// * `tags` - Current tags for the kata
    /// * `dependencies` - Current dependency names
    /// * `available_katas` - List of all kata names that can be dependencies
    /// * `exercises_dir` - Path to exercises directory for file operations
    pub fn new(
        kata: &Kata,
        tags: Vec<String>,
        dependencies: Vec<String>,
        available_katas: Vec<String>,
        exercises_dir: std::path::PathBuf,
    ) -> Self {
        let original_slug = kata.name.clone();

        // Display tags as comma-separated in category field
        // If no tags, fall back to the category from kata metadata
        let category_display = if !tags.is_empty() {
            tags.join(", ")
        } else {
            kata.category.clone()
        };

        Self {
            kata_id: kata.id,
            original_name: kata.name.clone(),
            original_slug: original_slug.clone(),
            name_input: kata.name.clone(),
            category_input: category_display,
            description_input: kata.description.clone(),
            difficulty: kata.base_difficulty as u8,
            selected_dependencies: dependencies,
            current_field: FormField::Name,
            cursor_position: kata.name.len(),
            available_katas,
            dependency_scroll_offset: 0,
            dependency_selected_index: 0,
            validation_errors: HashMap::new(),
            showing_confirmation: false,
            confirmation_slug: String::new(),
            confirmation_error: None,
            exercises_dir,
        }
    }

    /// Renders the edit kata screen.
    pub fn render(&self, frame: &mut Frame) {
        if self.showing_confirmation {
            self.render_confirmation(frame);
        } else {
            self.render_form(frame);
        }
    }

    fn render_form(&self, frame: &mut Frame) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3), // Header
                Constraint::Min(10),   // Form
                Constraint::Length(4), // Footer with editor shortcuts
            ])
            .split(frame.size());

        self.render_header(frame, chunks[0]);
        self.render_form_fields(frame, chunks[1]);
        self.render_footer(frame, chunks[2]);
    }

    fn render_header(&self, frame: &mut Frame, area: Rect) {
        let header = Paragraph::new(format!("Edit Kata: {}", self.original_name))
            .block(Block::default().borders(Borders::ALL));
        frame.render_widget(header, area);
    }

    fn render_form_fields(&self, frame: &mut Frame, area: Rect) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3), // Name
                Constraint::Length(3), // Category
                Constraint::Length(5), // Description (multiline)
                Constraint::Length(3), // Difficulty
                Constraint::Min(5),    // Dependencies (scrollable list)
            ])
            .split(area);

        // Name field
        self.render_text_field(
            frame,
            chunks[0],
            FormField::Name,
            &self.name_input,
            self.cursor_position,
        );

        // Category field
        self.render_text_field(
            frame,
            chunks[1],
            FormField::Category,
            &self.category_input,
            self.cursor_position,
        );

        // Description field
        self.render_text_field(
            frame,
            chunks[2],
            FormField::Description,
            &self.description_input,
            self.cursor_position,
        );

        // Difficulty field
        self.render_difficulty_field(frame, chunks[3]);

        // Dependencies field
        self.render_dependencies_field(frame, chunks[4]);
    }

    fn render_text_field(
        &self,
        frame: &mut Frame,
        area: Rect,
        field: FormField,
        text: &str,
        cursor_pos: usize,
    ) {
        let is_active = self.current_field == field;
        let border_style = if is_active {
            Style::default().fg(Color::Yellow)
        } else {
            Style::default()
        };

        let display_text = if is_active && field != FormField::Description {
            // Show cursor for single-line fields
            let mut display = text.to_string();
            if cursor_pos <= display.len() {
                display.insert(cursor_pos, '|');
            }
            display
        } else {
            text.to_string()
        };

        let error = self.validation_errors.get(&field);
        let title = if let Some(err) = error {
            format!("{} - Error: {}", field.label(), err)
        } else {
            field.label().to_string()
        };

        let title_style = if error.is_some() {
            Style::default().fg(Color::Red)
        } else {
            Style::default()
        };

        let paragraph = Paragraph::new(display_text)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(title)
                    .title_style(title_style)
                    .border_style(border_style),
            )
            .wrap(Wrap { trim: false });

        frame.render_widget(paragraph, area);
    }

    fn render_difficulty_field(&self, frame: &mut Frame, area: Rect) {
        let is_active = self.current_field == FormField::Difficulty;
        let border_style = if is_active {
            Style::default().fg(Color::Yellow)
        } else {
            Style::default()
        };

        let stars = "★".repeat(self.difficulty as usize);
        let empty = "☆".repeat((5 - self.difficulty) as usize);
        let display = format!("{}{} ({}/5)", stars, empty, self.difficulty);

        let help_text = if is_active {
            " [Use ↑/↓ to adjust]"
        } else {
            ""
        };

        let paragraph = Paragraph::new(format!("{}{}", display, help_text)).block(
            Block::default()
                .borders(Borders::ALL)
                .title("Difficulty")
                .border_style(border_style),
        );

        frame.render_widget(paragraph, area);
    }

    fn render_dependencies_field(&self, frame: &mut Frame, area: Rect) {
        let is_active = self.current_field == FormField::Dependencies;
        let border_style = if is_active {
            Style::default().fg(Color::Yellow)
        } else {
            Style::default()
        };

        let error = self.validation_errors.get(&FormField::Dependencies);
        let title = if let Some(err) = error {
            format!("Dependencies - Error: {}", err)
        } else if is_active {
            "Dependencies [j/k to navigate, Space to toggle]".to_string()
        } else {
            "Dependencies".to_string()
        };

        let title_style = if error.is_some() {
            Style::default().fg(Color::Red)
        } else {
            Style::default()
        };

        // Filter out the current kata from available dependencies
        let available_deps: Vec<_> = self
            .available_katas
            .iter()
            .filter(|k| *k != &self.original_slug)
            .collect();

        if available_deps.is_empty() {
            let paragraph = Paragraph::new("No katas available for dependencies").block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(title)
                    .title_style(title_style)
                    .border_style(border_style),
            );
            frame.render_widget(paragraph, area);
        } else {
            let items: Vec<ListItem> = available_deps
                .iter()
                .enumerate()
                .map(|(i, kata)| {
                    let is_selected = self.selected_dependencies.contains(*kata);
                    let checkbox = if is_selected { "[✓]" } else { "[ ]" };
                    let style = if is_active && i == self.dependency_selected_index {
                        Style::default().fg(Color::Yellow)
                    } else if is_selected {
                        Style::default().fg(Color::Green)
                    } else {
                        Style::default()
                    };
                    ListItem::new(format!("{} {}", checkbox, kata)).style(style)
                })
                .collect();

            let list = List::new(items).block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(title)
                    .title_style(title_style)
                    .border_style(border_style),
            );

            frame.render_widget(list, area);
        }
    }

    fn render_footer(&self, frame: &mut Frame, area: Rect) {
        let footer_lines = if self.current_field == FormField::Dependencies {
            vec![
                Line::from(vec![
                    Span::raw("[Tab] Next  "),
                    Span::raw("[Space] Toggle  "),
                    Span::raw("[j/k] Navigate  "),
                    Span::raw("[Enter] Submit  "),
                    Span::raw("[Esc] Cancel"),
                ]),
                Line::from(vec![
                    Span::styled("[F2] ", Style::default().fg(Color::Cyan)),
                    Span::raw("Edit template.py  "),
                    Span::styled("[F3] ", Style::default().fg(Color::Cyan)),
                    Span::raw("Edit reference.py  "),
                    Span::styled("[F4] ", Style::default().fg(Color::Cyan)),
                    Span::raw("Edit test_kata.py"),
                ]),
            ]
        } else if self.current_field == FormField::Difficulty {
            vec![
                Line::from(vec![
                    Span::raw("[Tab] Next  "),
                    Span::raw("[↑/↓] Adjust  "),
                    Span::raw("[Enter] Submit  "),
                    Span::raw("[Esc] Cancel"),
                ]),
                Line::from(vec![
                    Span::styled("[F2] ", Style::default().fg(Color::Cyan)),
                    Span::raw("Edit template.py  "),
                    Span::styled("[F3] ", Style::default().fg(Color::Cyan)),
                    Span::raw("Edit reference.py  "),
                    Span::styled("[F4] ", Style::default().fg(Color::Cyan)),
                    Span::raw("Edit test_kata.py"),
                ]),
            ]
        } else {
            vec![
                Line::from(vec![
                    Span::raw("[Tab] Next  "),
                    Span::raw("[Type] Edit  "),
                    Span::raw("[Enter] Submit  "),
                    Span::raw("[Esc] Cancel"),
                ]),
                Line::from(vec![
                    Span::styled("[F2] ", Style::default().fg(Color::Cyan)),
                    Span::raw("Edit template.py  "),
                    Span::styled("[F3] ", Style::default().fg(Color::Cyan)),
                    Span::raw("Edit reference.py  "),
                    Span::styled("[F4] ", Style::default().fg(Color::Cyan)),
                    Span::raw("Edit test_kata.py"),
                ]),
            ]
        };

        let footer = Paragraph::new(footer_lines).block(Block::default().borders(Borders::ALL));
        frame.render_widget(footer, area);
    }

    fn render_confirmation(&self, frame: &mut Frame) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(12), // Confirmation dialog
                Constraint::Min(0),     // Spacer
            ])
            .split(frame.size());

        // Center the dialog
        let dialog_chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Percentage(25),
                Constraint::Percentage(50),
                Constraint::Percentage(25),
            ])
            .split(chunks[0]);

        let dialog_area = dialog_chunks[1];

        if let Some(error) = &self.confirmation_error {
            // Show error message
            let error_text = format!(
                "Cannot update kata:\n\n{}\n\nPress Enter to return to form.",
                error
            );
            let paragraph = Paragraph::new(error_text)
                .style(Style::default().fg(Color::Red))
                .block(
                    Block::default()
                        .borders(Borders::ALL)
                        .title("Error")
                        .border_style(Style::default().fg(Color::Red)),
                )
                .wrap(Wrap { trim: false });

            frame.render_widget(paragraph, dialog_area);
        } else {
            // Show confirmation
            let name_changed = self.confirmation_slug != self.original_slug;
            let confirmation_text = if name_changed {
                format!(
                    "Update kata and rename directory:\n  {} → {}\n\nFiles to update:\n  - manifest.toml\n  - database entry\n\nScheduling state preserved.\n\nConfirm? [y/n]",
                    self.original_slug, self.confirmation_slug
                )
            } else {
                format!(
                    "Update kata metadata:\n  {}\n\nFiles to update:\n  - manifest.toml\n  - database entry\n\nScheduling state preserved.\n\nConfirm? [y/n]",
                    self.confirmation_slug
                )
            };

            let paragraph = Paragraph::new(confirmation_text)
                .block(
                    Block::default()
                        .borders(Borders::ALL)
                        .title("Confirm Update")
                        .border_style(Style::default().fg(Color::Yellow)),
                )
                .wrap(Wrap { trim: false });

            frame.render_widget(paragraph, dialog_area);
        }
    }

    /// Handles keyboard input and returns the appropriate action.
    pub fn handle_input(&mut self, code: KeyCode, exercises_dir: &Path) -> EditKataAction {
        if self.showing_confirmation {
            return self.handle_confirmation_input(code, exercises_dir);
        }

        match code {
            KeyCode::F(2) => {
                // Open template.py in external editor
                let template_path = self.exercises_dir.join(&self.original_slug).join("template.py");
                EditKataAction::OpenEditor {
                    file_path: template_path,
                }
            }
            KeyCode::F(3) => {
                // Open reference.py in external editor
                let reference_path = self.exercises_dir.join(&self.original_slug).join("reference.py");
                EditKataAction::OpenEditor {
                    file_path: reference_path,
                }
            }
            KeyCode::F(4) => {
                // Open test_kata.py in external editor
                let test_path = self.exercises_dir.join(&self.original_slug).join("test_kata.py");
                EditKataAction::OpenEditor {
                    file_path: test_path,
                }
            }
            KeyCode::Tab => {
                self.current_field = self.current_field.next();
                self.cursor_position = match self.current_field {
                    FormField::Name => self.name_input.len(),
                    FormField::Category => self.category_input.len(),
                    FormField::Description => self.description_input.len(),
                    _ => 0,
                };
                EditKataAction::None
            }
            KeyCode::BackTab => {
                self.current_field = self.current_field.prev();
                self.cursor_position = match self.current_field {
                    FormField::Name => self.name_input.len(),
                    FormField::Category => self.category_input.len(),
                    FormField::Description => self.description_input.len(),
                    _ => 0,
                };
                EditKataAction::None
            }
            KeyCode::Enter => {
                // Validate and show confirmation
                self.validate_form(exercises_dir);
                if self.validation_errors.is_empty() {
                    self.showing_confirmation = true;
                    self.confirmation_slug = slugify_kata_name(&self.name_input);
                    self.confirmation_error = None;

                    // Re-validate at confirmation time (defense-in-depth)
                    if self.confirmation_slug != self.original_slug {
                        // Name changed - validate new name doesn't conflict
                        if let Err(e) = validate_kata_name(&self.confirmation_slug, exercises_dir) {
                            self.confirmation_error = Some(e.to_string());
                        }
                    }

                    if self.confirmation_error.is_none() {
                        if let Err(e) =
                            validate_dependencies(&self.selected_dependencies, exercises_dir)
                        {
                            self.confirmation_error = Some(e.to_string());
                        }
                    }
                }
                EditKataAction::None
            }
            KeyCode::Esc => EditKataAction::Cancel,
            _ => {
                match self.current_field {
                    FormField::Name => {
                        Self::handle_text_input_static(
                            code,
                            &mut self.name_input,
                            &mut self.cursor_position,
                        );
                        self.validation_errors.remove(&FormField::Name);
                    }
                    FormField::Category => {
                        Self::handle_text_input_static(
                            code,
                            &mut self.category_input,
                            &mut self.cursor_position,
                        );
                        self.validation_errors.remove(&FormField::Category);
                    }
                    FormField::Description => {
                        Self::handle_text_input_static(
                            code,
                            &mut self.description_input,
                            &mut self.cursor_position,
                        );
                        self.validation_errors.remove(&FormField::Description);
                    }
                    FormField::Difficulty => self.handle_difficulty_input(code),
                    FormField::Dependencies => self.handle_dependencies_input(code),
                }
                EditKataAction::None
            }
        }
    }

    fn handle_confirmation_input(
        &mut self,
        code: KeyCode,
        _exercises_dir: &Path,
    ) -> EditKataAction {
        if self.confirmation_error.is_some() {
            // If there's an error, any key returns to form
            match code {
                KeyCode::Enter | KeyCode::Esc => {
                    self.showing_confirmation = false;
                    self.confirmation_error = None;
                }
                _ => {}
            }
            return EditKataAction::None;
        }

        match code {
            KeyCode::Char('y') | KeyCode::Char('Y') => {
                // Submit the form
                let form_data = KataFormData {
                    name: self.name_input.clone(),
                    category: self.category_input.clone(),
                    description: self.description_input.clone(),
                    difficulty: self.difficulty,
                    dependencies: self.selected_dependencies.clone(),
                };
                EditKataAction::Submit {
                    kata_id: self.kata_id,
                    original_slug: self.original_slug.clone(),
                    form_data,
                    new_slug: self.confirmation_slug.clone(),
                }
            }
            KeyCode::Char('n') | KeyCode::Char('N') | KeyCode::Esc => {
                // Return to form
                self.showing_confirmation = false;
                EditKataAction::None
            }
            _ => EditKataAction::None,
        }
    }

    fn handle_text_input_static(code: KeyCode, buffer: &mut String, cursor_position: &mut usize) {
        match code {
            KeyCode::Char(c) => {
                buffer.insert(*cursor_position, c);
                *cursor_position += 1;
            }
            KeyCode::Backspace => {
                if *cursor_position > 0 {
                    buffer.remove(*cursor_position - 1);
                    *cursor_position -= 1;
                }
            }
            KeyCode::Delete => {
                if *cursor_position < buffer.len() {
                    buffer.remove(*cursor_position);
                }
            }
            KeyCode::Left => {
                if *cursor_position > 0 {
                    *cursor_position -= 1;
                }
            }
            KeyCode::Right => {
                if *cursor_position < buffer.len() {
                    *cursor_position += 1;
                }
            }
            KeyCode::Home => {
                *cursor_position = 0;
            }
            KeyCode::End => {
                *cursor_position = buffer.len();
            }
            _ => {}
        }
    }

    fn handle_difficulty_input(&mut self, code: KeyCode) {
        match code {
            KeyCode::Up => {
                if self.difficulty < 5 {
                    self.difficulty += 1;
                }
            }
            KeyCode::Down => {
                if self.difficulty > 1 {
                    self.difficulty -= 1;
                }
            }
            _ => {}
        }
    }

    fn handle_dependencies_input(&mut self, code: KeyCode) {
        // Filter out the current kata from available dependencies
        let available_deps: Vec<_> = self
            .available_katas
            .iter()
            .filter(|k| *k != &self.original_slug)
            .collect();

        if available_deps.is_empty() {
            return;
        }

        match code {
            KeyCode::Char('j') | KeyCode::Down => {
                if self.dependency_selected_index < available_deps.len() - 1 {
                    self.dependency_selected_index += 1;
                }
            }
            KeyCode::Char('k') | KeyCode::Up => {
                if self.dependency_selected_index > 0 {
                    self.dependency_selected_index -= 1;
                }
            }
            KeyCode::Char(' ') => {
                // Toggle selection
                let kata = available_deps[self.dependency_selected_index].to_string();
                if let Some(pos) = self.selected_dependencies.iter().position(|k| k == &kata) {
                    self.selected_dependencies.remove(pos);
                } else {
                    self.selected_dependencies.push(kata);
                }
                self.validation_errors.remove(&FormField::Dependencies);
            }
            _ => {}
        }
    }

    fn validate_form(&mut self, exercises_dir: &Path) {
        self.validation_errors.clear();

        // Validate name
        if self.name_input.trim().is_empty() {
            self.validation_errors
                .insert(FormField::Name, "Name cannot be empty".to_string());
        } else {
            let slug = slugify_kata_name(&self.name_input);
            if slug.is_empty() {
                self.validation_errors.insert(
                    FormField::Name,
                    "Name must contain at least one alphanumeric character".to_string(),
                );
            }
        }

        // Validate category
        if self.category_input.trim().is_empty() {
            self.validation_errors
                .insert(FormField::Category, "Category cannot be empty".to_string());
        }

        // Validate description
        if self.description_input.trim().is_empty() {
            self.validation_errors.insert(
                FormField::Description,
                "Description cannot be empty".to_string(),
            );
        }

        // Validate dependencies (check they exist)
        if let Err(e) = validate_dependencies(&self.selected_dependencies, exercises_dir) {
            self.validation_errors
                .insert(FormField::Dependencies, e.to_string());
        }
    }
}
