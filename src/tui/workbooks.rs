use crate::core::kata_loader::{load_available_katas, AvailableKata};
use crate::core::workbook::load_workbooks;
use crate::core::workbook::Workbook;
use anyhow::Result;
use crossterm::event::KeyCode;
use ratatui::{
    layout::{Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, ListState, Paragraph, Wrap},
    Frame,
};
use std::collections::HashMap;
use std::path::PathBuf;

pub struct WorkbookScreen {
    workbooks: Vec<Workbook>,
    available_by_name: HashMap<String, AvailableKata>,
    selected: usize,
    list_state: ListState,
}

pub enum WorkbookAction {
    None,
    Back,
    OpenHtml(PathBuf),
    AddExercises {
        kata_names: Vec<String>,
        workbook_title: String,
    },
    PreviewFirst(AvailableKata),
}

impl WorkbookScreen {
    pub fn load() -> Result<Self> {
        let workbooks = load_workbooks()?;
        let available = load_available_katas()?;
        let mut available_by_name = HashMap::new();
        for kata in available {
            available_by_name.insert(kata.name.clone(), kata);
        }

        let mut list_state = ListState::default();
        if !workbooks.is_empty() {
            list_state.select(Some(0));
        }

        Ok(Self {
            workbooks,
            available_by_name,
            selected: 0,
            list_state,
        })
    }

    pub fn render(&mut self, frame: &mut Frame) {
        let layout = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Min(10), Constraint::Length(3)])
            .split(frame.size());

        let columns = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(40), Constraint::Percentage(60)])
            .split(layout[0]);

        self.render_list(frame, columns[0]);
        self.render_detail(frame, columns[1]);
        self.render_footer(frame, layout[1]);
    }

    pub fn handle_input(&mut self, code: KeyCode) -> WorkbookAction {
        if self.workbooks.is_empty() {
            return match code {
                KeyCode::Esc => WorkbookAction::Back,
                _ => WorkbookAction::None,
            };
        }

        match code {
            KeyCode::Char('j') | KeyCode::Down => {
                if self.selected + 1 < self.workbooks.len() {
                    self.selected += 1;
                }
                self.list_state.select(Some(self.selected));
                WorkbookAction::None
            }
            KeyCode::Char('k') | KeyCode::Up => {
                if self.selected > 0 {
                    self.selected -= 1;
                }
                self.list_state.select(Some(self.selected));
                WorkbookAction::None
            }
            KeyCode::Char('o') | KeyCode::Enter => {
                WorkbookAction::OpenHtml(self.workbooks[self.selected].html_path.clone())
            }
            KeyCode::Char('a') => WorkbookAction::AddExercises {
                kata_names: self.kata_names_for_selected(),
                workbook_title: self.workbooks[self.selected].meta.title.clone(),
            },
            KeyCode::Char('p') => {
                if let Some(first) = self.workbooks[self.selected]
                    .exercises
                    .first()
                    .and_then(|ex| self.available_by_name.get(&ex.kata))
                {
                    WorkbookAction::PreviewFirst(first.clone())
                } else {
                    WorkbookAction::None
                }
            }
            KeyCode::Esc => WorkbookAction::Back,
            _ => WorkbookAction::None,
        }
    }

    fn render_list(&mut self, frame: &mut Frame, area: ratatui::layout::Rect) {
        let items: Vec<ListItem> = if self.workbooks.is_empty() {
            vec![ListItem::new("No workbooks found")]
        } else {
            self.workbooks
                .iter()
                .map(|w| {
                    let count = w.exercises.len();
                    let line = Line::from(vec![
                        Span::styled(
                            &w.meta.title,
                            Style::default()
                                .fg(Color::Cyan)
                                .add_modifier(Modifier::BOLD),
                        ),
                        Span::raw(format!("  •  {} exercises", count)),
                    ]);
                    ListItem::new(line)
                })
                .collect()
        };

        let block = Block::default().borders(Borders::ALL).title("Workbooks");
        let list = List::new(items)
            .block(block)
            .highlight_style(
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD | Modifier::UNDERLINED),
            )
            .highlight_symbol("➜ ");

        frame.render_stateful_widget(list, area, &mut self.list_state);
    }

    fn render_detail(&self, frame: &mut Frame, area: ratatui::layout::Rect) {
        let mut lines = Vec::new();

        if let Some(current) = self.workbooks.get(self.selected) {
            lines.push(Line::from(vec![Span::styled(
                &current.meta.title,
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            )]));
            lines.push(Line::from(Span::raw("")));
            lines.push(Line::from(Span::styled(
                "Summary",
                Style::default()
                    .fg(Color::Gray)
                    .add_modifier(Modifier::BOLD),
            )));
            if current.meta.summary.is_empty() {
                lines.push(Line::from(Span::raw("No summary provided.")));
            } else {
                lines.push(Line::from(Span::raw(current.meta.summary.clone())));
            }

            lines.push(Line::from(Span::raw("")));
            lines.push(Line::from(Span::styled(
                "Learning goals",
                Style::default()
                    .fg(Color::Gray)
                    .add_modifier(Modifier::BOLD),
            )));
            if current.meta.learning_goals.is_empty() {
                lines.push(Line::from(Span::raw("None listed.")));
            } else {
                for goal in &current.meta.learning_goals {
                    lines.push(Line::from(format!("• {}", goal)));
                }
            }

            lines.push(Line::from(Span::raw("")));
            lines.push(Line::from(Span::styled(
                "Exercises",
                Style::default()
                    .fg(Color::Gray)
                    .add_modifier(Modifier::BOLD),
            )));
            for ex in &current.exercises {
                lines.push(Line::from(format!("• {}  ({})", ex.title, ex.kata)));
                lines.push(Line::from(Span::raw(format!(
                    "  {}",
                    truncate(&ex.objective, 80)
                ))));
            }

            if !current.meta.resources.is_empty() {
                lines.push(Line::from(Span::raw("")));
                lines.push(Line::from(Span::styled(
                    "Resources",
                    Style::default()
                        .fg(Color::Gray)
                        .add_modifier(Modifier::BOLD),
                )));
                for res in &current.meta.resources {
                    lines.push(Line::from(format!("• {} ({})", res.title, res.url)));
                }
            }
        } else {
            lines.push(Line::from("No workbook selected."));
        }

        let block = Block::default().borders(Borders::ALL).title("Details");
        let paragraph = Paragraph::new(lines).block(block).wrap(Wrap { trim: true });

        frame.render_widget(paragraph, area);
    }

    fn render_footer(&self, frame: &mut Frame, area: ratatui::layout::Rect) {
        let footer = if self.workbooks.is_empty() {
            "[Esc] Back".to_string()
        } else {
            "[j/k] Navigate  [o/Enter] Open workbook page  [a] Add all exercises to deck  [p] Preview first exercise  [Esc] Back".to_string()
        };

        let paragraph = Paragraph::new(footer)
            .block(Block::default().borders(Borders::ALL))
            .alignment(ratatui::layout::Alignment::Center);
        frame.render_widget(paragraph, area);
    }

    fn kata_names_for_selected(&self) -> Vec<String> {
        if let Some(current) = self.workbooks.get(self.selected) {
            current.exercises.iter().map(|ex| ex.kata.clone()).collect()
        } else {
            Vec::new()
        }
    }
}

fn truncate(input: &str, max_len: usize) -> String {
    if input.chars().count() <= max_len {
        return input.to_string();
    }
    let mut out = String::new();
    for (idx, ch) in input.chars().enumerate() {
        if idx >= max_len {
            break;
        }
        out.push(ch);
    }
    out.push_str("...");
    out
}
