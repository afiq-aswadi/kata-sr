use crate::db::repo::Kata;
use crate::runner::python_runner::TestResults;
use crate::tui::app::AppEvent;
use anyhow::Context;
use crossterm::event::KeyCode;
use ratatui::{
    layout::{Constraint, Direction, Layout},
    widgets::{Block, Borders, Paragraph, Wrap},
    Frame,
};
use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::sync::mpsc::Sender;

pub struct PracticeScreen {
    kata: Kata,
    template_path: PathBuf,
    status: PracticeStatus,
}

enum PracticeStatus {
    ShowingDescription,
    EditingInProgress,
    TestsRunning,
}

impl PracticeScreen {
    pub fn new(kata: Kata) -> anyhow::Result<Self> {
        let template_path = PathBuf::from(format!("/tmp/kata_{}.py", kata.id));

        let katas_root = std::env::var("KATA_SR_KATAS_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("katas"));
        let kata_dir = katas_root.join("exercises").join(&kata.name);
        let template_source = kata_dir.join("template.py");
        fs::copy(&template_source, &template_path).with_context(|| {
            format!(
                "failed to copy kata template from {}",
                template_source.display()
            )
        })?;

        Ok(Self {
            kata,
            template_path,
            status: PracticeStatus::ShowingDescription,
        })
    }

    /// Creates a practice screen for retry scenarios, preserving existing edits.
    ///
    /// Unlike `new()`, this method does NOT copy the template file, allowing the user
    /// to retry tests with their previous edits intact. This prevents data loss when
    /// users press 'r' to retry after test failures.
    ///
    /// # Arguments
    ///
    /// * `kata` - The kata to practice
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use kata_sr::db::repo::Kata;
    /// # use kata_sr::tui::practice::PracticeScreen;
    /// # let kata = Kata {
    /// #     id: 1,
    /// #     name: "test".to_string(),
    /// #     category: "test".to_string(),
    /// #     description: "test".to_string(),
    /// #     base_difficulty: 1,
    /// #     current_difficulty: 1.0,
    /// #     parent_kata_id: None,
    /// #     variation_params: None,
    /// #     next_review_at: Some(chrono::Utc::now()),
    /// #     last_reviewed_at: None,
    /// #     current_ease_factor: 2.5,
    /// #     current_interval_days: 1,
    /// #     current_repetition_count: 0,
    /// #     created_at: chrono::Utc::now(),
    /// # };
    /// let practice_screen = PracticeScreen::new_retry(kata)?;
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn new_retry(kata: Kata) -> anyhow::Result<Self> {
        let template_path = PathBuf::from(format!("/tmp/kata_{}.py", kata.id));

        // verify the file exists (it should, since we're retrying)
        if !template_path.exists() {
            anyhow::bail!(
                "expected existing practice file at {}, but it doesn't exist",
                template_path.display()
            );
        }

        Ok(Self {
            kata,
            template_path,
            status: PracticeStatus::ShowingDescription,
        })
    }

    pub fn render(&self, frame: &mut Frame) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3),
                Constraint::Min(10),
                Constraint::Length(5),
            ])
            .split(frame.size());

        let title = Paragraph::new(format!("Kata: {}", self.kata.name))
            .block(Block::default().borders(Borders::ALL));
        frame.render_widget(title, chunks[0]);

        let desc = Paragraph::new(self.kata.description.clone())
            .wrap(Wrap { trim: true })
            .block(Block::default().borders(Borders::ALL).title("Description"));
        frame.render_widget(desc, chunks[1]);

        let instructions = match self.status {
            PracticeStatus::ShowingDescription => {
                "[e] Edit & run tests  [t] Run tests  [Esc] Back to dashboard"
            }
            PracticeStatus::EditingInProgress => "Editing in nvim... (save and exit to run tests)",
            PracticeStatus::TestsRunning => "Running tests...",
        };
        let inst = Paragraph::new(instructions)
            .block(Block::default().borders(Borders::ALL).title("Actions"));
        frame.render_widget(inst, chunks[2]);
    }

    pub fn handle_input(
        &mut self,
        code: KeyCode,
        event_tx: Sender<AppEvent>,
    ) -> anyhow::Result<PracticeAction> {
        match code {
            KeyCode::Char('e') => {
                let tx_clone = event_tx.clone();
                match self.edit_and_run_tests(tx_clone) {
                    Ok(_) => Ok(PracticeAction::EditorExited),
                    Err(err) => {
                        let _ = event_tx
                            .send(AppEvent::TestComplete(TestResults::error(err.to_string())));
                        self.status = PracticeStatus::ShowingDescription;
                        Ok(PracticeAction::EditorExited)
                    }
                }
            }
            KeyCode::Char('t') => {
                self.run_tests(event_tx);
                Ok(PracticeAction::None)
            }
            KeyCode::Esc => Ok(PracticeAction::BackToDashboard),
            _ => Ok(PracticeAction::None),
        }
    }

    fn edit_and_run_tests(&mut self, event_tx: Sender<AppEvent>) -> anyhow::Result<()> {
        self.launch_editor()?;
        self.run_tests(event_tx);
        Ok(())
    }

    fn launch_editor(&mut self) -> anyhow::Result<()> {
        self.status = PracticeStatus::EditingInProgress;

        let result = (|| -> anyhow::Result<()> {
            // Determine which editor to use (respects EDITOR env var)
            let editor = std::env::var("EDITOR").unwrap_or_else(|_| "nvim".to_string());

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

            // Launch editor and wait for it to complete
            let status_result = Command::new(&editor).arg(&self.template_path).status();

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
                status_result.with_context(|| format!("failed to launch {}", editor))?;

            if !editor_status.success() {
                anyhow::bail!(
                    "{} exited with non-zero status (code {:?})",
                    editor,
                    editor_status.code()
                );
            }

            Ok(())
        })();

        self.status = PracticeStatus::ShowingDescription;
        result
    }

    fn run_tests(&mut self, event_tx: Sender<AppEvent>) {
        self.status = PracticeStatus::TestsRunning;

        let kata_name = self.kata.name.clone();
        let template_path = self.template_path.clone();

        std::thread::spawn(move || {
            let results =
                crate::runner::python_runner::run_python_tests(&kata_name, &template_path);
            event_tx.send(AppEvent::TestComplete(results)).unwrap();
        });
    }
}

pub enum PracticeAction {
    None,
    BackToDashboard,
    EditorExited, // Signal that terminal needs refresh after editor
}
