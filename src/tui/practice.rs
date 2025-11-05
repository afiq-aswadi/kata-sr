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
                if let Err(err) = self.edit_and_run_tests(tx_clone) {
                    let _ =
                        event_tx.send(AppEvent::TestComplete(TestResults::error(err.to_string())));
                    self.status = PracticeStatus::ShowingDescription;
                }
                Ok(PracticeAction::None)
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
            crossterm::terminal::disable_raw_mode()
                .context("failed to disable raw mode before launching editor")?;

            let status_result = Command::new("nvim").arg(&self.template_path).status();

            crossterm::terminal::enable_raw_mode()
                .context("failed to re-enable raw mode after exiting editor")?;

            let editor_status = status_result.context("failed to launch nvim")?;

            if !editor_status.success() {
                anyhow::bail!(
                    "nvim exited with non-zero status (code {:?})",
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
}
