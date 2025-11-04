# Agent 3: Rust TUI Application

## Mission

Build the interactive terminal UI using ratatui. Create dashboard, practice, and results screens with keyboard navigation. Integrate Python runner for test execution. Make it responsive and intuitive.

## Dependencies

**Requires Agent 1 to be complete:**
- KataRepository interface
- SM2State and QualityRating
- DependencyGraph
- PythonEnv setup

You can start designing UI layouts and screen structure while Agent 1 works, but integration requires their completion.

## What You're Building

### 1. Main TUI Application
Event loop handling keyboard input and async test results

### 2. Dashboard Screen
Shows katas due today, locked katas, stats, and streak

### 3. Practice Screen
Display kata description, spawn editor, trigger test runs

### 4. Results Screen
Show test output with pass/fail, collect quality rating (0-3)

### 5. Python Runner Integration
Spawn pytest in background thread, parse JSON results

## Detailed Specifications

### Update Cargo.toml

```toml
[dependencies]
# ... existing from Agent 1 ...
ratatui = "0.26"
crossterm = "0.27"
tokio = { version = "1.36", features = ["sync", "rt"] }
```

### Main Event Loop

```rust
// src/tui/app.rs

use std::sync::mpsc::{channel, Receiver, Sender};
use crossterm::event::{self, Event, KeyCode};
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use std::io;

pub enum AppEvent {
    Input(KeyCode),
    TestComplete(TestResults),
    Quit,
}

pub enum Screen {
    Dashboard,
    Practice(Kata),
    Results(Kata, TestResults),
}

pub struct App {
    pub current_screen: Screen,
    pub repo: KataRepository,
    pub event_tx: Sender<AppEvent>,
    pub event_rx: Receiver<AppEvent>,
}

impl App {
    pub fn new(repo: KataRepository) -> Self {
        let (tx, rx) = channel();
        Self {
            current_screen: Screen::Dashboard,
            repo,
            event_tx: tx,
            event_rx: rx,
        }
    }

    pub fn run(&mut self) -> anyhow::Result<()> {
        // Setup terminal
        crossterm::terminal::enable_raw_mode()?;
        let mut stdout = io::stdout();
        crossterm::execute!(
            stdout,
            crossterm::terminal::EnterAlternateScreen,
            crossterm::event::EnableMouseCapture
        )?;
        let backend = CrosstermBackend::new(stdout);
        let mut terminal = Terminal::new(backend)?;

        let result = self.event_loop(&mut terminal);

        // Cleanup
        crossterm::terminal::disable_raw_mode()?;
        crossterm::execute!(
            terminal.backend_mut(),
            crossterm::terminal::LeaveAlternateScreen,
            crossterm::event::DisableMouseCapture
        )?;
        terminal.show_cursor()?;

        result
    }

    fn event_loop(&mut self, terminal: &mut Terminal<CrosstermBackend<io::Stdout>>) -> anyhow::Result<()> {
        loop {
            // Render current screen
            terminal.draw(|f| self.render(f))?;

            // Handle events
            if event::poll(std::time::Duration::from_millis(100))? {
                if let Event::Key(key) = event::read()? {
                    match key.code {
                        KeyCode::Char('q') => return Ok(()),
                        code => self.handle_input(code)?,
                    }
                }
            }

            // Check for async events (test completion)
            if let Ok(event) = self.event_rx.try_recv() {
                self.handle_event(event)?;
            }
        }
    }

    fn render(&self, frame: &mut ratatui::Frame) {
        match &self.current_screen {
            Screen::Dashboard => self.render_dashboard(frame),
            Screen::Practice(kata) => self.render_practice(frame, kata),
            Screen::Results(kata, results) => self.render_results(frame, kata, results),
        }
    }

    fn handle_input(&mut self, code: KeyCode) -> anyhow::Result<()> {
        match &mut self.current_screen {
            Screen::Dashboard => self.handle_dashboard_input(code),
            Screen::Practice(_) => self.handle_practice_input(code),
            Screen::Results(_, _) => self.handle_results_input(code),
        }
    }

    fn handle_event(&mut self, event: AppEvent) -> anyhow::Result<()> {
        match event {
            AppEvent::TestComplete(results) => {
                if let Screen::Practice(kata) = &self.current_screen {
                    self.current_screen = Screen::Results(kata.clone(), results);
                }
            }
            AppEvent::Quit => return Err(anyhow::anyhow!("Quit")),
            _ => {}
        }
        Ok(())
    }

    // Implement render_* and handle_*_input methods below
}

pub struct TestResults {
    pub passed: bool,
    pub num_passed: i32,
    pub num_failed: i32,
    pub num_skipped: i32,
    pub duration_ms: i64,
    pub results: Vec<TestResult>,
}

pub struct TestResult {
    pub test_name: String,
    pub status: String,  // "passed", "failed", "skipped"
    pub output: String,
}
```

### Dashboard Screen

```rust
// src/tui/dashboard.rs

use ratatui::{
    layout::{Constraint, Direction, Layout},
    widgets::{Block, Borders, List, ListItem, Paragraph},
    Frame,
};
use crate::db::repo::KataRepository;

pub struct Dashboard {
    pub katas_due: Vec<Kata>,
    pub locked_katas: Vec<(Kata, String)>,  // (kata, reason)
    pub selected_index: usize,
    pub stats: DashboardStats,
}

pub struct DashboardStats {
    pub streak_days: i32,
    pub total_reviews_today: i32,
    pub success_rate_7d: f64,
}

impl Dashboard {
    pub fn load(repo: &KataRepository) -> anyhow::Result<Self> {
        // Query katas due today
        let now = chrono::Utc::now();
        let katas_due = repo.get_katas_due(now)?;

        // Query locked katas (dependencies not met)
        let dep_graph = repo.load_dependency_graph()?;
        let success_counts = repo.get_success_counts()?;
        let all_katas = repo.get_all_katas()?;

        let locked_katas = all_katas
            .into_iter()
            .filter_map(|k| {
                if !dep_graph.is_unlocked(k.id, &success_counts) {
                    let blocking = dep_graph.get_blocking_dependencies(k.id, &success_counts);
                    let reason = format!(
                        "Requires: {} (need {} more)",
                        blocking[0].0, blocking[0].1 - blocking[0].2
                    );
                    Some((k, reason))
                } else {
                    None
                }
            })
            .collect();

        // Load stats
        let stats = DashboardStats {
            streak_days: repo.get_current_streak()?,
            total_reviews_today: repo.get_reviews_count_today()?,
            success_rate_7d: repo.get_success_rate_last_n_days(7)?,
        };

        Ok(Self {
            katas_due,
            locked_katas,
            selected_index: 0,
            stats,
        })
    }

    pub fn render(&self, frame: &mut Frame) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3),  // Header
                Constraint::Min(10),    // Main content
                Constraint::Length(5),  // Stats footer
            ])
            .split(frame.size());

        // Header
        let header = Paragraph::new(format!(
            "Kata Spaced Repetition - {} katas due today",
            self.katas_due.len()
        ))
        .block(Block::default().borders(Borders::ALL));
        frame.render_widget(header, chunks[0]);

        // Kata list
        let items: Vec<ListItem> = self
            .katas_due
            .iter()
            .enumerate()
            .map(|(i, kata)| {
                let marker = if i == self.selected_index { ">" } else { " " };
                let text = format!(
                    "{} {} (difficulty: {:.1})",
                    marker, kata.name, kata.current_difficulty
                );
                ListItem::new(text)
            })
            .collect();

        let list = List::new(items)
            .block(Block::default().borders(Borders::ALL).title("Due Today"));
        frame.render_widget(list, chunks[1]);

        // Stats
        let stats_text = format!(
            "Streak: {} days | Reviews today: {} | 7-day success rate: {:.1}%",
            self.stats.streak_days,
            self.stats.total_reviews_today,
            self.stats.success_rate_7d * 100.0
        );
        let stats = Paragraph::new(stats_text)
            .block(Block::default().borders(Borders::ALL).title("Stats"));
        frame.render_widget(stats, chunks[2]);
    }

    pub fn handle_input(&mut self, code: KeyCode) -> DashboardAction {
        match code {
            KeyCode::Down | KeyCode::Char('j') => {
                if self.selected_index < self.katas_due.len().saturating_sub(1) {
                    self.selected_index += 1;
                }
                DashboardAction::None
            }
            KeyCode::Up | KeyCode::Char('k') => {
                self.selected_index = self.selected_index.saturating_sub(1);
                DashboardAction::None
            }
            KeyCode::Enter => {
                if let Some(kata) = self.katas_due.get(self.selected_index) {
                    DashboardAction::SelectKata(kata.clone())
                } else {
                    DashboardAction::None
                }
            }
            _ => DashboardAction::None,
        }
    }
}

pub enum DashboardAction {
    None,
    SelectKata(Kata),
}
```

### Practice Screen

```rust
// src/tui/practice.rs

use std::fs;
use std::process::Command;
use ratatui::{
    layout::{Constraint, Direction, Layout},
    widgets::{Block, Borders, Paragraph, Wrap},
    Frame,
};

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
        // Write template to /tmp
        let template_path = PathBuf::from(format!("/tmp/kata_{}.py", kata.id));

        // Load template from kata exercises directory
        let kata_dir = PathBuf::from("katas/exercises").join(&kata.name);
        let template_source = kata_dir.join("template.py");
        fs::copy(&template_source, &template_path)?;

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
                Constraint::Length(3),  // Title
                Constraint::Min(10),    // Description
                Constraint::Length(5),  // Instructions
            ])
            .split(frame.size());

        // Title
        let title = Paragraph::new(format!("Kata: {}", self.kata.name))
            .block(Block::default().borders(Borders::ALL));
        frame.render_widget(title, chunks[0]);

        // Description
        let desc = Paragraph::new(self.kata.description.clone())
            .wrap(Wrap { trim: true })
            .block(Block::default().borders(Borders::ALL).title("Description"));
        frame.render_widget(desc, chunks[1]);

        // Instructions
        let instructions = match self.status {
            PracticeStatus::ShowingDescription => {
                "[e] Edit in nvim  [q] Back to dashboard"
            }
            PracticeStatus::EditingInProgress => {
                "Editing in nvim... (save and exit to return)"
            }
            PracticeStatus::TestsRunning => {
                "Running tests..."
            }
        };
        let inst = Paragraph::new(instructions)
            .block(Block::default().borders(Borders::ALL).title("Actions"));
        frame.render_widget(inst, chunks[2]);
    }

    pub fn handle_input(&mut self, code: KeyCode, event_tx: Sender<AppEvent>) -> PracticeAction {
        match code {
            KeyCode::Char('e') => {
                self.launch_editor(event_tx);
                PracticeAction::None
            }
            KeyCode::Char('t') => {
                self.run_tests(event_tx);
                PracticeAction::None
            }
            KeyCode::Esc => PracticeAction::BackToDashboard,
            _ => PracticeAction::None,
        }
    }

    fn launch_editor(&mut self, _event_tx: Sender<AppEvent>) {
        // Temporarily leave TUI mode
        crossterm::terminal::disable_raw_mode().unwrap();

        // Spawn nvim
        let status = Command::new("nvim")
            .arg(&self.template_path)
            .status()
            .expect("Failed to launch nvim");

        // Re-enter TUI mode
        crossterm::terminal::enable_raw_mode().unwrap();

        if !status.success() {
            eprintln!("Editor exited with error");
        }
    }

    fn run_tests(&mut self, event_tx: Sender<AppEvent>) {
        self.status = PracticeStatus::TestsRunning;

        let kata_id = self.kata.name.clone();
        let template_path = self.template_path.clone();

        // Spawn background thread
        std::thread::spawn(move || {
            let results = run_python_tests(&kata_id, &template_path);
            event_tx.send(AppEvent::TestComplete(results)).unwrap();
        });
    }
}

pub enum PracticeAction {
    None,
    BackToDashboard,
}
```

### Results Screen

```rust
// src/tui/results.rs

use ratatui::{
    layout::{Constraint, Direction, Layout},
    style::{Color, Style},
    widgets::{Block, Borders, List, ListItem, Paragraph},
    Frame,
};

pub struct ResultsScreen {
    kata: Kata,
    results: TestResults,
    selected_rating: usize,  // 0-3
}

impl ResultsScreen {
    pub fn new(kata: Kata, results: TestResults) -> Self {
        Self {
            kata,
            results,
            selected_rating: 2,  // Default to "Good"
        }
    }

    pub fn render(&self, frame: &mut Frame) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3),   // Header
                Constraint::Min(10),     // Test results
                Constraint::Length(7),   // Rating selection
            ])
            .split(frame.size());

        // Header
        let status = if self.results.passed {
            "All tests passed!"
        } else {
            "Some tests failed"
        };
        let header = Paragraph::new(format!(
            "{} ({}/{} passed in {}ms)",
            status,
            self.results.num_passed,
            self.results.num_passed + self.results.num_failed,
            self.results.duration_ms
        ))
        .block(Block::default().borders(Borders::ALL));
        frame.render_widget(header, chunks[0]);

        // Test results
        let items: Vec<ListItem> = self
            .results
            .results
            .iter()
            .map(|r| {
                let (symbol, color) = match r.status.as_str() {
                    "passed" => ("✓", Color::Green),
                    "failed" => ("✗", Color::Red),
                    _ => ("○", Color::Yellow),
                };
                let text = format!("{} {}", symbol, r.test_name);
                let mut item = ListItem::new(text);
                item = item.style(Style::default().fg(color));
                item
            })
            .collect();

        let list = List::new(items)
            .block(Block::default().borders(Borders::ALL).title("Test Results"));
        frame.render_widget(list, chunks[1]);

        // Rating selection
        let ratings = ["[0] Again", "[1] Hard", "[2] Good", "[3] Easy"];
        let rating_text = ratings
            .iter()
            .enumerate()
            .map(|(i, r)| {
                if i == self.selected_rating {
                    format!("> {}", r)
                } else {
                    format!("  {}", r)
                }
            })
            .collect::<Vec<_>>()
            .join("\n");

        let rating = Paragraph::new(rating_text)
            .block(Block::default().borders(Borders::ALL).title("Rate Difficulty"));
        frame.render_widget(rating, chunks[2]);
    }

    pub fn handle_input(&mut self, code: KeyCode) -> ResultsAction {
        match code {
            KeyCode::Char('0') => self.selected_rating = 0,
            KeyCode::Char('1') => self.selected_rating = 1,
            KeyCode::Char('2') => self.selected_rating = 2,
            KeyCode::Char('3') => self.selected_rating = 3,
            KeyCode::Enter => {
                return ResultsAction::SubmitRating(self.selected_rating as u8)
            }
            _ => {}
        }
        ResultsAction::None
    }
}

pub enum ResultsAction {
    None,
    SubmitRating(u8),  // 0-3
}
```

### Python Runner Integration

```rust
// src/runner/python_runner.rs

use std::process::Command;
use std::path::Path;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct TestResults {
    pub passed: bool,
    pub num_passed: i32,
    pub num_failed: i32,
    pub num_skipped: i32,
    pub duration_ms: i64,
    pub results: Vec<TestResult>,
}

#[derive(Debug, Deserialize)]
pub struct TestResult {
    pub test_name: String,
    pub status: String,
    pub output: String,
}

pub fn run_python_tests(kata_id: &str, template_path: &Path) -> TestResults {
    // Get Python interpreter from PythonEnv (Agent 1's work)
    let python_path = "katas/.venv/bin/python";  // TODO: get from PythonEnv

    let output = Command::new(python_path)
        .args(&["-m", "runner", kata_id, template_path.to_str().unwrap()])
        .current_dir("katas")
        .output()
        .expect("Failed to run Python tests");

    if !output.status.success() {
        eprintln!("Python runner failed: {:?}", output.stderr);
        return TestResults {
            passed: false,
            num_passed: 0,
            num_failed: 1,
            num_skipped: 0,
            duration_ms: 0,
            results: vec![TestResult {
                test_name: "runner_error".to_string(),
                status: "failed".to_string(),
                output: String::from_utf8_lossy(&output.stderr).to_string(),
            }],
        };
    }

    let json = String::from_utf8_lossy(&output.stdout);
    serde_json::from_str(&json).unwrap_or_else(|e| {
        eprintln!("Failed to parse JSON: {}", e);
        TestResults {
            passed: false,
            num_passed: 0,
            num_failed: 1,
            num_skipped: 0,
            duration_ms: 0,
            results: vec![],
        }
    })
}
```

## File Structure You'll Create

```
src/
├── tui/
│   ├── mod.rs
│   ├── app.rs              # Main event loop
│   ├── dashboard.rs        # Dashboard screen
│   ├── practice.rs         # Practice screen
│   └── results.rs          # Results screen
└── runner/
    ├── mod.rs
    └── python_runner.rs    # Spawn pytest, parse JSON
```

## Testing

Focus on integration testing. Create a test kata and verify:
- Dashboard loads katas correctly
- Practice screen writes template to /tmp
- Test runner parses JSON correctly
- Rating updates SM-2 state and reschedules

## Acceptance Criteria

- [ ] TUI launches and shows dashboard
- [ ] Dashboard lists katas due today
- [ ] Selecting kata opens practice screen
- [ ] Practice screen shows description and instructions
- [ ] Pressing 'e' launches nvim with template
- [ ] After editing, can trigger test run
- [ ] Results screen shows pass/fail with colors
- [ ] Rating selection (0-3) works
- [ ] Submitting rating updates database and returns to dashboard
- [ ] Keyboard navigation (j/k, arrows) works smoothly
- [ ] No crashes, graceful error handling

## Handoff to Other Agents

You're the primary user-facing component. Work closely with Agent 5 on integration testing.

## Notes

- Use vim keybindings (j/k for navigation) - user is a nvim user
- Keep UI clean and minimal - this is a personal tool, not flashy
- Error messages should be helpful (e.g., "Python environment not found - run kata-sr setup")
- Test the editor integration thoroughly - entering/exiting raw mode is tricky
- Consider adding a help screen ('?') with keybindings
