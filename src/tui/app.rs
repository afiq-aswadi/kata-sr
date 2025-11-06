//! Main TUI application and event loop.
//!
//! This module provides the central coordinator for all TUI screens,
//! handling keyboard input, async test events, and screen transitions.

use anyhow::Context;
use chrono::{Duration, Utc};
use crossterm::event::{self, Event, KeyCode};
use ratatui::backend::CrosstermBackend;
use ratatui::{Frame, Terminal};
use std::io;
use std::sync::mpsc::{channel, Receiver, Sender};

use super::create_kata::{CreateKataAction, CreateKataScreen};
use super::dashboard::{Dashboard, DashboardAction};
use super::details::{DetailsAction, DetailsScreen};
use super::keybindings;
use super::library::{Library, LibraryAction};
use super::practice::{PracticeAction, PracticeScreen};
use super::results::{ResultsAction, ResultsScreen};
use crate::core::analytics::Analytics;
use crate::core::scheduler::QualityRating;
use crate::db::repo::{Kata, KataRepository, NewKata, NewSession};
use crate::runner::python_runner::TestResults;

/// Events that can be sent to the main application loop.
///
/// These events allow for async communication between background threads
/// (e.g., test runner) and the main UI thread.
#[derive(Debug)]
pub enum AppEvent {
    /// Keyboard input received
    Input(KeyCode),
    /// Test execution completed
    TestComplete(TestResults),
    /// Application should quit
    Quit,
}

/// Current screen being displayed in the TUI.
///
/// Each screen maintains its own state and handles its own rendering.
pub enum Screen {
    /// Dashboard showing katas due today and stats
    Dashboard,
    /// Practice screen for a specific kata
    Practice(Kata, PracticeScreen),
    /// Results screen showing test output and rating selection
    Results(Kata, ResultsScreen),
    /// Help screen showing keybindings
    Help,
    /// Library screen for browsing and adding katas
    Library(Library),
    /// Details screen for viewing kata information
    Details(DetailsScreen),
    /// Create kata screen for generating new kata files
    CreateKata(CreateKataScreen),
}

/// Internal enum for handling screen transitions without borrow conflicts.
enum ScreenAction {
    StartPractice(Kata),
    ReturnToDashboard,
    SubmitRating(Kata, u8),
    OpenLibrary,
    AddKataFromLibrary(String),
    BackFromLibrary,
    ViewDetails(crate::core::kata_loader::AvailableKata, bool),
    BackFromDetails,
    RetryKata(Kata),
    RemoveKataFromDeck(Kata),
    OpenCreateKata,
    SubmitNewKata {
        form_data: crate::core::kata_generator::KataFormData,
        slug: String,
    },
    CancelCreateKata,
}

/// Main application state and event loop coordinator.
///
/// The App struct manages screen transitions, event handling,
/// and database interactions.
pub struct App {
    /// Current screen being displayed
    pub current_screen: Screen,
    /// Cached dashboard state
    pub dashboard: Dashboard,
    /// Database repository for kata and session management
    pub repo: KataRepository,
    /// Sender for async events
    pub event_tx: Sender<AppEvent>,
    /// Receiver for async events
    pub event_rx: Receiver<AppEvent>,
    /// Whether we're showing help screen
    showing_help: bool,
    /// Whether terminal needs to be cleared (after external editor)
    needs_terminal_clear: bool,
}

impl App {
    /// Creates a new App with the given repository.
    ///
    /// Loads initial dashboard state from the database.
    ///
    /// # Arguments
    ///
    /// * `repo` - Database repository for kata management
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use kata_sr::db::repo::KataRepository;
    /// # use kata_sr::tui::app::App;
    /// let repo = KataRepository::new("kata.db")?;
    /// let app = App::new(repo)?;
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn new(repo: KataRepository) -> anyhow::Result<Self> {
        let (tx, rx) = channel();
        let dashboard = Dashboard::load(&repo)?;

        Ok(Self {
            current_screen: Screen::Dashboard,
            dashboard,
            repo,
            event_tx: tx,
            event_rx: rx,
            showing_help: false,
            needs_terminal_clear: false,
        })
    }

    /// Runs the TUI application.
    ///
    /// Sets up the terminal, runs the event loop, and ensures proper cleanup
    /// even on error.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use kata_sr::db::repo::KataRepository;
    /// # use kata_sr::tui::app::App;
    /// let repo = KataRepository::new("kata.db")?;
    /// let mut app = App::new(repo)?;
    /// app.run()?;
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn run(&mut self) -> anyhow::Result<()> {
        crossterm::terminal::enable_raw_mode().context("Failed to enable raw mode")?;

        let mut stdout = io::stdout();
        crossterm::execute!(
            stdout,
            crossterm::terminal::EnterAlternateScreen,
            crossterm::event::EnableMouseCapture
        )
        .context("Failed to setup terminal")?;

        let backend = CrosstermBackend::new(stdout);
        let mut terminal = Terminal::new(backend).context("Failed to create terminal")?;

        let result = self.event_loop(&mut terminal);

        // cleanup - always execute even on error
        crossterm::terminal::disable_raw_mode().context("Failed to disable raw mode")?;
        crossterm::execute!(
            terminal.backend_mut(),
            crossterm::terminal::LeaveAlternateScreen,
            crossterm::event::DisableMouseCapture
        )
        .context("Failed to restore terminal")?;
        terminal.show_cursor().context("Failed to show cursor")?;

        result
    }

    /// Main event loop.
    ///
    /// Continuously renders the current screen, polls for events,
    /// and handles both keyboard input and async events.
    fn event_loop(
        &mut self,
        terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    ) -> anyhow::Result<()> {
        loop {
            // render current screen
            terminal
                .draw(|f| self.render(f))
                .context("Failed to draw frame")?;

            // poll for keyboard events with timeout
            if event::poll(std::time::Duration::from_millis(100))
                .context("Failed to poll events")?
            {
                if let Event::Key(key) = event::read().context("Failed to read event")? {
                    match key.code {
                        KeyCode::Char('q') => {
                            // global quit
                            return Ok(());
                        }
                        KeyCode::Char('?') => {
                            // toggle help screen
                            self.toggle_help();
                        }
                        code => {
                            self.handle_input(code)?;

                            // Clear terminal if needed (after external editor)
                            if self.needs_terminal_clear {
                                terminal.clear().context("Failed to clear terminal")?;
                                self.needs_terminal_clear = false;
                            }
                        }
                    }
                }
            }

            // check for async events (non-blocking)
            if let Ok(event) = self.event_rx.try_recv() {
                self.handle_event(event)?;
            }
        }
    }

    /// Renders the current screen.
    ///
    /// Delegates to the appropriate screen's render method.
    fn render(&mut self, frame: &mut Frame) {
        if self.showing_help {
            keybindings::render_help_screen(frame);
        } else {
            match &self.current_screen {
                Screen::Dashboard => {
                    self.dashboard.render(frame);
                }
                Screen::Practice(_, practice_screen) => {
                    practice_screen.render(frame);
                }
                Screen::Results(_, results_screen) => {
                    results_screen.render(frame);
                }
                Screen::Help => {
                    keybindings::render_help_screen(frame);
                }
                Screen::Library(library) => {
                    library.render(frame);
                }
                Screen::Details(details) => {
                    details.render(frame);
                }
                Screen::CreateKata(create_kata) => {
                    create_kata.render(frame);
                }
            }
        }
    }

    /// Handles keyboard input by delegating to the current screen.
    fn handle_input(&mut self, code: KeyCode) -> anyhow::Result<()> {
        if self.showing_help {
            // any key exits help screen
            self.showing_help = false;
            return Ok(());
        }

        // extract action from current screen to avoid borrow checker issues
        let action_result = match &mut self.current_screen {
            Screen::Dashboard => {
                // handle library key 'l' in dashboard
                if code == KeyCode::Char('l') {
                    return self.execute_action(ScreenAction::OpenLibrary);
                }
                let action = self.dashboard.handle_input(code);
                match action {
                    DashboardAction::SelectKata(kata) => Some(ScreenAction::StartPractice(kata)),
                    DashboardAction::RemoveKata(kata) => Some(ScreenAction::RemoveKataFromDeck(kata)),
                    DashboardAction::None => None,
                }
            }
            Screen::Practice(_kata, practice_screen) => {
                let action = practice_screen.handle_input(code, self.event_tx.clone())?;
                match action {
                    PracticeAction::BackToDashboard => Some(ScreenAction::ReturnToDashboard),
                    PracticeAction::EditorExited => {
                        self.needs_terminal_clear = true;
                        None
                    }
                    PracticeAction::None => None,
                }
            }
            Screen::Results(kata, results_screen) => {
                let action = results_screen.handle_input(code);
                match action {
                    ResultsAction::SubmitRating(rating) => {
                        Some(ScreenAction::SubmitRating(kata.clone(), rating))
                    }
                    ResultsAction::Retry => Some(ScreenAction::RetryKata(kata.clone())),
                    ResultsAction::BackToDashboard => Some(ScreenAction::ReturnToDashboard),
                    ResultsAction::None => None,
                }
            }
            Screen::Help => Some(ScreenAction::ReturnToDashboard),
            Screen::Library(library) => {
                let action = library.handle_input(code);
                match action {
                    LibraryAction::AddKata(name) => Some(ScreenAction::AddKataFromLibrary(name)),
                    LibraryAction::Back => Some(ScreenAction::BackFromLibrary),
                    LibraryAction::ViewDetails(kata) => {
                        let in_deck = library.kata_ids_in_deck.contains(&kata.name);
                        Some(ScreenAction::ViewDetails(kata, in_deck))
                    }
                    LibraryAction::CreateKata => Some(ScreenAction::OpenCreateKata),
                    LibraryAction::None => None,
                }
            }
            Screen::Details(details) => {
                let action = details.handle_input(code);
                match action {
                    DetailsAction::AddKata(name) => Some(ScreenAction::AddKataFromLibrary(name)),
                    DetailsAction::Back => Some(ScreenAction::BackFromDetails),
                    DetailsAction::None => None,
                }
            }
            Screen::CreateKata(create_kata) => {
                let exercises_dir = std::path::Path::new("katas/exercises");
                let action = create_kata.handle_input(code, exercises_dir);
                match action {
                    CreateKataAction::Submit { form_data, slug } => {
                        Some(ScreenAction::SubmitNewKata { form_data, slug })
                    }
                    CreateKataAction::Cancel => Some(ScreenAction::CancelCreateKata),
                    CreateKataAction::None => None,
                }
            }
        };

        // handle the extracted action (no longer borrowing self.current_screen)
        if let Some(action) = action_result {
            self.execute_action(action)?;
        }

        Ok(())
    }

    /// Executes a screen action, handling all state transitions.
    fn execute_action(&mut self, action: ScreenAction) -> anyhow::Result<()> {
        match action {
            ScreenAction::StartPractice(kata) => {
                let practice_screen = PracticeScreen::new(kata.clone())?;
                self.current_screen = Screen::Practice(kata, practice_screen);
            }
            ScreenAction::ReturnToDashboard => {
                self.dashboard = Dashboard::load(&self.repo)?;
                self.current_screen = Screen::Dashboard;
            }
            ScreenAction::SubmitRating(kata, rating) => {
                self.handle_rating_submission(kata, rating)?;
                self.dashboard = Dashboard::load(&self.repo)?;
                self.current_screen = Screen::Dashboard;
            }
            ScreenAction::OpenLibrary => {
                let library = Library::load(&self.repo)?;
                self.current_screen = Screen::Library(library);
            }
            ScreenAction::AddKataFromLibrary(kata_name) => {
                // load available katas
                let available_katas = crate::core::kata_loader::load_available_katas()?;
                if let Some(available_kata) = available_katas.iter().find(|k| k.name == kata_name) {
                    // create NewKata
                    let new_kata = NewKata {
                        name: available_kata.name.clone(),
                        category: available_kata.category.clone(),
                        description: available_kata.description.clone(),
                        base_difficulty: available_kata.base_difficulty,
                        parent_kata_id: None,
                        variation_params: None,
                    };

                    // add to database with next_review_at = now (so it appears as due)
                    self.repo.create_kata(&new_kata, Utc::now())?;

                    // update library state if on library screen, or navigate back if on details
                    match &mut self.current_screen {
                        Screen::Library(library) => {
                            library.mark_as_added(&kata_name);
                        }
                        Screen::Details(_) => {
                            // navigate back to library with updated state
                            let library = Library::load(&self.repo)?;
                            self.current_screen = Screen::Library(library);
                        }
                        _ => {}
                    }

                    // reload dashboard so counts are updated
                    self.dashboard = Dashboard::load(&self.repo)?;
                }
            }
            ScreenAction::BackFromLibrary => {
                self.current_screen = Screen::Dashboard;
            }
            ScreenAction::ViewDetails(kata, in_deck) => {
                let details = DetailsScreen::new(kata, in_deck);
                self.current_screen = Screen::Details(details);
            }
            ScreenAction::BackFromDetails => {
                let library = Library::load(&self.repo)?;
                self.current_screen = Screen::Library(library);
            }
            ScreenAction::RetryKata(kata) => {
                let practice_screen = PracticeScreen::new_retry(kata.clone())?;
                self.current_screen = Screen::Practice(kata, practice_screen);
            }
            ScreenAction::RemoveKataFromDeck(kata) => {
                // Delete the kata from the database
                self.repo.delete_kata(kata.id)?;

                // Reload dashboard to reflect the change
                self.dashboard = Dashboard::load(&self.repo)?;
                self.current_screen = Screen::Dashboard;
            }
            ScreenAction::OpenCreateKata => {
                // Load available katas for dependency selection
                let available_katas = crate::core::kata_loader::load_available_katas()?;
                let kata_names: Vec<String> = available_katas.iter().map(|k| k.name.clone()).collect();

                let create_kata_screen = CreateKataScreen::new(kata_names);
                self.current_screen = Screen::CreateKata(create_kata_screen);
            }
            ScreenAction::SubmitNewKata { form_data, slug } => {
                use crate::core::kata_generator::generate_kata_files;

                let exercises_dir = std::path::Path::new("katas/exercises");

                // Generate the kata files
                match generate_kata_files(&form_data, exercises_dir) {
                    Ok(created_slug) => {
                        // Success! Return to library
                        let library = Library::load(&self.repo)?;
                        self.current_screen = Screen::Library(library);

                        // TODO: Show success message to user
                        eprintln!(
                            "Created kata '{}'! Press 'a' in Library to add to deck.",
                            created_slug
                        );
                    }
                    Err(e) => {
                        // Error! Stay on CreateKata screen
                        // TODO: Show error message to user
                        eprintln!("Failed to create kata: {}", e);

                        // For now, return to library on error
                        let library = Library::load(&self.repo)?;
                        self.current_screen = Screen::Library(library);
                    }
                }
            }
            ScreenAction::CancelCreateKata => {
                // Return to library
                let library = Library::load(&self.repo)?;
                self.current_screen = Screen::Library(library);
            }
        }
        Ok(())
    }

    /// Handles async events from background threads.
    fn handle_event(&mut self, event: AppEvent) -> anyhow::Result<()> {
        match event {
            AppEvent::TestComplete(results) => {
                // transition from Practice to Results
                if let Screen::Practice(kata, _) = &self.current_screen {
                    let results_screen = ResultsScreen::new(kata.clone(), results);
                    self.current_screen = Screen::Results(kata.clone(), results_screen);
                }
            }
            AppEvent::Quit => {
                return Err(anyhow::anyhow!("Quit event received"));
            }
            AppEvent::Input(_) => {
                // input events are handled in the main loop
            }
        }
        Ok(())
    }

    /// Toggles between the current screen and the help screen.
    fn toggle_help(&mut self) {
        self.showing_help = !self.showing_help;
    }

    /// Handles rating submission by updating SM-2 state and creating a session record.
    ///
    /// # Arguments
    ///
    /// * `kata` - The kata that was reviewed
    /// * `rating` - User's quality rating (0-3)
    fn handle_rating_submission(&mut self, kata: Kata, rating: u8) -> anyhow::Result<()> {
        // convert rating to QualityRating
        let quality_rating = QualityRating::from_int(rating as i32)
            .ok_or_else(|| anyhow::anyhow!("Invalid rating: {}", rating))?;

        // get current SM-2 state and update it
        let mut state = kata.sm2_state();
        let interval_days = state.update(quality_rating);

        // calculate next review date
        let now = Utc::now();
        let next_review = now + Duration::days(interval_days);

        // update kata in database
        self.repo
            .update_kata_after_review(kata.id, &state, next_review, now)
            .context("Failed to update kata after review")?;

        // create session record
        let session = NewSession {
            kata_id: kata.id,
            started_at: now,
            completed_at: Some(now),
            test_results_json: None, // could serialize TestResults if needed
            num_passed: None,
            num_failed: None,
            num_skipped: None,
            duration_ms: None,
            quality_rating: Some(rating as i32),
        };

        self.repo
            .create_session(&session)
            .context("Failed to create session record")?;

        // update daily statistics
        let analytics = Analytics::new(&self.repo);
        analytics
            .update_daily_stats()
            .context("Failed to update daily stats")?;

        Ok(())
    }
}
