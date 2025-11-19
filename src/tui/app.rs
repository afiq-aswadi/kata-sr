//! Main TUI application and event loop.
//!
//! This module provides the central coordinator for all TUI screens,
//! handling keyboard input, async test events, and screen transitions.

use anyhow::Context;
use chrono::{Duration, Utc};
use crossterm::event::{self, Event, KeyCode};
use ratatui::backend::CrosstermBackend;
use ratatui::layout::Rect;
use ratatui::{Frame, Terminal};
use std::io;
use std::sync::mpsc::{channel, Receiver, Sender};

use crate::core::analytics::Analytics;
use crate::core::fsrs::{FsrsParams, Rating};
use crate::db::repo::{Kata, NewKata, NewSession};
use super::create_kata::CreateKataScreen;
use super::dashboard::Dashboard;
use super::details::DetailsScreen;
use super::done::DoneScreen;
use super::edit_kata::EditKataScreen;
use super::keybindings;
use super::library::Library;
use super::practice::PracticeScreen;
use super::results::ResultsScreen;
use super::session_detail::SessionDetailScreen;
use super::session_history::SessionHistoryScreen;
use super::settings::SettingsScreen;
use super::startup::StartupScreen;
use crate::config::AppConfig;
use crate::db::repo::KataRepository;
use crate::runner::python_runner::TestResults;

/// Style for popup messages.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PopupStyle {
    /// Success message (green)
    Success,
    /// Error message (red)
    Error,
}

/// A popup message to display to the user.
#[derive(Debug, Clone)]
pub struct PopupMessage {
    pub title: String,
    pub message: String,
    pub style: PopupStyle,
}

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

use super::screen::{Screen, ScreenAction};

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
    /// Application configuration
    pub config: AppConfig,
    /// Sender for async events
    pub event_tx: Sender<AppEvent>,
    /// Receiver for async events
    pub event_rx: Receiver<AppEvent>,
    /// Whether we're showing help screen
    showing_help: bool,
    /// Whether terminal needs to be cleared (after external editor)
    needs_terminal_clear: bool,
    /// Popup message to display to the user
    popup_message: Option<PopupMessage>,
    /// Screen to restore when closing settings
    previous_screen_before_settings: Option<Box<Screen>>,
    /// Library hide_flagged state (preserved across library reloads)
    library_hide_flagged: bool,
    /// Dashboard hide_flagged state (preserved across dashboard reloads)
    dashboard_hide_flagged: bool,
}

impl App {
    /// Creates a new App with the given repository and configuration.
    ///
    /// Loads initial dashboard state from the database.
    ///
    /// # Arguments
    ///
    /// * `repo` - Database repository for kata management
    /// * `config` - Application configuration
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use kata_sr::db::repo::KataRepository;
    /// # use kata_sr::tui::app::App;
    /// # use kata_sr::config::AppConfig;
    /// let repo = KataRepository::new("kata.db")?;
    /// let config = AppConfig::load()?;
    /// let app = App::new(repo, config)?;
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn new(repo: KataRepository, config: AppConfig) -> anyhow::Result<Self> {
        let (tx, rx) = channel();
        let dashboard = Dashboard::load(&repo, config.display.heatmap_days)?;
        let startup_screen = StartupScreen::load(&repo, config.display.heatmap_days)?;

        let app = Self {
            current_screen: Screen::Startup(startup_screen),
            dashboard,
            repo,
            config,
            event_tx: tx,
            event_rx: rx,
            showing_help: false,
            needs_terminal_clear: false,
            popup_message: None,
            previous_screen_before_settings: None,
            library_hide_flagged: false,
            dashboard_hide_flagged: false,
        };

        Ok(app)
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
    /// # use kata_sr::config::AppConfig;
    /// let repo = KataRepository::new("kata.db")?;
    /// let config = AppConfig::default();
    /// let mut app = App::new(repo, config)?;
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
                        KeyCode::Esc => {
                            // Handle Esc with priority for help/popup dismissal
                            if self.showing_help {
                                self.showing_help = false;
                            } else if self.popup_message.is_some() {
                                self.popup_message = None;
                            } else {
                                // Let screen handle Esc first, then apply global "return to startup" for simple back actions
                                self.handle_escape_with_redirect()?;

                                // Clear terminal if needed (after external editor)
                                if self.needs_terminal_clear {
                                    terminal.clear().context("Failed to clear terminal")?;
                                    self.needs_terminal_clear = false;
                                }
                            }
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
            return;
        }

        self.current_screen.render(frame, &self.dashboard);

        // Render popup overlay if present
        if self.popup_message.is_some() {
            self.render_popup_overlay(frame);
        }
    }

    /// Handles keyboard input by delegating to the current screen.
    fn handle_input(&mut self, code: KeyCode) -> anyhow::Result<()> {
        if self.showing_help {
            // any key exits help screen
            self.showing_help = false;
            return Ok(());
        }

        // If popup is showing, any key dismisses it
        if self.popup_message.is_some() {
            self.popup_message = None;
            return Ok(());
        }

        // extract action from current screen to avoid borrow checker issues
        let action_result = self.current_screen.handle_input(code, &mut self.dashboard, self.event_tx.clone());

        // handle the extracted action (no longer borrowing self.current_screen)
        if let Ok(action) = action_result {
            self.execute_action(action)?;
        }

        Ok(())
    }

    /// Handles Escape key with screen-first priority, then redirects "back" actions to startup.
    ///
    /// This allows screens to handle Escape for modal cleanup (e.g., closing solution view,
    /// submitting give-up rating) while still providing global "return to startup" behavior
    /// for simple back navigation.
    fn handle_escape_with_redirect(&mut self) -> anyhow::Result<()> {
        // Let the current screen handle Escape normally
        self.handle_input(KeyCode::Esc)?;

        // After screen processes Escape, check if it resulted in "back to dashboard" navigation.
        // We can't directly intercept the action, but we can observe the screen state.
        // For screens that use Esc to go back, they've already navigated.
        // Just check if we're on Dashboard or Done screen and redirect to Startup.
        match &self.current_screen {
            Screen::Dashboard | Screen::Done(_) => {
                // Screen navigated to dashboard, redirect to startup instead
                self.current_screen = Screen::Startup(StartupScreen::load(
                    &self.repo,
                    self.config.display.heatmap_days,
                )?);
            }
            _ => {
                // Screen handled Esc internally (modal closed, rating submitted, etc.)
                // or we're already on startup - no additional action needed
            }
        }

        Ok(())
    }

    /// Executes a screen action, handling all state transitions.
    fn execute_action(&mut self, action: ScreenAction) -> anyhow::Result<()> {
        match action {
            // Practice Init
            ScreenAction::StartPractice(_)
            | ScreenAction::AttemptKataWithoutDeck(_)
            | ScreenAction::StartNextDue
            | ScreenAction::RetryKata(_) => {
                self.handle_practice_init_action(action)?;
            }

            // Results
            ScreenAction::SubmitRating(_, _, _)
            | ScreenAction::BuryKata(_)
            | ScreenAction::ResultsSolutionViewed(_, _)
            | ScreenAction::ResultsToggleFlagWithReason(_, _) => {
                self.handle_results_action(action)?;
            }

            // Library
            ScreenAction::OpenLibrary
            | ScreenAction::AddKataFromLibrary(_)
            | ScreenAction::BackFromLibrary
            | ScreenAction::RemoveKataFromDeck(_)
            | ScreenAction::LibraryToggleFlagKata(_)
            | ScreenAction::LibraryToggleHideFlagged
            | ScreenAction::LibraryToggleFlagWithReason(_, _) => {
                self.handle_library_action(action)?;
            }

            // Dashboard
            ScreenAction::ReturnToDashboard
            | ScreenAction::DashboardSelectKata(_)
            | ScreenAction::DashboardRemoveKata(_)
            | ScreenAction::DashboardEditKata(_)
            | ScreenAction::DashboardToggleFlagKata(_)
            | ScreenAction::DashboardToggleHideFlagged => {
                self.handle_dashboard_action(action)?;
            }

            // Session
            ScreenAction::ViewSessionHistory(_)
            | ScreenAction::ViewSessionDetail(_)
            | ScreenAction::DeleteSession(_)
            | ScreenAction::BackFromSessionHistory
            | ScreenAction::BackFromSessionDetail => {
                self.handle_session_action(action)?;
            }

            // Kata CRUD
            ScreenAction::OpenCreateKata
            | ScreenAction::SubmitNewKata { .. }
            | ScreenAction::CancelCreateKata
            | ScreenAction::OpenEditKata(_)
            | ScreenAction::SubmitEditKata { .. }
            | ScreenAction::OpenEditorFile(_)
            | ScreenAction::CancelEditKata
            | ScreenAction::EditKataByName(_) => {
                self.handle_kata_crud_action(action)?;
            }

            // Settings
            ScreenAction::OpenSettings
            | ScreenAction::CloseSettings
            | ScreenAction::SaveSettings(_) => {
                self.handle_settings_action(action)?;
            }

            // Misc
            ScreenAction::PracticeEditorExited
            | ScreenAction::ViewDetails(_, _)
            | ScreenAction::BackFromDetails => {
                self.handle_misc_action(action)?;
            }

            ScreenAction::None => {}
        }
        Ok(())
    }

    // --- Action Handlers ---

    fn handle_practice_init_action(&mut self, action: ScreenAction) -> anyhow::Result<()> {
        match action {
            ScreenAction::StartPractice(kata) => {
                let practice_screen =
                    PracticeScreen::new(kata.clone(), self.config.editor.clone())?;
                self.current_screen = Screen::Practice(kata, practice_screen);
            }
            ScreenAction::AttemptKataWithoutDeck(available_kata) => {
                let preview_kata = Kata {
                    id: -1,
                    name: available_kata.name.clone(),
                    category: available_kata.category.clone(),
                    description: available_kata.description.clone(),
                    tags: available_kata.tags.clone(),
                    base_difficulty: available_kata.base_difficulty,
                    current_difficulty: available_kata.base_difficulty as f64,
                    parent_kata_id: None,
                    variation_params: None,
                    next_review_at: None,
                    last_reviewed_at: None,
                    current_ease_factor: 2.5,
                    current_interval_days: 0,
                    current_repetition_count: 0,
                    fsrs_stability: 0.0,
                    fsrs_difficulty: 0.0,
                    fsrs_elapsed_days: 0,
                    fsrs_scheduled_days: 0,
                    fsrs_reps: 0,
                    fsrs_lapses: 0,
                    fsrs_state: "New".to_string(),
                    scheduler_type: "FSRS".to_string(),
                    is_problematic: false,
                    problematic_notes: None,
                    flagged_at: None,
                    created_at: Utc::now(),
                };
                let practice_screen =
                    PracticeScreen::new(preview_kata.clone(), self.config.editor.clone())?;
                self.current_screen = Screen::Practice(preview_kata, practice_screen);
            }
            ScreenAction::StartNextDue => {
                if self.dashboard.katas_due.is_empty() {
                    self.dashboard = Dashboard::load_with_filter(
                        &self.repo,
                        self.config.display.heatmap_days,
                        self.dashboard_hide_flagged,
                    )?;
                }
                if let Some(next_kata) = self.dashboard.katas_due.first().cloned() {
                    let practice_screen =
                        PracticeScreen::new(next_kata.clone(), self.config.editor.clone())?;
                    self.current_screen = Screen::Practice(next_kata, practice_screen);
                } else {
                    self.refresh_dashboard_screen()?;
                }
            }
            ScreenAction::RetryKata(kata) => {
                let practice_screen =
                    PracticeScreen::new_retry(kata.clone(), self.config.editor.clone())?;
                self.current_screen = Screen::Practice(kata, practice_screen);
            }
            _ => {}
        }
        Ok(())
    }

    fn handle_results_action(&mut self, action: ScreenAction) -> anyhow::Result<()> {
        match action {
            ScreenAction::SubmitRating(kata, rating, results) => {
                self.handle_rating_submission(kata, rating, results)?;
                self.dashboard = Dashboard::load_with_filter(
                    &self.repo,
                    self.config.display.heatmap_days,
                    self.dashboard_hide_flagged,
                )?;
                if let Screen::Results(_, results_screen) = &mut self.current_screen {
                    results_screen.mark_rating_submitted(rating, self.dashboard.katas_due.len());
                }
            }
            ScreenAction::BuryKata(kata) => {
                self.handle_bury_kata(&kata)?;
                self.dashboard = Dashboard::load_with_filter(
                    &self.repo,
                    self.config.display.heatmap_days,
                    self.dashboard_hide_flagged,
                )?;
                self.refresh_dashboard_screen()?;
            }
            ScreenAction::ResultsSolutionViewed(kata, results) => {
                self.needs_terminal_clear = true;
                
                if kata.id == -1 {
                    // Preview mode: return to library
                    self.execute_action(ScreenAction::OpenLibrary)?;
                } else {
                    // Normal mode: auto-submit Rating::Again (1)
                    self.execute_action(ScreenAction::SubmitRating(kata, 1, results))?;
                }
            }
            ScreenAction::ResultsToggleFlagWithReason(kata, reason) => {
                let kata_id = kata.id;
                let was_problematic = kata.is_problematic;

                if was_problematic {
                    self.repo.unflag_kata(kata_id)?;
                } else {
                    self.repo.flag_kata(kata.id, reason)?;
                }

                if let Some(fresh_kata) = self.repo.get_kata_by_id(kata_id)? {
                    let old_screen =
                        std::mem::replace(&mut self.current_screen, Screen::Dashboard);
                    if let Screen::Results(_, mut rs) = old_screen {
                        rs.update_kata(fresh_kata.clone());
                        self.current_screen = Screen::Results(fresh_kata, rs);
                    }
                }

                self.dashboard = Dashboard::load_with_filter(
                    &self.repo,
                    self.config.display.heatmap_days,
                    self.dashboard_hide_flagged,
                )?;
            }
            _ => {}
        }
        Ok(())
    }

    fn handle_library_action(&mut self, action: ScreenAction) -> anyhow::Result<()> {
        match action {
            ScreenAction::OpenLibrary => {
                let library = Library::load_with_filter(
                    &self.repo,
                    &self.config.library.default_sort,
                    self.config.library.default_sort_ascending,
                    self.library_hide_flagged,
                )?;
                self.current_screen = Screen::Library(library);
            }
            ScreenAction::AddKataFromLibrary(kata_name) => {
                let available_katas = crate::core::kata_loader::load_available_katas()?;
                if let Some(available_kata) = available_katas.iter().find(|k| k.name == kata_name) {
                    let new_kata = NewKata {
                        name: available_kata.name.clone(),
                        category: available_kata.category.clone(),
                        description: available_kata.description.clone(),
                        base_difficulty: available_kata.base_difficulty,
                        parent_kata_id: None,
                        variation_params: None,
                    };

                    self.repo.create_kata(&new_kata, Utc::now())?;

                    match &mut self.current_screen {
                        Screen::Library(library) => {
                            library.mark_as_added(&kata_name);
                            self.needs_terminal_clear = true;
                        }
                        Screen::Details(_) => {
                            let library = Library::load_with_filter(
                                &self.repo,
                                &self.config.library.default_sort,
                                self.config.library.default_sort_ascending,
                                self.library_hide_flagged,
                            )?;
                            self.current_screen = Screen::Library(library);
                            self.needs_terminal_clear = true;
                        }
                        _ => {}
                    }

                    self.dashboard = Dashboard::load_with_filter(
                        &self.repo,
                        self.config.display.heatmap_days,
                        self.dashboard_hide_flagged,
                    )?;
                }
            }
            ScreenAction::BackFromLibrary => {
                self.refresh_dashboard_screen()?;
            }
            ScreenAction::RemoveKataFromDeck(kata) => {
                self.repo.delete_kata(kata.id)?;

                self.dashboard = Dashboard::load_with_filter(
                    &self.repo,
                    self.config.display.heatmap_days,
                    self.dashboard_hide_flagged,
                )?;

                match &mut self.current_screen {
                    Screen::Library(library) => {
                        library.mark_as_removed(&kata.name);
                        library.refresh_deck(&self.repo)?;
                    }
                    _ => {
                        self.refresh_dashboard_screen()?;
                    }
                }
            }
            ScreenAction::LibraryToggleFlagKata(kata) => {
                if kata.is_problematic {
                    self.repo.unflag_kata(kata.id)?;
                } else {
                    self.repo.flag_kata(kata.id, None)?;
                }
                self.dashboard = Dashboard::load_with_filter(
                    &self.repo,
                    self.config.display.heatmap_days,
                    self.dashboard_hide_flagged,
                )?;
                return self.execute_action(ScreenAction::OpenLibrary);
            }
            ScreenAction::LibraryToggleHideFlagged => {
                if let Screen::Library(library) = &self.current_screen {
                     self.library_hide_flagged = !library.hide_flagged;
                    let new_library = Library::load_with_filter(
                        &self.repo,
                        &self.config.library.default_sort,
                        self.config.library.default_sort_ascending,
                        self.library_hide_flagged,
                    )?;
                    self.current_screen = Screen::Library(new_library);
                }
            }
            ScreenAction::LibraryToggleFlagWithReason(kata, reason) => {
                if kata.is_problematic {
                    self.repo.unflag_kata(kata.id)?;
                } else {
                    self.repo.flag_kata(kata.id, reason)?;
                }
                self.dashboard = Dashboard::load_with_filter(
                    &self.repo,
                    self.config.display.heatmap_days,
                    self.dashboard_hide_flagged,
                )?;
                return self.execute_action(ScreenAction::OpenLibrary);
            }
            _ => {}
        }
        Ok(())
    }

    fn handle_dashboard_action(&mut self, action: ScreenAction) -> anyhow::Result<()> {
        match action {
            ScreenAction::ReturnToDashboard => {
                self.dashboard = Dashboard::load_with_filter(
                    &self.repo,
                    self.config.display.heatmap_days,
                    self.dashboard_hide_flagged,
                )?;
                self.refresh_dashboard_screen()?;
            }
            ScreenAction::DashboardSelectKata(kata) => {
                 let practice_screen =
                    PracticeScreen::new(kata.clone(), self.config.editor.clone())?;
                self.current_screen = Screen::Practice(kata, practice_screen);
            }
            ScreenAction::DashboardRemoveKata(kata) => {
                 self.execute_action(ScreenAction::RemoveKataFromDeck(kata))?;
            }
            ScreenAction::DashboardEditKata(kata) => {
                 self.execute_action(ScreenAction::OpenEditKata(kata.id))?;
            }
            ScreenAction::DashboardToggleFlagKata(kata) => {
                if kata.is_problematic {
                    self.repo.unflag_kata(kata.id)?;
                } else {
                    self.repo.flag_kata(kata.id, None)?;
                }
                self.dashboard = Dashboard::load_with_filter(
                    &self.repo,
                    self.config.display.heatmap_days,
                    self.dashboard_hide_flagged,
                )?;
                if let Screen::Practice(_, _) = &self.current_screen {
                     if let Some(fresh_kata) = self.repo.get_kata_by_id(kata.id)? {
                        let old_screen =
                            std::mem::replace(&mut self.current_screen, Screen::Dashboard);
                        if let Screen::Practice(_, ps) = old_screen {
                            self.current_screen = Screen::Practice(fresh_kata, ps);
                        }
                    }
                }
            }
            ScreenAction::DashboardToggleHideFlagged => {
                self.dashboard_hide_flagged = !self.dashboard_hide_flagged;
                self.dashboard = Dashboard::load_with_filter(
                    &self.repo,
                    self.config.display.heatmap_days,
                    self.dashboard_hide_flagged,
                )?;
                if let Screen::Done(_) = self.current_screen {
                     self.refresh_dashboard_screen()?;
                }
            }
            _ => {}
        }
        Ok(())
    }

    fn handle_session_action(&mut self, action: ScreenAction) -> anyhow::Result<()> {
        match action {
            ScreenAction::ViewSessionHistory(kata) => {
                let session_history = SessionHistoryScreen::new(kata, &self.repo)?;
                self.current_screen = Screen::SessionHistory(session_history);
            }
            ScreenAction::ViewSessionDetail(session_id) => {
                let session_detail = SessionDetailScreen::new(session_id, &self.repo)?;
                self.current_screen = Screen::SessionDetail(session_detail);
            }
            ScreenAction::DeleteSession(session_id) => {
                self.repo.delete_session(session_id)?;

                if let Screen::SessionHistory(ref session_history) = &self.current_screen {
                    let kata = session_history.kata.clone();
                    let mut new_session_history = SessionHistoryScreen::new(kata, &self.repo)?;
                    new_session_history.selected = session_history
                        .selected
                        .min(new_session_history.sessions.len().saturating_sub(1));
                    self.current_screen = Screen::SessionHistory(new_session_history);
                }
            }
            ScreenAction::BackFromSessionHistory => {
                self.refresh_dashboard_screen()?;
            }
            ScreenAction::BackFromSessionDetail => {
                if let Screen::SessionDetail(ref session_detail) = &self.current_screen {
                    let kata = self
                        .repo
                        .get_kata_by_id(session_detail.session.kata_id)?
                        .ok_or_else(|| anyhow::anyhow!("Kata not found"))?;
                    let session_history = SessionHistoryScreen::new(kata, &self.repo)?;
                    self.current_screen = Screen::SessionHistory(session_history);
                }
            }
            _ => {}
        }
        Ok(())
    }

    fn handle_kata_crud_action(&mut self, action: ScreenAction) -> anyhow::Result<()> {
        match action {
            ScreenAction::OpenCreateKata => {
                let available_katas = crate::core::kata_loader::load_available_katas()?;
                let kata_names: Vec<String> =
                    available_katas.iter().map(|k| k.name.clone()).collect();

                let create_kata_screen = CreateKataScreen::new(kata_names);
                self.current_screen = Screen::CreateKata(create_kata_screen);
            }
            ScreenAction::SubmitNewKata { form_data, slug: _ } => {
                use crate::core::kata_generator::generate_kata_files;
                let exercises_dir = std::path::Path::new("katas/exercises");

                match generate_kata_files(&form_data, exercises_dir) {
                    Ok(created_slug) => {
                        let library = Library::load_with_filter(
                            &self.repo,
                            &self.config.library.default_sort,
                            self.config.library.default_sort_ascending,
                            self.library_hide_flagged,
                        )?;
                        self.current_screen = Screen::Library(library);

                        self.popup_message = Some(PopupMessage {
                            title: "Success!".to_string(),
                            message: format!(
                                "Created kata '{}'!\n\nPress 'a' in the All Katas tab to add it to your deck.",
                                created_slug
                            ),
                            style: PopupStyle::Success,
                        });
                    }
                    Err(e) => {
                        let library = Library::load_with_filter(
                            &self.repo,
                            &self.config.library.default_sort,
                            self.config.library.default_sort_ascending,
                            self.library_hide_flagged,
                        )?;
                        self.current_screen = Screen::Library(library);

                        self.popup_message = Some(PopupMessage {
                            title: "Error Creating Kata".to_string(),
                            message: format!("Failed to create kata:\n\n{}", e),
                            style: PopupStyle::Error,
                        });
                    }
                }
            }
            ScreenAction::CancelCreateKata => {
                let library = Library::load_with_filter(
                    &self.repo,
                    &self.config.library.default_sort,
                    self.config.library.default_sort_ascending,
                    self.library_hide_flagged,
                )?;
                self.current_screen = Screen::Library(library);
            }
            ScreenAction::OpenEditKata(kata_id) => {
                if let Some(kata) = self.repo.get_kata_by_id(kata_id)? {
                    let tags = self.repo.get_kata_tags(kata_id)?;
                    let dependencies = self.repo.get_kata_dependencies(kata_id)?;

                    let dep_names: Vec<String> = dependencies
                        .iter()
                        .filter_map(|dep_id| {
                            self.repo
                                .get_kata_by_id(*dep_id)
                                .ok()
                                .flatten()
                                .map(|k| k.name)
                        })
                        .collect();

                    let available_katas = crate::core::kata_loader::load_available_katas()?;
                    let kata_names: Vec<String> =
                        available_katas.iter().map(|k| k.name.clone()).collect();

                    let exercises_dir = std::path::PathBuf::from("katas/exercises");
                    let edit_kata_screen =
                        EditKataScreen::new(&kata, tags, dep_names, kata_names, exercises_dir);
                    self.current_screen = Screen::EditKata(edit_kata_screen);
                }
            }
            ScreenAction::SubmitEditKata {
                kata_id,
                original_slug,
                form_data,
                new_slug,
            } => {
                use crate::core::kata_generator::{
                    rename_kata_directory, update_dependency_in_manifest, update_manifest,
                };

                let exercises_dir = std::path::Path::new("katas/exercises");
                let name_changed = original_slug != new_slug;

                let original_kata = self
                    .repo
                    .get_kata_by_id(kata_id)?
                    .ok_or_else(|| anyhow::anyhow!("Kata not found"))?;
                let original_dependencies = self.repo.get_kata_dependencies(kata_id)?;
                let original_tags = self.repo.get_kata_tags(kata_id)?;

                let new_tags: Vec<String> = form_data
                    .category
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect();

                let primary_category = new_tags
                    .first()
                    .cloned()
                    .unwrap_or_else(|| form_data.category.clone());

                if name_changed {
                    match self.repo.update_kata_full_metadata(
                        kata_id,
                        &new_slug,
                        &form_data.description,
                        &primary_category,
                        form_data.difficulty as i32,
                    ) {
                        Ok(_) => {
                            if let Err(e) = self.repo.set_kata_tags(kata_id, &new_tags) {
                                let _ = self.repo.update_kata_full_metadata(
                                    kata_id,
                                    &original_kata.name,
                                    &original_kata.description,
                                    &original_kata.category,
                                    original_kata.base_difficulty,
                                );
                                eprintln!("Failed to update tags: {}", e);
                                return Ok(());
                            }

                            let dep_ids = self.resolve_dependency_ids(&form_data.dependencies)?;
                            if let Err(e) = self.repo.replace_dependencies(kata_id, &dep_ids) {
                                let _ = self.repo.update_kata_full_metadata(
                                    kata_id,
                                    &original_kata.name,
                                    &original_kata.description,
                                    &original_kata.category,
                                    original_kata.base_difficulty,
                                );
                                let _ = self.repo.set_kata_tags(kata_id, &original_tags);
                                eprintln!("Failed to update dependencies: {}", e);
                                return Ok(());
                            }

                            match rename_kata_directory(exercises_dir, &original_slug, &new_slug) {
                                Ok(_) => {
                                    let kata_dir = exercises_dir.join(&new_slug);
                                    if let Err(e) =
                                        update_manifest(&kata_dir, &form_data, &new_slug, &new_tags)
                                    {
                                        let _ = rename_kata_directory(
                                            exercises_dir,
                                            &new_slug,
                                            &original_slug,
                                        );
                                        let _ = self.repo.update_kata_full_metadata(
                                            kata_id,
                                            &original_kata.name,
                                            &original_kata.description,
                                            &original_kata.category,
                                            original_kata.base_difficulty,
                                        );
                                        let _ = self.repo.set_kata_tags(kata_id, &original_tags);
                                        let _ = self
                                            .repo
                                            .replace_dependencies(kata_id, &original_dependencies);
                                        eprintln!("Failed to update manifest: {}", e);
                                        return Ok(());
                                    }

                                    let dependent_kata_ids =
                                        self.repo.get_dependent_katas(kata_id)?;
                                    for dep_kata_id in &dependent_kata_ids {
                                        if let Ok(Some(dep_kata)) =
                                            self.repo.get_kata_by_id(*dep_kata_id)
                                        {
                                            let dep_kata_dir = exercises_dir.join(&dep_kata.name);
                                            if let Err(e) = update_dependency_in_manifest(
                                                &dep_kata_dir,
                                                &original_slug,
                                                &new_slug,
                                            ) {
                                                eprintln!("Failed to update dependent manifest for '{}': {}", dep_kata.name, e);

                                                for rollback_id in &dependent_kata_ids {
                                                    if rollback_id == dep_kata_id {
                                                        break;
                                                    }
                                                    if let Ok(Some(rb_kata)) =
                                                        self.repo.get_kata_by_id(*rollback_id)
                                                    {
                                                        let rb_kata_dir =
                                                            exercises_dir.join(&rb_kata.name);
                                                        let _ = update_dependency_in_manifest(
                                                            &rb_kata_dir,
                                                            &new_slug,
                                                            &original_slug,
                                                        );
                                                    }
                                                }

                                                let _ = rename_kata_directory(
                                                    exercises_dir,
                                                    &new_slug,
                                                    &original_slug,
                                                );
                                                let _ = self.repo.update_kata_full_metadata(
                                                    kata_id,
                                                    &original_kata.name,
                                                    &original_kata.description,
                                                    &original_kata.category,
                                                    original_kata.base_difficulty,
                                                );
                                                let _ = self
                                                    .repo
                                                    .set_kata_tags(kata_id, &original_tags);
                                                let _ = self.repo.replace_dependencies(
                                                    kata_id,
                                                    &original_dependencies,
                                                );
                                                return Ok(());
                                            }
                                        }
                                    }

                                    let library = Library::load_with_filter(
                                        &self.repo,
                                        &self.config.library.default_sort,
                                        self.config.library.default_sort_ascending,
                                        self.library_hide_flagged,
                                    )?;
                                    self.current_screen = Screen::Library(library);
                                    eprintln!("Kata '{}' updated successfully!", new_slug);
                                }
                                Err(e) => {
                                    let _ = self.repo.update_kata_full_metadata(
                                        kata_id,
                                        &original_kata.name,
                                        &original_kata.description,
                                        &original_kata.category,
                                        original_kata.base_difficulty,
                                    );
                                    let _ = self.repo.set_kata_tags(kata_id, &original_tags);
                                    let _ = self
                                        .repo
                                        .replace_dependencies(kata_id, &original_dependencies);
                                    eprintln!("Failed to rename kata directory: {}", e);
                                }
                            }
                        }
                        Err(e) => {
                            eprintln!("Failed to update database: {}", e);
                        }
                    }
                } else {
                    match self.repo.update_kata_metadata(
                        kata_id,
                        &form_data.description,
                        &primary_category,
                        form_data.difficulty as i32,
                    ) {
                        Ok(_) => {
                            if let Err(e) = self.repo.set_kata_tags(kata_id, &new_tags) {
                                let _ = self.repo.update_kata_metadata(
                                    kata_id,
                                    &original_kata.description,
                                    &original_kata.category,
                                    original_kata.base_difficulty,
                                );
                                eprintln!("Failed to update tags: {}", e);
                                return Ok(());
                            }

                            let dep_ids = self.resolve_dependency_ids(&form_data.dependencies)?;
                            if let Err(e) = self.repo.replace_dependencies(kata_id, &dep_ids) {
                                let _ = self.repo.update_kata_metadata(
                                    kata_id,
                                    &original_kata.description,
                                    &original_kata.category,
                                    original_kata.base_difficulty,
                                );
                                let _ = self.repo.set_kata_tags(kata_id, &original_tags);
                                eprintln!("Failed to update dependencies: {}", e);
                                return Ok(());
                            }

                            let kata_dir = exercises_dir.join(&original_slug);
                            if let Err(e) =
                                update_manifest(&kata_dir, &form_data, &original_slug, &new_tags)
                            {
                                let _ = self.repo.update_kata_metadata(
                                    kata_id,
                                    &original_kata.description,
                                    &original_kata.category,
                                    original_kata.base_difficulty,
                                );
                                let _ = self.repo.set_kata_tags(kata_id, &original_tags);
                                let _ = self
                                    .repo
                                    .replace_dependencies(kata_id, &original_dependencies);
                                eprintln!("Failed to update manifest: {}", e);
                                return Ok(());
                            }

                            let library = Library::load_with_filter(
                                &self.repo,
                                &self.config.library.default_sort,
                                self.config.library.default_sort_ascending,
                                self.library_hide_flagged,
                            )?;
                            self.current_screen = Screen::Library(library);
                            eprintln!("Kata '{}' updated successfully!", original_slug);
                        }
                        Err(e) => {
                            eprintln!("Failed to update kata: {}", e);
                        }
                    }
                }
            }
            ScreenAction::OpenEditorFile(file_path) => {
                use std::io::Write;
                use std::process::Command;

                let editor = std::env::var("EDITOR").unwrap_or_else(|_| "nvim".to_string());

                let mut stdout = std::io::stdout();
                crossterm::execute!(
                    stdout,
                    crossterm::terminal::Clear(crossterm::terminal::ClearType::All),
                    crossterm::cursor::Show,
                    crossterm::terminal::LeaveAlternateScreen,
                )
                .context("Failed to leave alternate screen before launching editor")?;

                crossterm::terminal::disable_raw_mode()
                    .context("Failed to disable raw mode before launching editor")?;

                let parts: Vec<&str> = editor.split_whitespace().collect();
                if parts.is_empty() {
                    anyhow::bail!("EDITOR environment variable is empty");
                }
                let editor_cmd = parts[0];
                let editor_args = &parts[1..];

                let status_result = Command::new(editor_cmd)
                    .args(editor_args)
                    .arg(&file_path)
                    .status();

                crossterm::terminal::enable_raw_mode()
                    .context("Failed to re-enable raw mode after exiting editor")?;

                crossterm::execute!(
                    stdout,
                    crossterm::terminal::EnterAlternateScreen,
                    crossterm::terminal::Clear(crossterm::terminal::ClearType::All),
                    crossterm::cursor::Hide,
                    crossterm::cursor::MoveTo(0, 0),
                )
                .context("Failed to re-enter alternate screen after exiting editor")?;

                stdout
                    .flush()
                    .context("Failed to flush stdout after terminal reset")?;

                let editor_status =
                    status_result.with_context(|| format!("Failed to launch {}", editor))?;

                if !editor_status.success() {
                    eprintln!(
                        "{} exited with non-zero status (code {:?})",
                        editor,
                        editor_status.code()
                    );
                }
            }
            ScreenAction::CancelEditKata => {
                let library = Library::load_with_filter(
                    &self.repo,
                    &self.config.library.default_sort,
                    self.config.library.default_sort_ascending,
                    self.library_hide_flagged,
                )?;
                self.current_screen = Screen::Library(library);
            }
            ScreenAction::EditKataByName(name) => {
                if let Ok(Some(kata)) = self.repo.get_kata_by_name(&name) {
                    self.execute_action(ScreenAction::OpenEditKata(kata.id))?;
                }
            }
            _ => {}
        }
        Ok(())
    }

    fn handle_settings_action(&mut self, action: ScreenAction) -> anyhow::Result<()> {
        match action {
            ScreenAction::OpenSettings => {
                let previous = std::mem::replace(
                    &mut self.current_screen,
                    Screen::Dashboard,
                );
                self.previous_screen_before_settings = Some(Box::new(previous));

                let settings_screen = SettingsScreen::new(self.config.clone());
                self.current_screen = Screen::Settings(settings_screen);
            }
            ScreenAction::CloseSettings => {
                if let Some(previous) = self.previous_screen_before_settings.take() {
                    self.current_screen = *previous;
                } else {
                    self.refresh_dashboard_screen()?;
                }
            }
            ScreenAction::SaveSettings(config) => {
                match config.save() {
                    Ok(_) => {
                        self.config = config;
                        self.popup_message = Some(PopupMessage {
                            title: "Settings Saved".to_string(),
                            message: "Settings have been saved to ~/.config/kata-sr/config.toml".to_string(),
                            style: PopupStyle::Success,
                        });
                    }
                    Err(e) => {
                        self.popup_message = Some(PopupMessage {
                            title: "Error Saving Settings".to_string(),
                            message: format!("Failed to save settings:\n\n{}", e),
                            style: PopupStyle::Error,
                        });
                    }
                }
            }
            _ => {}
        }
        Ok(())
    }

    fn handle_misc_action(&mut self, action: ScreenAction) -> anyhow::Result<()> {
        match action {
            ScreenAction::PracticeEditorExited => {
                self.needs_terminal_clear = true;
            }
            ScreenAction::ViewDetails(kata, in_deck) => {
                let details = DetailsScreen::new(kata, in_deck);
                self.current_screen = Screen::Details(details);
            }
            ScreenAction::BackFromDetails => {
                let library = Library::load_with_filter(
                    &self.repo,
                    &self.config.library.default_sort,
                    self.config.library.default_sort_ascending,
                    self.library_hide_flagged,
                )?;
                self.current_screen = Screen::Library(library);
            }
            _ => {}
        }
        Ok(())
    }

    /// Resolves kata dependency names to their database IDs.
    fn resolve_dependency_ids(&self, dep_names: &[String]) -> anyhow::Result<Vec<i64>> {
        let mut dep_ids = Vec::new();
        for name in dep_names {
            if let Some(kata) = self.repo.get_kata_by_name(name)? {
                dep_ids.push(kata.id);
            }
        }
        Ok(dep_ids)
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

    fn refresh_dashboard_screen(&mut self) -> anyhow::Result<()> {
        if self.dashboard.katas_due.is_empty() {
            let next_review = self.repo.get_next_scheduled_review()?;
            let done_screen = DoneScreen::new(
                self.dashboard.stats.total_reviews_today,
                self.dashboard.stats.streak_days,
                next_review,
            );
            self.current_screen = Screen::Done(done_screen);
        } else {
            self.current_screen = Screen::Dashboard;
        }
        Ok(())
    }

    /// Handles rating submission by updating FSRS-5 state and creating a session record.
    ///
    /// # Arguments
    ///
    /// * `kata` - The kata that was reviewed
    /// * `rating` - User's quality rating (1-4 for FSRS: Again/Hard/Good/Easy)
    /// * `results` - Test results from the practice session
    fn handle_rating_submission(
        &mut self,
        kata: Kata,
        rating: u8,
        results: TestResults,
    ) -> anyhow::Result<()> {
        // Convert rating to FSRS Rating enum (1-4 scale)
        let fsrs_rating = Rating::from_int(rating as i32)
            .ok_or_else(|| anyhow::anyhow!("Invalid rating: {}", rating))?;

        // Get FSRS parameters (use default if none saved)
        let params = self
            .repo
            .get_latest_fsrs_params()
            .context("Failed to get FSRS params")?
            .unwrap_or_else(|| FsrsParams::default());

        // Get current FSRS card state and schedule next review
        let mut card = kata.fsrs_card();
        let now = Utc::now();
        card.schedule(fsrs_rating, &params, now);

        // Calculate next review date
        let next_review = now + Duration::days(card.scheduled_days as i64);

        // Update kata in database
        self.repo
            .update_kata_after_fsrs_review(kata.id, &card, next_review, now)
            .context("Failed to update kata after FSRS review")?;

        // Read the user's code attempt from the temp file
        let template_path = std::path::PathBuf::from(format!("/tmp/kata_{}.py", kata.id));
        let code_attempt = std::fs::read_to_string(&template_path).ok();

        // Serialize test results to JSON
        let test_results_json = serde_json::to_string(&results.results).ok();

        // Create session record
        let session = NewSession {
            kata_id: kata.id,
            started_at: now,
            completed_at: Some(now),
            test_results_json,
            num_passed: Some(results.num_passed as i32),
            num_failed: Some(results.num_failed as i32),
            num_skipped: Some(results.num_skipped as i32),
            duration_ms: Some(results.duration_ms as i64),
            quality_rating: Some(rating as i32),
            code_attempt,
        };

        self.repo
            .create_session(&session)
            .context("Failed to create session record")?;

        // Update daily statistics
        let analytics = Analytics::new(&self.repo);
        analytics
            .update_daily_stats()
            .context("Failed to update daily stats")?;

        Ok(())
    }

    /// Handles burying a kata, postponing it to the next day without affecting FSRS state.
    ///
    /// # Arguments
    ///
    /// * `kata` - The kata to bury
    fn handle_bury_kata(&mut self, kata: &Kata) -> anyhow::Result<()> {
        // Bury the kata by setting next_review_at to tomorrow
        self.repo
            .bury_kata(kata.id)
            .context("Failed to bury kata")?;

        Ok(())
    }

    /// Renders a popup overlay with a message to the user.
    fn render_popup_overlay(&self, frame: &mut Frame) {
        use ratatui::{
            style::{Color, Style},
            widgets::{Block, Borders, Clear, Paragraph, Wrap},
        };

        let popup_msg = match &self.popup_message {
            Some(msg) => msg,
            None => return,
        };

        // Create centered rectangle for popup (60% width, 40% height)
        let area = centered_rect(60, 40, frame.size());

        // Clear the area behind the popup
        frame.render_widget(Clear, area);

        // Determine colors based on popup style
        let (title_color, border_color) = match popup_msg.style {
            PopupStyle::Success => (Color::Green, Color::Green),
            PopupStyle::Error => (Color::Red, Color::Red),
        };

        // Render the popup
        let popup = Paragraph::new(popup_msg.message.clone())
            .wrap(Wrap { trim: false })
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(border_color))
                    .title(format!(" {} ", popup_msg.title))
                    .title_style(Style::default().fg(title_color)),
            );

        frame.render_widget(popup, area);

        // Render instructions at the bottom of popup
        let instructions_area = Rect {
            x: area.x,
            y: area.y + area.height - 2,
            width: area.width,
            height: 1,
        };

        let instructions =
            Paragraph::new("Press any key to dismiss").style(Style::default().fg(Color::Gray));
        frame.render_widget(instructions, instructions_area);
    }
}

/// Creates a centered rectangle for popup overlays.
///
/// # Arguments
///
/// * `percent_x` - Width as percentage of parent area
/// * `percent_y` - Height as percentage of parent area
/// * `r` - Parent rectangle
fn centered_rect(percent_x: u16, percent_y: u16, r: Rect) -> Rect {
    use ratatui::layout::{Constraint, Direction, Layout};

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
