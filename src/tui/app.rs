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

use super::create_kata::{CreateKataAction, CreateKataScreen};
use super::dashboard::{Dashboard, DashboardAction};
use super::details::{DetailsAction, DetailsScreen};
use super::done::{DoneAction, DoneScreen};
use super::edit_kata::{EditKataAction, EditKataScreen};
use super::keybindings;
use super::library::{Library, LibraryAction};
use super::practice::{PracticeAction, PracticeScreen};
use super::results::{ResultsAction, ResultsScreen};
use super::session_detail::{SessionDetailAction, SessionDetailScreen};
use super::session_history::{SessionHistoryAction, SessionHistoryScreen};
use super::settings::{SettingsAction, SettingsScreen};
use super::startup::{StartupAction, StartupScreen};
use crate::config::AppConfig;
use crate::core::analytics::Analytics;
use crate::core::fsrs::{FsrsParams, Rating};
use crate::db::repo::{Kata, KataRepository, NewKata, NewSession};
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

/// Current screen being displayed in the TUI.
///
/// Each screen maintains its own state and handles its own rendering.
pub enum Screen {
    /// Startup screen with ASCII art
    Startup(StartupScreen),
    /// Dashboard showing katas due today and stats
    Dashboard,
    /// Celebration screen when no reviews are due
    Done(DoneScreen),
    /// Practice screen for a specific kata
    Practice(Kata, PracticeScreen),
    /// Results screen showing test output and rating selection
    Results(Kata, ResultsScreen),
    /// Help screen showing keybindings
    Help,
    /// Settings screen for editing configuration
    Settings(SettingsScreen),
    /// Library screen for browsing and adding katas
    Library(Library),
    /// Details screen for viewing kata information
    Details(DetailsScreen),
    /// Create kata screen for generating new kata files
    CreateKata(CreateKataScreen),
    /// Edit kata screen for modifying existing katas
    EditKata(EditKataScreen),
    /// Session history screen showing past sessions for a kata
    SessionHistory(SessionHistoryScreen),
    /// Session detail screen showing full test results for a session
    SessionDetail(SessionDetailScreen),
    /// Course library screen for browsing available courses
    CourseLibrary(super::course_library::CourseLibrary),
    /// Course detail screen for viewing course sections
    CourseDetail(super::course_detail::CourseDetail),
}

/// Internal enum for handling screen transitions without borrow conflicts.
enum ScreenAction {
    StartPractice(Kata),
    AttemptKataWithoutDeck(crate::core::kata_loader::AvailableKata),
    ReturnToDashboard,
    SubmitRating(Kata, u8, TestResults),
    BuryKata(Kata),
    StartNextDue,
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
    OpenEditKata(i64),
    SubmitEditKata {
        kata_id: i64,
        original_slug: String,
        form_data: crate::core::kata_generator::KataFormData,
        new_slug: String,
    },
    OpenEditorFile(std::path::PathBuf),
    CancelEditKata,
    OpenSettings,
    CloseSettings,
    ViewSessionHistory(Kata),
    ViewSessionDetail(i64), // session_id
    DeleteSession(i64),     // session_id
    BackFromSessionHistory,
    BackFromSessionDetail,
    OpenCourseLibrary,
    SelectCourse(crate::db::repo::Course),
    ViewCourseHTML(String), // html_path
    BackFromCourseLibrary,
    BackFromCourseDetail,
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

        match &mut self.current_screen {
            Screen::Startup(startup_screen) => {
                startup_screen.render(frame);
            }
            Screen::Dashboard => {
                self.dashboard.render(frame);
            }
            Screen::Done(done_screen) => {
                done_screen.render(frame);
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
            Screen::Settings(settings_screen) => {
                settings_screen.render(frame);
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
            Screen::EditKata(edit_kata) => {
                edit_kata.render(frame);
            }
            Screen::SessionHistory(session_history) => {
                session_history.render(frame);
            }
            Screen::SessionDetail(session_detail) => {
                session_detail.render(frame);
            }
            Screen::CourseLibrary(course_library) => {
                course_library.render(frame);
            }
            Screen::CourseDetail(course_detail) => {
                course_detail.render(frame);
            }
        }

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
        let action_result = match &mut self.current_screen {
            Screen::Startup(startup_screen) => {
                let action = startup_screen.handle_input(code);
                match action {
                    StartupAction::StartReview => Some(ScreenAction::ReturnToDashboard),
                    StartupAction::OpenLibrary => Some(ScreenAction::OpenLibrary),
                    StartupAction::OpenSettings => Some(ScreenAction::OpenSettings),
                    StartupAction::None => None,
                }
            }
            Screen::Dashboard => {
                // handle library key 'l' in dashboard
                if code == KeyCode::Char('l') {
                    return self.execute_action(ScreenAction::OpenLibrary);
                }
                // handle courses key 'c' in dashboard
                if code == KeyCode::Char('c') {
                    return self.execute_action(ScreenAction::OpenCourseLibrary);
                }
                // handle settings key 's' in dashboard
                if code == KeyCode::Char('s') {
                    return self.execute_action(ScreenAction::OpenSettings);
                }
                // handle history key 'h' in dashboard (only if kata is selected)
                if code == KeyCode::Char('h') && !self.dashboard.katas_due.is_empty() {
                    if let Some(kata) = self.dashboard.katas_due.get(self.dashboard.selected_index)
                    {
                        return self.execute_action(ScreenAction::ViewSessionHistory(kata.clone()));
                    }
                }
                let action = self.dashboard.handle_input(code);
                match action {
                    DashboardAction::SelectKata(kata) => Some(ScreenAction::StartPractice(kata)),
                    DashboardAction::RemoveKata(kata) => {
                        Some(ScreenAction::RemoveKataFromDeck(kata))
                    }
                    DashboardAction::EditKata(kata) => Some(ScreenAction::OpenEditKata(kata.id)),
                    DashboardAction::ToggleFlagKata(kata) => {
                        // Toggle the problematic flag in database
                        if kata.is_problematic {
                            self.repo.unflag_kata(kata.id)?;
                        } else {
                            self.repo.flag_kata(kata.id, None)?;
                        }
                        // Reload dashboard to get fresh kata state (preserve hide_flagged state)
                        self.dashboard = Dashboard::load_with_filter(
                            &self.repo,
                            self.config.display.heatmap_days,
                            self.dashboard_hide_flagged,
                        )?;
                        None
                    }
                    DashboardAction::ToggleHideFlagged => {
                        // Toggle the hide_flagged filter and reload dashboard
                        self.dashboard_hide_flagged = !self.dashboard.hide_flagged;
                        self.dashboard = Dashboard::load_with_filter(
                            &self.repo,
                            self.config.display.heatmap_days,
                            self.dashboard_hide_flagged,
                        )?;
                        None
                    }
                    DashboardAction::None => None,
                }
            }
            Screen::Practice(kata, practice_screen) => {
                // Handle 'f' to toggle flag on current kata
                if code == KeyCode::Char('f') {
                    let kata_id = kata.id;
                    let was_problematic = kata.is_problematic;

                    // Toggle the flag in database
                    if was_problematic {
                        self.repo.unflag_kata(kata_id)?;
                    } else {
                        self.repo.flag_kata(kata_id, None)?;
                    }

                    // Reload fresh kata state from database and update Screen enum
                    if let Some(fresh_kata) = self.repo.get_kata_by_id(kata_id)? {
                        // Extract the practice_screen by temporarily replacing current_screen
                        let old_screen =
                            std::mem::replace(&mut self.current_screen, Screen::Dashboard);
                        if let Screen::Practice(_, ps) = old_screen {
                            self.current_screen = Screen::Practice(fresh_kata, ps);
                        }
                    }

                    // Also reload dashboard for consistency (preserve hide_flagged state)
                    self.dashboard = Dashboard::load_with_filter(
                        &self.repo,
                        self.config.display.heatmap_days,
                        self.dashboard_hide_flagged,
                    )?;
                    return Ok(());
                }

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
            Screen::Done(done_screen) => match done_screen.handle_input(code) {
                DoneAction::BrowseLibrary => Some(ScreenAction::OpenLibrary),
                DoneAction::ToggleHideFlagged => {
                    // Toggle the hide_flagged filter and reload dashboard
                    self.dashboard_hide_flagged = !self.dashboard_hide_flagged;
                    self.dashboard = Dashboard::load_with_filter(
                        &self.repo,
                        self.config.display.heatmap_days,
                        self.dashboard_hide_flagged,
                    )?;
                    // Refresh the screen - if there are now katas due, show dashboard; otherwise stay on Done
                    self.refresh_dashboard_screen()?;
                    None
                }
                DoneAction::None => None,
            },
            Screen::Results(kata, results_screen) => {
                let action = results_screen.handle_input(code);
                match action {
                    ResultsAction::SubmitRating(rating) => Some(ScreenAction::SubmitRating(
                        kata.clone(),
                        rating,
                        results_screen.get_results().clone(),
                    )),
                    ResultsAction::BuryCard => Some(ScreenAction::BuryKata(kata.clone())),
                    ResultsAction::Retry => Some(ScreenAction::RetryKata(kata.clone())),
                    ResultsAction::GiveUp => {
                        // Open solution in editor
                        match results_screen.open_solution_in_editor(true) {
                            Ok(()) => {
                                // Successfully viewed solution, signal terminal clear needed
                                self.needs_terminal_clear = true;

                                // Check if this is preview mode (kata not in deck)
                                if kata.id == -1 {
                                    // Preview mode: don't save rating, just return to library
                                    Some(ScreenAction::OpenLibrary)
                                } else {
                                    // Normal mode: auto-submit Rating::Again (1)
                                    Some(ScreenAction::SubmitRating(
                                        kata.clone(),
                                        1,
                                        results_screen.get_results().clone(),
                                    ))
                                }
                            }
                            Err(e) => {
                                // Failed to open editor - stay on results screen
                                eprintln!("Failed to open solution: {}", e);
                                None
                            }
                        }
                    }
                    ResultsAction::SolutionViewed => {
                        // External editor was used, signal terminal needs clearing
                        self.needs_terminal_clear = true;
                        None
                    }
                    ResultsAction::StartNextDue => Some(ScreenAction::StartNextDue),
                    ResultsAction::ReviewAnother => Some(ScreenAction::ReturnToDashboard),
                    ResultsAction::BackToDashboard => Some(ScreenAction::ReturnToDashboard),
                    ResultsAction::BackToLibrary => Some(ScreenAction::OpenLibrary),
                    ResultsAction::OpenSettings => Some(ScreenAction::OpenSettings),
                    ResultsAction::ToggleFlagWithReason(reason) => {
                        let kata_id = kata.id;
                        let was_problematic = kata.is_problematic;

                        // Toggle the flag in database with reason
                        if was_problematic {
                            self.repo.unflag_kata(kata_id)?;
                        } else {
                            self.repo.flag_kata(kata_id, reason)?;
                        }

                        // Reload fresh kata state from database and update Screen enum
                        if let Some(fresh_kata) = self.repo.get_kata_by_id(kata_id)? {
                            // Extract the results_screen by temporarily replacing current_screen
                            let old_screen =
                                std::mem::replace(&mut self.current_screen, Screen::Dashboard);
                            if let Screen::Results(_, mut rs) = old_screen {
                                // Update the kata inside ResultsScreen so flag popup shows correct status
                                rs.update_kata(fresh_kata.clone());
                                self.current_screen = Screen::Results(fresh_kata, rs);
                            }
                        }

                        // Also reload dashboard for consistency (preserve hide_flagged state)
                        self.dashboard = Dashboard::load_with_filter(
                            &self.repo,
                            self.config.display.heatmap_days,
                            self.dashboard_hide_flagged,
                        )?;
                        None
                    }
                    ResultsAction::None => None,
                }
            }
            Screen::Help => Some(ScreenAction::ReturnToDashboard),
            Screen::Settings(settings_screen) => {
                let action = settings_screen.handle_input(code);
                match action {
                    SettingsAction::Cancel => Some(ScreenAction::CloseSettings),
                    SettingsAction::Save(config) => {
                        // Save config to file
                        match config.save() {
                            Ok(_) => {
                                // Update app config
                                self.config = config;
                                // Show success popup
                                self.popup_message = Some(PopupMessage {
                                    title: "Settings Saved".to_string(),
                                    message:
                                        "Settings have been saved to ~/.config/kata-sr/config.toml"
                                            .to_string(),
                                    style: PopupStyle::Success,
                                });
                                None
                            }
                            Err(e) => {
                                // Show error popup
                                self.popup_message = Some(PopupMessage {
                                    title: "Error Saving Settings".to_string(),
                                    message: format!("Failed to save settings:\n\n{}", e),
                                    style: PopupStyle::Error,
                                });
                                None
                            }
                        }
                    }
                    SettingsAction::None => None,
                }
            }
            Screen::Library(library) => {
                // handle history key 'h' in My Deck tab
                if code == KeyCode::Char('h') {
                    use super::library::LibraryTab;
                    if library.active_tab == LibraryTab::MyDeck
                        && library.deck_selected < library.deck_katas.len()
                    {
                        let kata = library.deck_katas[library.deck_selected].clone();
                        Some(ScreenAction::ViewSessionHistory(kata))
                    } else {
                        None
                    }
                } else {
                    let action = library.handle_input(code);
                    match action {
                        LibraryAction::AddKata(name) => {
                            Some(ScreenAction::AddKataFromLibrary(name))
                        }
                        LibraryAction::AttemptKata(available_kata) => {
                            Some(ScreenAction::AttemptKataWithoutDeck(available_kata))
                        }
                        LibraryAction::PracticeKata(kata) => {
                            Some(ScreenAction::StartPractice(kata))
                        }
                        LibraryAction::RemoveKata(kata) => {
                            Some(ScreenAction::RemoveKataFromDeck(kata))
                        }
                        LibraryAction::ToggleFlagKata(kata) => {
                            // Toggle the problematic flag in database (deprecated - kept for backward compatibility)
                            if kata.is_problematic {
                                self.repo.unflag_kata(kata.id)?;
                            } else {
                                self.repo.flag_kata(kata.id, None)?;
                            }
                            // Reload dashboard for consistency (preserve hide_flagged state)
                            self.dashboard = Dashboard::load_with_filter(
                                &self.repo,
                                self.config.display.heatmap_days,
                                self.dashboard_hide_flagged,
                            )?;
                            // Reload library with fresh kata states
                            return self.execute_action(ScreenAction::OpenLibrary);
                        }
                        LibraryAction::ToggleHideFlagged => {
                            // Toggle the hide_flagged filter and reload library
                            self.library_hide_flagged = !library.hide_flagged;
                            let new_library = Library::load_with_filter(
                                &self.repo,
                                &self.config.library.default_sort,
                                self.config.library.default_sort_ascending,
                                self.library_hide_flagged,
                            )?;
                            self.current_screen = Screen::Library(new_library);
                            None
                        }
                        LibraryAction::ToggleFlagWithReason(kata, reason) => {
                            // Toggle the problematic flag in database with reason
                            if kata.is_problematic {
                                self.repo.unflag_kata(kata.id)?;
                            } else {
                                self.repo.flag_kata(kata.id, reason)?;
                            }
                            // Reload dashboard for consistency (preserve hide_flagged state)
                            self.dashboard = Dashboard::load_with_filter(
                                &self.repo,
                                self.config.display.heatmap_days,
                                self.dashboard_hide_flagged,
                            )?;
                            // Reload library with fresh kata states
                            return self.execute_action(ScreenAction::OpenLibrary);
                        }
                        LibraryAction::Back => Some(ScreenAction::BackFromLibrary),
                        LibraryAction::ViewDetails(kata) => {
                            let in_deck = library.kata_ids_in_deck.contains(&kata.name);
                            Some(ScreenAction::ViewDetails(kata, in_deck))
                        }
                        LibraryAction::CreateKata => Some(ScreenAction::OpenCreateKata),
                        LibraryAction::EditKataById(kata_id) => {
                            Some(ScreenAction::OpenEditKata(kata_id))
                        }
                        LibraryAction::EditKataByName(name) => {
                            // Look up kata by name to get ID
                            if let Ok(Some(kata)) = self.repo.get_kata_by_name(&name) {
                                Some(ScreenAction::OpenEditKata(kata.id))
                            } else {
                                None
                            }
                        }
                        LibraryAction::None => None,
                    }
                }
            }
            Screen::Details(details) => {
                let action = details.handle_input(code);
                match action {
                    DetailsAction::AddKata(name) => Some(ScreenAction::AddKataFromLibrary(name)),
                    DetailsAction::Back => Some(ScreenAction::BackFromDetails),
                    DetailsAction::EditKata(name) => {
                        // Look up kata by name to get ID
                        if let Ok(Some(kata)) = self.repo.get_kata_by_name(&name) {
                            Some(ScreenAction::OpenEditKata(kata.id))
                        } else {
                            None
                        }
                    }
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
            Screen::EditKata(edit_kata) => {
                let exercises_dir = std::path::Path::new("katas/exercises");
                let action = edit_kata.handle_input(code, exercises_dir);
                match action {
                    EditKataAction::Submit {
                        kata_id,
                        original_slug,
                        form_data,
                        new_slug,
                    } => Some(ScreenAction::SubmitEditKata {
                        kata_id,
                        original_slug,
                        form_data,
                        new_slug,
                    }),
                    EditKataAction::OpenEditor { file_path } => {
                        Some(ScreenAction::OpenEditorFile(file_path))
                    }
                    EditKataAction::Cancel => Some(ScreenAction::CancelEditKata),
                    EditKataAction::None => None,
                }
            }
            Screen::SessionHistory(session_history) => {
                let action = session_history.handle_input(code);
                match action {
                    SessionHistoryAction::ViewDetails(session_id) => {
                        Some(ScreenAction::ViewSessionDetail(session_id))
                    }
                    SessionHistoryAction::Delete(session_id) => {
                        Some(ScreenAction::DeleteSession(session_id))
                    }
                    SessionHistoryAction::Back => Some(ScreenAction::BackFromSessionHistory),
                    SessionHistoryAction::None => None,
                }
            }
            Screen::SessionDetail(session_detail) => {
                let action = session_detail.handle_input(code);
                match action {
                    SessionDetailAction::Back => Some(ScreenAction::BackFromSessionDetail),
                    SessionDetailAction::None => None,
                }
            }
            Screen::CourseLibrary(course_library) => {
                use super::course_library::CourseLibraryAction;
                let action = course_library.handle_input(code);
                match action {
                    CourseLibraryAction::SelectCourse(course) => {
                        Some(ScreenAction::SelectCourse(course))
                    }
                    CourseLibraryAction::ViewDetails(_course) => {
                        // TODO: Show course details in a popup or separate screen
                        None
                    }
                    CourseLibraryAction::Back => Some(ScreenAction::BackFromCourseLibrary),
                    CourseLibraryAction::None => None,
                }
            }
            Screen::CourseDetail(course_detail) => {
                use super::course_detail::CourseDetailAction;
                let action = course_detail.handle_input(code);
                match action {
                    CourseDetailAction::ViewSection(section) => {
                        Some(ScreenAction::ViewCourseHTML(section.html_path.clone()))
                    }
                    CourseDetailAction::PracticeExercise(kata_name) => {
                        // Look up kata by name and start practice
                        if let Ok(Some(kata)) = self.repo.get_kata_by_name(&kata_name) {
                            Some(ScreenAction::StartPractice(kata))
                        } else {
                            None
                        }
                    }
                    CourseDetailAction::NextSection | CourseDetailAction::PreviousSection => {
                        // Update progress after navigation
                        if let Some(section) = course_detail.get_current_section() {
                            let _ = self.repo.upsert_course_progress(
                                &crate::db::repo::NewCourseProgress {
                                    course_id: course_detail.course.id,
                                    last_section_id: Some(section.id),
                                },
                            );
                        }
                        None
                    }
                    CourseDetailAction::Back => Some(ScreenAction::BackFromCourseDetail),
                    CourseDetailAction::None => None,
                }
            }
        };

        // handle the extracted action (no longer borrowing self.current_screen)
        if let Some(action) = action_result {
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
            ScreenAction::StartPractice(kata) => {
                let practice_screen =
                    PracticeScreen::new(kata.clone(), self.config.editor.clone())?;
                self.current_screen = Screen::Practice(kata, practice_screen);
            }
            ScreenAction::AttemptKataWithoutDeck(available_kata) => {
                // Create a temporary Kata object for preview mode (id = -1)
                let preview_kata = Kata {
                    id: -1, // Special ID to indicate preview mode
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
            ScreenAction::ReturnToDashboard => {
                self.dashboard = Dashboard::load_with_filter(
                    &self.repo,
                    self.config.display.heatmap_days,
                    self.dashboard_hide_flagged,
                )?;
                self.refresh_dashboard_screen()?;
            }
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
                            // Force terminal clear to prevent display corruption
                            self.needs_terminal_clear = true;
                        }
                        Screen::Details(_) => {
                            // navigate back to library with updated state
                            let library = Library::load_with_filter(
                                &self.repo,
                                &self.config.library.default_sort,
                                self.config.library.default_sort_ascending,
                                self.library_hide_flagged,
                            )?;
                            self.current_screen = Screen::Library(library);
                            // Force terminal clear to prevent display corruption
                            self.needs_terminal_clear = true;
                        }
                        _ => {}
                    }

                    // reload dashboard so counts are updated (preserve hide_flagged state)
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
            ScreenAction::RetryKata(kata) => {
                let practice_screen =
                    PracticeScreen::new_retry(kata.clone(), self.config.editor.clone())?;
                self.current_screen = Screen::Practice(kata, practice_screen);
            }
            ScreenAction::RemoveKataFromDeck(kata) => {
                // Delete the kata from the database
                self.repo.delete_kata(kata.id)?;

                // Reload dashboard to reflect the change (preserve hide_flagged state)
                self.dashboard = Dashboard::load_with_filter(
                    &self.repo,
                    self.config.display.heatmap_days,
                    self.dashboard_hide_flagged,
                )?;

                // If on library screen, update library state
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
            ScreenAction::OpenCreateKata => {
                // Load available katas for dependency selection
                let available_katas = crate::core::kata_loader::load_available_katas()?;
                let kata_names: Vec<String> =
                    available_katas.iter().map(|k| k.name.clone()).collect();

                let create_kata_screen = CreateKataScreen::new(kata_names);
                self.current_screen = Screen::CreateKata(create_kata_screen);
            }
            ScreenAction::SubmitNewKata { form_data, slug: _ } => {
                use crate::core::kata_generator::generate_kata_files;

                let exercises_dir = std::path::Path::new("katas/exercises");

                // Generate the kata files
                match generate_kata_files(&form_data, exercises_dir) {
                    Ok(created_slug) => {
                        // Success! Return to library and show success popup
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
                        // Error! Return to library and show error popup
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
                // Return to library
                let library = Library::load_with_filter(
                    &self.repo,
                    &self.config.library.default_sort,
                    self.config.library.default_sort_ascending,
                    self.library_hide_flagged,
                )?;
                self.current_screen = Screen::Library(library);
            }
            ScreenAction::OpenEditKata(kata_id) => {
                // Load kata from database
                if let Some(kata) = self.repo.get_kata_by_id(kata_id)? {
                    // Load tags and dependencies
                    let tags = self.repo.get_kata_tags(kata_id)?;
                    let dependencies = self.repo.get_kata_dependencies(kata_id)?;

                    // Get dependency names (not IDs)
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

                    // Load available katas for dependency selection
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

                // Get original state for rollback
                let original_kata = self
                    .repo
                    .get_kata_by_id(kata_id)?
                    .ok_or_else(|| anyhow::anyhow!("Kata not found"))?;
                let original_dependencies = self.repo.get_kata_dependencies(kata_id)?;
                let original_tags = self.repo.get_kata_tags(kata_id)?;

                // Parse category field as comma-separated tags
                let new_tags: Vec<String> = form_data
                    .category
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect();

                // Use the first tag as the primary category for backward compatibility
                let primary_category = new_tags
                    .first()
                    .cloned()
                    .unwrap_or_else(|| form_data.category.clone());

                if name_changed {
                    // 1. Update database first
                    match self.repo.update_kata_full_metadata(
                        kata_id,
                        &new_slug,
                        &form_data.description,
                        &primary_category,
                        form_data.difficulty as i32,
                    ) {
                        Ok(_) => {
                            // 2. Update tags in DB
                            if let Err(e) = self.repo.set_kata_tags(kata_id, &new_tags) {
                                // Rollback metadata changes
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

                            // 3. Update dependencies in DB
                            let dep_ids = self.resolve_dependency_ids(&form_data.dependencies)?;
                            if let Err(e) = self.repo.replace_dependencies(kata_id, &dep_ids) {
                                // Rollback ALL database changes
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

                            // 4. Now rename directory (if this fails, rollback DB)
                            match rename_kata_directory(exercises_dir, &original_slug, &new_slug) {
                                Ok(_) => {
                                    // 5. Update manifest in new location
                                    let kata_dir = exercises_dir.join(&new_slug);
                                    if let Err(e) =
                                        update_manifest(&kata_dir, &form_data, &new_slug, &new_tags)
                                    {
                                        // Rollback: rename directory back and revert ALL database changes
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

                                    // 6. Update all dependent kata manifests to use new slug
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
                                                // Rollback: rename directory back, revert DB, and rollback any
                                                // already-updated dependent manifests
                                                eprintln!("Failed to update dependent manifest for '{}': {}", dep_kata.name, e);

                                                // Try to rollback dependent manifests we already updated
                                                for rollback_id in &dependent_kata_ids {
                                                    if rollback_id == dep_kata_id {
                                                        break; // Don't rollback the one that failed
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

                                                // Rollback renamed kata
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

                                    // Success! Return to library
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
                                    // Rollback ALL database changes
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
                                    // Stay on edit screen
                                }
                            }
                        }
                        Err(e) => {
                            eprintln!("Failed to update database: {}", e);
                            // Stay on edit screen (nothing to rollback - first operation failed)
                        }
                    }
                } else {
                    // Name didn't change, just update metadata
                    match self.repo.update_kata_metadata(
                        kata_id,
                        &form_data.description,
                        &primary_category,
                        form_data.difficulty as i32,
                    ) {
                        Ok(_) => {
                            // Update tags in DB
                            if let Err(e) = self.repo.set_kata_tags(kata_id, &new_tags) {
                                // Rollback metadata changes
                                let _ = self.repo.update_kata_metadata(
                                    kata_id,
                                    &original_kata.description,
                                    &original_kata.category,
                                    original_kata.base_difficulty,
                                );
                                eprintln!("Failed to update tags: {}", e);
                                return Ok(());
                            }

                            // Update dependencies
                            let dep_ids = self.resolve_dependency_ids(&form_data.dependencies)?;
                            if let Err(e) = self.repo.replace_dependencies(kata_id, &dep_ids) {
                                // Rollback metadata changes
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

                            // Update manifest with new tags
                            let kata_dir = exercises_dir.join(&original_slug);
                            if let Err(e) =
                                update_manifest(&kata_dir, &form_data, &original_slug, &new_tags)
                            {
                                // Rollback ALL database changes
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

                            // Success! Return to library
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
                            // Stay on edit screen (nothing to rollback - first operation failed)
                        }
                    }
                }
            }
            ScreenAction::OpenEditorFile(file_path) => {
                // Spawn external editor with proper terminal handling
                use std::io::Write;
                use std::process::Command;

                let editor = std::env::var("EDITOR").unwrap_or_else(|_| "nvim".to_string());

                // Exit alternate screen and disable raw mode to hand control to editor
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

                // Launch editor and wait for it to complete
                let status_result = Command::new(&editor).arg(&file_path).status();

                // Re-enable raw mode and re-enter alternate screen to restore TUI
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

                // Flush to ensure all commands are executed
                stdout
                    .flush()
                    .context("Failed to flush stdout after terminal reset")?;

                // Check editor exit status
                let editor_status =
                    status_result.with_context(|| format!("Failed to launch {}", editor))?;

                if !editor_status.success() {
                    eprintln!(
                        "{} exited with non-zero status (code {:?})",
                        editor,
                        editor_status.code()
                    );
                }

                // Stay on edit screen
            }
            ScreenAction::CancelEditKata => {
                // Return to library
                let library = Library::load_with_filter(
                    &self.repo,
                    &self.config.library.default_sort,
                    self.config.library.default_sort_ascending,
                    self.library_hide_flagged,
                )?;
                self.current_screen = Screen::Library(library);
            }
            ScreenAction::OpenSettings => {
                // Save current screen before opening settings so we can restore it later
                let previous = std::mem::replace(
                    &mut self.current_screen,
                    Screen::Dashboard, // temporary placeholder
                );
                self.previous_screen_before_settings = Some(Box::new(previous));

                let settings_screen = SettingsScreen::new(self.config.clone());
                self.current_screen = Screen::Settings(settings_screen);
            }
            ScreenAction::CloseSettings => {
                // Restore previous screen if we have one, otherwise go to dashboard
                if let Some(previous) = self.previous_screen_before_settings.take() {
                    self.current_screen = *previous;
                } else {
                    self.refresh_dashboard_screen()?;
                }
            }
            ScreenAction::ViewSessionHistory(kata) => {
                let session_history = SessionHistoryScreen::new(kata, &self.repo)?;
                self.current_screen = Screen::SessionHistory(session_history);
            }
            ScreenAction::ViewSessionDetail(session_id) => {
                let session_detail = SessionDetailScreen::new(session_id, &self.repo)?;
                self.current_screen = Screen::SessionDetail(session_detail);
            }
            ScreenAction::DeleteSession(session_id) => {
                // Delete the session from the database
                self.repo.delete_session(session_id)?;

                // Refresh the session history screen to show updated list
                if let Screen::SessionHistory(ref session_history) = &self.current_screen {
                    let kata = session_history.kata.clone();
                    let mut new_session_history = SessionHistoryScreen::new(kata, &self.repo)?;
                    // Preserve selection, adjusting if we deleted the last item
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
                // Navigate back to session history
                // We need to get the kata from the session detail screen
                if let Screen::SessionDetail(ref session_detail) = &self.current_screen {
                    let kata = self
                        .repo
                        .get_kata_by_id(session_detail.session.kata_id)?
                        .ok_or_else(|| anyhow::anyhow!("Kata not found"))?;
                    let session_history = SessionHistoryScreen::new(kata, &self.repo)?;
                    self.current_screen = Screen::SessionHistory(session_history);
                }
            }
            ScreenAction::OpenCourseLibrary => {
                use super::course_library::CourseLibrary;
                let course_library = CourseLibrary::load(&self.repo)?;
                self.current_screen = Screen::CourseLibrary(course_library);
            }
            ScreenAction::SelectCourse(course) => {
                use super::course_detail::CourseDetail;
                let course_detail = CourseDetail::load(&self.repo, course)?;
                self.current_screen = Screen::CourseDetail(course_detail);
            }
            ScreenAction::ViewCourseHTML(html_path) => {
                // Open HTML file in default browser
                self.open_html_in_browser(&html_path)?;
                self.needs_terminal_clear = true;
            }
            ScreenAction::BackFromCourseLibrary => {
                self.refresh_dashboard_screen()?;
            }
            ScreenAction::BackFromCourseDetail => {
                use super::course_library::CourseLibrary;
                let course_library = CourseLibrary::load(&self.repo)?;
                self.current_screen = Screen::CourseLibrary(course_library);
            }
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

    /// Opens an HTML file in the default browser.
    ///
    /// Supports multiple platforms (Linux, macOS, Windows).
    fn open_html_in_browser(&self, html_path: &str) -> anyhow::Result<()> {
        use std::process::Command;

        // Determine the command to open the browser based on the platform
        #[cfg(target_os = "linux")]
        let open_command = "xdg-open";
        #[cfg(target_os = "macos")]
        let open_command = "open";
        #[cfg(target_os = "windows")]
        let open_command = "start";

        // Execute the command
        Command::new(open_command)
            .arg(html_path)
            .spawn()
            .with_context(|| format!("Failed to open HTML file: {}", html_path))?;

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
