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
}

/// Internal enum for handling screen transitions without borrow conflicts.
enum ScreenAction {
    StartPractice(Kata),
    ReturnToDashboard,
    SubmitRating(Kata, u8),
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
    BackFromSessionHistory,
    BackFromSessionDetail,
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
        let dashboard = Dashboard::load(&repo)?;

        let app = Self {
            current_screen: Screen::Startup(StartupScreen::new()),
            dashboard,
            repo,
            config,
            event_tx: tx,
            event_rx: rx,
            showing_help: false,
            needs_terminal_clear: false,
            popup_message: None,
            previous_screen_before_settings: None,
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
                // handle settings key 's' in dashboard
                if code == KeyCode::Char('s') {
                    return self.execute_action(ScreenAction::OpenSettings);
                }
                // handle history key 'h' in dashboard (only if kata is selected)
                if code == KeyCode::Char('h') && !self.dashboard.katas_due.is_empty() {
                    if let Some(kata) = self.dashboard.katas_due.get(self.dashboard.selected_index) {
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
                        // Reload dashboard to get fresh kata state
                        self.dashboard = Dashboard::load(&self.repo)?;
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
                        let old_screen = std::mem::replace(&mut self.current_screen, Screen::Dashboard);
                        if let Screen::Practice(_, ps) = old_screen {
                            self.current_screen = Screen::Practice(fresh_kata, ps);
                        }
                    }

                    // Also reload dashboard for consistency
                    self.dashboard = Dashboard::load(&self.repo)?;
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
                DoneAction::None => None,
            },
            Screen::Results(kata, results_screen) => {
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
                        // Extract the results_screen by temporarily replacing current_screen
                        let old_screen = std::mem::replace(&mut self.current_screen, Screen::Dashboard);
                        if let Screen::Results(_, rs) = old_screen {
                            self.current_screen = Screen::Results(fresh_kata, rs);
                        }
                    }

                    // Also reload dashboard for consistency
                    self.dashboard = Dashboard::load(&self.repo)?;
                    return Ok(());
                }

                let action = results_screen.handle_input(code);
                match action {
                    ResultsAction::SubmitRating(rating) => {
                        Some(ScreenAction::SubmitRating(kata.clone(), rating))
                    }
                    ResultsAction::BuryCard => Some(ScreenAction::BuryKata(kata.clone())),
                    ResultsAction::Retry => Some(ScreenAction::RetryKata(kata.clone())),
                    ResultsAction::GiveUp => {
                        // Load and show solution. When user closes it (Esc), it auto-submits Rating::Again
                        results_screen.load_and_show_solution(true);
                        None
                    }
                    ResultsAction::StartNextDue => Some(ScreenAction::StartNextDue),
                    ResultsAction::ReviewAnother => Some(ScreenAction::ReturnToDashboard),
                    ResultsAction::BackToDashboard => Some(ScreenAction::ReturnToDashboard),
                    ResultsAction::OpenSettings => Some(ScreenAction::OpenSettings),
                    ResultsAction::None => None,
                }
            }
            Screen::Help => Some(ScreenAction::ReturnToDashboard),
            Screen::Settings(settings_screen) => {
                let action = settings_screen.handle_input(code);
                match action {
                    SettingsAction::Cancel => Some(ScreenAction::CloseSettings),
                    SettingsAction::Save => {
                        // TODO: implement save
                        Some(ScreenAction::CloseSettings)
                    }
                    SettingsAction::None => None,
                }
            }
            Screen::Library(library) => {
                // handle history key 'h' in My Deck tab
                if code == KeyCode::Char('h') {
                    use super::library::LibraryTab;
                    if library.active_tab == LibraryTab::MyDeck && library.deck_selected < library.deck_katas.len() {
                        let kata = library.deck_katas[library.deck_selected].clone();
                        Some(ScreenAction::ViewSessionHistory(kata))
                    } else {
                        None
                    }
                } else {
                    let action = library.handle_input(code);
                    match action {
                        LibraryAction::AddKata(name) => Some(ScreenAction::AddKataFromLibrary(name)),
                        LibraryAction::RemoveKata(kata) => Some(ScreenAction::RemoveKataFromDeck(kata)),
                        LibraryAction::ToggleFlagKata(kata) => {
                            // Toggle the problematic flag in database
                            if kata.is_problematic {
                                self.repo.unflag_kata(kata.id)?;
                            } else {
                                self.repo.flag_kata(kata.id, None)?;
                            }
                            // Reload dashboard for consistency
                            self.dashboard = Dashboard::load(&self.repo)?;
                            // Reload library with fresh kata states
                            return self.execute_action(ScreenAction::OpenLibrary);
                        }
                        LibraryAction::Back => Some(ScreenAction::BackFromLibrary),
                        LibraryAction::ViewDetails(kata) => {
                            let in_deck = library.kata_ids_in_deck.contains(&kata.name);
                            Some(ScreenAction::ViewDetails(kata, in_deck))
                        }
                        LibraryAction::CreateKata => Some(ScreenAction::OpenCreateKata),
                        LibraryAction::EditKataById(kata_id) => Some(ScreenAction::OpenEditKata(kata_id)),
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
                self.refresh_dashboard_screen()?;
            }
            ScreenAction::SubmitRating(kata, rating) => {
                self.handle_rating_submission(kata, rating)?;
                self.dashboard = Dashboard::load(&self.repo)?;
                if let Screen::Results(_, results_screen) = &mut self.current_screen {
                    results_screen.mark_rating_submitted(rating, self.dashboard.katas_due.len());
                }
            }
            ScreenAction::BuryKata(kata) => {
                self.handle_bury_kata(&kata)?;
                self.dashboard = Dashboard::load(&self.repo)?;
                self.refresh_dashboard_screen()?;
            }
            ScreenAction::StartNextDue => {
                if self.dashboard.katas_due.is_empty() {
                    self.dashboard = Dashboard::load(&self.repo)?;
                }
                if let Some(next_kata) = self.dashboard.katas_due.first().cloned() {
                    let practice_screen = PracticeScreen::new(next_kata.clone())?;
                    self.current_screen = Screen::Practice(next_kata, practice_screen);
                } else {
                    self.refresh_dashboard_screen()?;
                }
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
                            // Force terminal clear to prevent display corruption
                            self.needs_terminal_clear = true;
                        }
                        Screen::Details(_) => {
                            // navigate back to library with updated state
                            let library = Library::load(&self.repo)?;
                            self.current_screen = Screen::Library(library);
                            // Force terminal clear to prevent display corruption
                            self.needs_terminal_clear = true;
                        }
                        _ => {}
                    }

                    // reload dashboard so counts are updated
                    self.dashboard = Dashboard::load(&self.repo)?;
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
                        let library = Library::load(&self.repo)?;
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
                        let library = Library::load(&self.repo)?;
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
                let library = Library::load(&self.repo)?;
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
                            self.repo.get_kata_by_id(*dep_id).ok().flatten().map(|k| k.name)
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
                use crate::core::kata_generator::{rename_kata_directory, update_dependency_in_manifest, update_manifest};

                let exercises_dir = std::path::Path::new("katas/exercises");
                let name_changed = original_slug != new_slug;

                // Get original state for rollback
                let original_kata = self.repo.get_kata_by_id(kata_id)?
                    .ok_or_else(|| anyhow::anyhow!("Kata not found"))?;
                let original_dependencies = self.repo.get_kata_dependencies(kata_id)?;
                let tags = self.repo.get_kata_tags(kata_id)?;

                if name_changed {
                    // 1. Update database first
                    match self.repo.update_kata_full_metadata(
                        kata_id,
                        &new_slug,
                        &form_data.description,
                        &form_data.category,
                        form_data.difficulty as i32,
                    ) {
                        Ok(_) => {
                            // 2. Update dependencies in DB
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
                                eprintln!("Failed to update dependencies: {}", e);
                                return Ok(());
                            }

                            // 3. Now rename directory (if this fails, rollback DB)
                            match rename_kata_directory(exercises_dir, &original_slug, &new_slug) {
                                Ok(_) => {
                                    // 4. Update manifest in new location
                                    let kata_dir = exercises_dir.join(&new_slug);
                                    if let Err(e) =
                                        update_manifest(&kata_dir, &form_data, &new_slug, &tags)
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
                                        let _ = self.repo.replace_dependencies(kata_id, &original_dependencies);
                                        eprintln!("Failed to update manifest: {}", e);
                                        return Ok(());
                                    }

                                    // 5. Update all dependent kata manifests to use new slug
                                    let dependent_kata_ids = self.repo.get_dependent_katas(kata_id)?;
                                    for dep_kata_id in &dependent_kata_ids {
                                        if let Ok(Some(dep_kata)) = self.repo.get_kata_by_id(*dep_kata_id) {
                                            let dep_kata_dir = exercises_dir.join(&dep_kata.name);
                                            if let Err(e) = update_dependency_in_manifest(
                                                &dep_kata_dir,
                                                &original_slug,
                                                &new_slug
                                            ) {
                                                // Rollback: rename directory back, revert DB, and rollback any
                                                // already-updated dependent manifests
                                                eprintln!("Failed to update dependent manifest for '{}': {}", dep_kata.name, e);

                                                // Try to rollback dependent manifests we already updated
                                                for rollback_id in &dependent_kata_ids {
                                                    if rollback_id == dep_kata_id {
                                                        break; // Don't rollback the one that failed
                                                    }
                                                    if let Ok(Some(rb_kata)) = self.repo.get_kata_by_id(*rollback_id) {
                                                        let rb_kata_dir = exercises_dir.join(&rb_kata.name);
                                                        let _ = update_dependency_in_manifest(
                                                            &rb_kata_dir,
                                                            &new_slug,
                                                            &original_slug
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
                                                let _ = self.repo.replace_dependencies(kata_id, &original_dependencies);
                                                return Ok(());
                                            }
                                        }
                                    }

                                    // Success! Return to library
                                    let library = Library::load(&self.repo)?;
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
                                    let _ = self.repo.replace_dependencies(kata_id, &original_dependencies);
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
                        &form_data.category,
                        form_data.difficulty as i32,
                    ) {
                        Ok(_) => {
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
                                eprintln!("Failed to update dependencies: {}", e);
                                return Ok(());
                            }

                            // Update manifest with tags preserved
                            let kata_dir = exercises_dir.join(&original_slug);
                            if let Err(e) = update_manifest(&kata_dir, &form_data, &original_slug, &tags) {
                                // Rollback ALL database changes
                                let _ = self.repo.update_kata_metadata(
                                    kata_id,
                                    &original_kata.description,
                                    &original_kata.category,
                                    original_kata.base_difficulty,
                                );
                                let _ = self.repo.replace_dependencies(kata_id, &original_dependencies);
                                eprintln!("Failed to update manifest: {}", e);
                                return Ok(());
                            }

                            // Success! Return to library
                            let library = Library::load(&self.repo)?;
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
                let editor_status = status_result
                    .with_context(|| format!("Failed to launch {}", editor))?;

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
                let library = Library::load(&self.repo)?;
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
            ScreenAction::BackFromSessionHistory => {
                self.refresh_dashboard_screen()?;
            }
            ScreenAction::BackFromSessionDetail => {
                // Navigate back to session history
                // We need to get the kata from the session detail screen
                if let Screen::SessionDetail(ref session_detail) = &self.current_screen {
                    let kata = self.repo.get_kata_by_id(session_detail.session.kata_id)?
                        .ok_or_else(|| anyhow::anyhow!("Kata not found"))?;
                    let session_history = SessionHistoryScreen::new(kata, &self.repo)?;
                    self.current_screen = Screen::SessionHistory(session_history);
                }
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
    fn handle_rating_submission(&mut self, kata: Kata, rating: u8) -> anyhow::Result<()> {
        // Convert rating to FSRS Rating enum (1-4 scale)
        let fsrs_rating = Rating::from_int(rating as i32)
            .ok_or_else(|| anyhow::anyhow!("Invalid rating: {}", rating))?;

        // Get FSRS parameters (use default if none saved)
        let params = self.repo
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

        // Create session record
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

        let instructions = Paragraph::new("Press any key to dismiss")
            .style(Style::default().fg(Color::Gray));
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
