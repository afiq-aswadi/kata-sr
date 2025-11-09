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
use super::keybindings;
use super::library::{Library, LibraryAction};
use super::practice::{PracticeAction, PracticeScreen};
use super::results::{ResultsAction, ResultsScreen};
use super::session_detail::{SessionDetailAction, SessionDetailScreen};
use super::session_history::{SessionHistoryAction, SessionHistoryScreen};
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
    /// Library screen for browsing and adding katas
    Library(Library),
    /// Details screen for viewing kata information
    Details(DetailsScreen),
    /// Create kata screen for generating new kata files
    CreateKata(CreateKataScreen),
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

        let mut app = Self {
            current_screen: Screen::Dashboard,
            dashboard,
            repo,
            event_tx: tx,
            event_rx: rx,
            showing_help: false,
            needs_terminal_clear: false,
            popup_message: None,
        };

        app.refresh_dashboard_screen()?;
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
            return;
        }

        match &mut self.current_screen {
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
            Screen::Library(library) => {
                library.render(frame);
            }
            Screen::Details(details) => {
                details.render(frame);
            }
            Screen::CreateKata(create_kata) => {
                create_kata.render(frame);
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
            Screen::Dashboard => {
                // handle library key 'l' in dashboard
                if code == KeyCode::Char('l') {
                    return self.execute_action(ScreenAction::OpenLibrary);
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
            Screen::Done(done_screen) => match done_screen.handle_input(code) {
                DoneAction::BrowseLibrary => Some(ScreenAction::OpenLibrary),
                DoneAction::None => None,
            },
            Screen::Results(kata, results_screen) => {
                let action = results_screen.handle_input(code);
                match action {
                    ResultsAction::SubmitRating(rating) => {
                        Some(ScreenAction::SubmitRating(kata.clone(), rating))
                    }
                    ResultsAction::Retry => Some(ScreenAction::RetryKata(kata.clone())),
                    ResultsAction::StartNextDue => Some(ScreenAction::StartNextDue),
                    ResultsAction::ReviewAnother => Some(ScreenAction::ReturnToDashboard),
                    ResultsAction::BackToDashboard => Some(ScreenAction::ReturnToDashboard),
                    ResultsAction::None => None,
                }
            }
            Screen::Help => Some(ScreenAction::ReturnToDashboard),
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
                        LibraryAction::Back => Some(ScreenAction::BackFromLibrary),
                        LibraryAction::ViewDetails(kata) => {
                            let in_deck = library.kata_ids_in_deck.contains(&kata.name);
                            Some(ScreenAction::ViewDetails(kata, in_deck))
                        }
                        LibraryAction::CreateKata => Some(ScreenAction::OpenCreateKata),
                        LibraryAction::None => None,
                    }
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
