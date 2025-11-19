use crossterm::event::KeyCode;
use ratatui::Frame;
use std::path::PathBuf;

use crate::db::repo::Kata;
use crate::runner::python_runner::TestResults;
use crate::tui::create_kata::{CreateKataAction, CreateKataScreen};
use crate::tui::dashboard::{Dashboard, DashboardAction};
use crate::tui::details::{DetailsAction, DetailsScreen};
use crate::tui::done::{DoneAction, DoneScreen};
use crate::tui::edit_kata::{EditKataAction, EditKataScreen};
use crate::tui::keybindings;
use crate::tui::library::{Library, LibraryAction};
use crate::tui::practice::{PracticeAction, PracticeScreen};
use crate::tui::results::{ResultsAction, ResultsScreen};
use crate::tui::session_detail::{SessionDetailAction, SessionDetailScreen};
use crate::tui::session_history::{SessionHistoryAction, SessionHistoryScreen};
use crate::tui::settings::{SettingsAction, SettingsScreen};
use crate::tui::startup::{StartupAction, StartupScreen};

/// Current screen being displayed in the TUI.
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

/// Internal enum for handling screen transitions.
pub enum ScreenAction {
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
    OpenEditorFile(PathBuf),
    CancelEditKata,
    OpenSettings,
    CloseSettings,
    ViewSessionHistory(Kata),
    ViewSessionDetail(i64), // session_id
    DeleteSession(i64),     // session_id
    BackFromSessionHistory,
    BackFromSessionDetail,
    
    // Dashboard specific actions that need App handling
    DashboardSelectKata(Kata),
    DashboardRemoveKata(Kata),
    DashboardEditKata(Kata),
    DashboardToggleFlagKata(Kata),
    DashboardToggleHideFlagged,
    
    // Settings specific actions
    SaveSettings(crate::config::AppConfig),
    
    // Practice specific actions
    PracticeEditorExited,
    
    // Results specific actions
    ResultsSolutionViewed,
    ResultsToggleFlagWithReason(Kata, Option<String>),
    
    // Library specific actions
    LibraryToggleFlagKata(Kata),
    LibraryToggleHideFlagged,
    LibraryToggleFlagWithReason(Kata, Option<String>),
    
    // Edit kata by name (needs repo lookup)
    EditKataByName(String),
    
    None,
}

impl Screen {
    pub fn render(&mut self, frame: &mut Frame, dashboard: &Dashboard) {
        match self {
            Screen::Startup(startup_screen) => startup_screen.render(frame),
            Screen::Dashboard => dashboard.render(frame),
            Screen::Done(done_screen) => done_screen.render(frame),
            Screen::Practice(_, practice_screen) => practice_screen.render(frame),
            Screen::Results(_, results_screen) => results_screen.render(frame),
            Screen::Help => keybindings::render_help_screen(frame),
            Screen::Settings(settings_screen) => settings_screen.render(frame),
            Screen::Library(library) => library.render(frame),
            Screen::Details(details) => details.render(frame),
            Screen::CreateKata(create_kata) => create_kata.render(frame),
            Screen::EditKata(edit_kata) => edit_kata.render(frame),
            Screen::SessionHistory(session_history) => session_history.render(frame),
            Screen::SessionDetail(session_detail) => session_detail.render(frame),
        }
    }

    pub fn handle_input(
        &mut self, 
        code: KeyCode, 
        dashboard: &mut Dashboard,
        event_tx: std::sync::mpsc::Sender<crate::tui::app::AppEvent>
    ) -> anyhow::Result<ScreenAction> {
        match self {
            Screen::Startup(startup_screen) => {
                let action = startup_screen.handle_input(code);
                Ok(match action {
                    StartupAction::StartReview => ScreenAction::ReturnToDashboard,
                    StartupAction::OpenLibrary => ScreenAction::OpenLibrary,
                    StartupAction::OpenSettings => ScreenAction::OpenSettings,
                    StartupAction::None => ScreenAction::None,
                })
            }
            Screen::Dashboard => {
                // handle library key 'l' in dashboard
                if code == KeyCode::Char('l') {
                    return Ok(ScreenAction::OpenLibrary);
                }
                // handle settings key 's' in dashboard
                if code == KeyCode::Char('s') {
                    return Ok(ScreenAction::OpenSettings);
                }
                // handle history key 'h' in dashboard (only if kata is selected)
                if code == KeyCode::Char('h') && !dashboard.katas_due.is_empty() {
                    if let Some(kata) = dashboard.katas_due.get(dashboard.selected_index) {
                        return Ok(ScreenAction::ViewSessionHistory(kata.clone()));
                    }
                }
                
                let action = dashboard.handle_input(code);
                Ok(match action {
                    DashboardAction::SelectKata(kata) => ScreenAction::StartPractice(kata),
                    DashboardAction::RemoveKata(kata) => ScreenAction::RemoveKataFromDeck(kata),
                    DashboardAction::EditKata(kata) => ScreenAction::OpenEditKata(kata.id),
                    DashboardAction::ToggleFlagKata(kata) => ScreenAction::DashboardToggleFlagKata(kata),
                    DashboardAction::ToggleHideFlagged => ScreenAction::DashboardToggleHideFlagged,
                    DashboardAction::None => ScreenAction::None,
                })
            }
            Screen::Practice(kata, practice_screen) => {
                // Handle 'f' to toggle flag on current kata
                if code == KeyCode::Char('f') {
                    return Ok(ScreenAction::DashboardToggleFlagKata(kata.clone()));
                }

                let action = practice_screen.handle_input(code, event_tx)?;
                Ok(match action {
                    PracticeAction::BackToDashboard => ScreenAction::ReturnToDashboard,
                    PracticeAction::EditorExited => ScreenAction::PracticeEditorExited,
                    PracticeAction::None => ScreenAction::None,
                })
            }
            Screen::Done(done_screen) => {
                let action = done_screen.handle_input(code);
                Ok(match action {
                    DoneAction::BrowseLibrary => ScreenAction::OpenLibrary,
                    DoneAction::ToggleHideFlagged => ScreenAction::DashboardToggleHideFlagged,
                    DoneAction::None => ScreenAction::None,
                })
            }
            Screen::Results(kata, results_screen) => {
                let action = results_screen.handle_input(code);
                Ok(match action {
                    ResultsAction::SubmitRating(rating) => ScreenAction::SubmitRating(
                        kata.clone(),
                        rating,
                        results_screen.get_results().clone(),
                    ),
                    ResultsAction::BuryCard => ScreenAction::BuryKata(kata.clone()),
                    ResultsAction::Retry => ScreenAction::RetryKata(kata.clone()),
                    ResultsAction::GiveUp => {
                        // Open solution in editor
                        match results_screen.open_solution_in_editor(true) {
                            Ok(()) => {
                                // Check if this is preview mode (kata not in deck)
                                if kata.id == -1 {
                                    // Preview mode: return to library
                                    ScreenAction::OpenLibrary
                                } else {
                                    // Normal mode: auto-submit Rating::Again (1)
                                    ScreenAction::SubmitRating(
                                        kata.clone(),
                                        1, // Rating::Again
                                        results_screen.get_results().clone(),
                                    )
                                }
                            }
                            Err(e) => {
                                eprintln!("Failed to open solution: {}", e);
                                ScreenAction::None
                            }
                        }
                    }
                    ResultsAction::SolutionViewed => ScreenAction::ResultsSolutionViewed,
                    ResultsAction::StartNextDue => ScreenAction::StartNextDue,
                    ResultsAction::ReviewAnother => ScreenAction::ReturnToDashboard,
                    ResultsAction::BackToDashboard => ScreenAction::ReturnToDashboard,
                    ResultsAction::BackToLibrary => ScreenAction::OpenLibrary,
                    ResultsAction::OpenSettings => ScreenAction::OpenSettings,
                    ResultsAction::ToggleFlagWithReason(reason) => ScreenAction::ResultsToggleFlagWithReason(kata.clone(), reason),
                    ResultsAction::None => ScreenAction::None,
                })
            }
            Screen::Help => Ok(ScreenAction::ReturnToDashboard),
            Screen::Settings(settings_screen) => {
                let action = settings_screen.handle_input(code);
                Ok(match action {
                    SettingsAction::Cancel => ScreenAction::CloseSettings,
                    SettingsAction::Save(config) => ScreenAction::SaveSettings(config),
                    SettingsAction::None => ScreenAction::None,
                })
            }
            Screen::Library(library) => {
                // handle history key 'h' in My Deck tab
                if code == KeyCode::Char('h') {
                    use crate::tui::library::LibraryTab;
                    if library.active_tab == LibraryTab::MyDeck
                        && library.deck_selected < library.deck_katas.len()
                    {
                        let kata = library.deck_katas[library.deck_selected].clone();
                        return Ok(ScreenAction::ViewSessionHistory(kata));
                    }
                }
                
                let action = library.handle_input(code);
                Ok(match action {
                    LibraryAction::AddKata(name) => ScreenAction::AddKataFromLibrary(name),
                    LibraryAction::AttemptKata(available_kata) => ScreenAction::AttemptKataWithoutDeck(available_kata),
                    LibraryAction::PracticeKata(kata) => ScreenAction::StartPractice(kata),
                    LibraryAction::RemoveKata(kata) => ScreenAction::RemoveKataFromDeck(kata),
                    LibraryAction::ToggleFlagKata(kata) => ScreenAction::LibraryToggleFlagKata(kata),
                    LibraryAction::ToggleHideFlagged => ScreenAction::LibraryToggleHideFlagged,
                    LibraryAction::ToggleFlagWithReason(kata, reason) => ScreenAction::LibraryToggleFlagWithReason(kata, reason),
                    LibraryAction::Back => ScreenAction::BackFromLibrary,
                    LibraryAction::ViewDetails(kata) => {
                        let in_deck = library.kata_ids_in_deck.contains(&kata.name);
                        ScreenAction::ViewDetails(kata, in_deck)
                    }
                    LibraryAction::CreateKata => ScreenAction::OpenCreateKata,
                    LibraryAction::EditKataById(kata_id) => ScreenAction::OpenEditKata(kata_id),
                    LibraryAction::EditKataByName(name) => ScreenAction::EditKataByName(name),
                     _ => ScreenAction::None,
                })
            }
            Screen::Details(details) => {
                let action = details.handle_input(code);
                Ok(match action {
                    DetailsAction::AddKata(name) => ScreenAction::AddKataFromLibrary(name),
                    DetailsAction::Back => ScreenAction::BackFromDetails,
                    DetailsAction::EditKata(name) => ScreenAction::EditKataByName(name),
                    DetailsAction::None => ScreenAction::None,
                })
            }
            Screen::CreateKata(create_kata) => {
                let exercises_dir = std::path::Path::new("katas/exercises");
                let action = create_kata.handle_input(code, exercises_dir);
                Ok(match action {
                    CreateKataAction::Submit { form_data, slug } => {
                        ScreenAction::SubmitNewKata { form_data, slug }
                    }
                    CreateKataAction::Cancel => ScreenAction::CancelCreateKata,
                    CreateKataAction::None => ScreenAction::None,
                })
            }
            Screen::EditKata(edit_kata) => {
                let exercises_dir = std::path::Path::new("katas/exercises");
                let action = edit_kata.handle_input(code, exercises_dir);
                Ok(match action {
                    EditKataAction::Submit {
                        kata_id,
                        original_slug,
                        form_data,
                        new_slug,
                    } => ScreenAction::SubmitEditKata {
                        kata_id,
                        original_slug,
                        form_data,
                        new_slug,
                    },
                    EditKataAction::OpenEditor { file_path } => {
                        ScreenAction::OpenEditorFile(file_path)
                    }
                    EditKataAction::Cancel => ScreenAction::CancelEditKata,
                    EditKataAction::None => ScreenAction::None,
                })
            }
            Screen::SessionHistory(session_history) => {
                let action = session_history.handle_input(code);
                Ok(match action {
                    SessionHistoryAction::ViewDetails(session_id) => {
                        ScreenAction::ViewSessionDetail(session_id)
                    }
                    SessionHistoryAction::Delete(session_id) => {
                        ScreenAction::DeleteSession(session_id)
                    }
                    SessionHistoryAction::Back => ScreenAction::BackFromSessionHistory,
                    SessionHistoryAction::None => ScreenAction::None,
                })
            }
            Screen::SessionDetail(session_detail) => {
                let action = session_detail.handle_input(code);
                Ok(match action {
                    SessionDetailAction::Back => ScreenAction::BackFromSessionDetail,
                    SessionDetailAction::None => ScreenAction::None,
                })
            }
        }
    }
}
