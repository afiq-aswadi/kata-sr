//! Integration tests for the database layer.
//!
//! These tests verify the complete database workflow including migrations,
//! CRUD operations, constraint enforcement, and complex queries.

use chrono::{Duration, Utc};
use kata_sr::core::scheduler::{QualityRating, SM2State};
use kata_sr::db::repo::{KataRepository, NewKata, NewSession};
use std::collections::HashMap;

fn setup_test_repo() -> KataRepository {
    let repo = KataRepository::new_in_memory().unwrap();
    repo.run_migrations().unwrap();
    repo
}

#[test]
fn test_migrations_run_successfully() {
    let repo = KataRepository::new_in_memory().unwrap();
    assert!(repo.run_migrations().is_ok());
}

#[test]
fn test_migrations_are_idempotent() {
    let repo = KataRepository::new_in_memory().unwrap();
    repo.run_migrations().unwrap();
    // running again should not fail
    assert!(repo.run_migrations().is_ok());
}

#[test]
fn test_create_kata_returns_id() {
    let repo = setup_test_repo();
    let new_kata = NewKata {
        name: "test_kata".to_string(),
        category: "test".to_string(),
        description: "Test description".to_string(),
        base_difficulty: 3,
        parent_kata_id: None,
        variation_params: None,
    };

    let kata_id = repo.create_kata(&new_kata, Utc::now()).unwrap();
    assert!(kata_id > 0);
}

#[test]
fn test_create_kata_enforces_unique_name() {
    let repo = setup_test_repo();
    let new_kata = NewKata {
        name: "unique_kata".to_string(),
        category: "test".to_string(),
        description: "Test".to_string(),
        base_difficulty: 2,
        parent_kata_id: None,
        variation_params: None,
    };

    repo.create_kata(&new_kata, Utc::now()).unwrap();

    // attempting to create another kata with same name should fail
    let result = repo.create_kata(&new_kata, Utc::now());
    assert!(result.is_err());
}

#[test]
fn test_get_kata_by_id_returns_correct_kata() {
    let repo = setup_test_repo();
    let new_kata = NewKata {
        name: "find_me".to_string(),
        category: "search".to_string(),
        description: "Findable kata".to_string(),
        base_difficulty: 4,
        parent_kata_id: None,
        variation_params: Some("{\"param\": \"value\"}".to_string()),
    };

    let kata_id = repo.create_kata(&new_kata, Utc::now()).unwrap();
    let kata = repo.get_kata_by_id(kata_id).unwrap().unwrap();

    assert_eq!(kata.id, kata_id);
    assert_eq!(kata.name, "find_me");
    assert_eq!(kata.category, "search");
    assert_eq!(kata.base_difficulty, 4);
    assert_eq!(kata.current_difficulty, 4.0);
    assert!(kata.variation_params.is_some());
}

#[test]
fn test_get_kata_by_id_returns_none_for_missing() {
    let repo = setup_test_repo();
    let result = repo.get_kata_by_id(999).unwrap();
    assert!(result.is_none());
}

#[test]
fn test_get_all_katas_returns_all_created() {
    let repo = setup_test_repo();

    for i in 1..=3 {
        let new_kata = NewKata {
            name: format!("kata_{}", i),
            category: "test".to_string(),
            description: format!("Kata number {}", i),
            base_difficulty: i,
            parent_kata_id: None,
            variation_params: None,
        };
        repo.create_kata(&new_kata, Utc::now()).unwrap();
    }

    let all_katas = repo.get_all_katas().unwrap();
    assert_eq!(all_katas.len(), 3);
}

#[test]
fn test_get_katas_due_returns_never_reviewed() {
    let repo = setup_test_repo();
    let new_kata = NewKata {
        name: "never_reviewed".to_string(),
        category: "test".to_string(),
        description: "Has never been reviewed".to_string(),
        base_difficulty: 2,
        parent_kata_id: None,
        variation_params: None,
    };

    repo.create_kata(&new_kata, Utc::now()).unwrap();

    let due = repo.get_katas_due(Utc::now()).unwrap();
    assert_eq!(due.len(), 1);
    assert!(due[0].next_review_at.is_none());
}

#[test]
fn test_get_katas_due_returns_overdue() {
    let repo = setup_test_repo();
    let new_kata = NewKata {
        name: "overdue_kata".to_string(),
        category: "test".to_string(),
        description: "Due yesterday".to_string(),
        base_difficulty: 3,
        parent_kata_id: None,
        variation_params: None,
    };

    let kata_id = repo.create_kata(&new_kata, Utc::now()).unwrap();

    let yesterday = Utc::now() - Duration::days(1);
    let state = SM2State::new();
    repo.update_kata_after_review(kata_id, &state, yesterday, yesterday)
        .unwrap();

    let due = repo.get_katas_due(Utc::now()).unwrap();
    assert_eq!(due.len(), 1);
    assert_eq!(due[0].id, kata_id);
}

#[test]
fn test_get_katas_due_excludes_future() {
    let repo = setup_test_repo();
    let new_kata = NewKata {
        name: "future_kata".to_string(),
        category: "test".to_string(),
        description: "Due tomorrow".to_string(),
        base_difficulty: 2,
        parent_kata_id: None,
        variation_params: None,
    };

    let kata_id = repo.create_kata(&new_kata, Utc::now()).unwrap();

    let tomorrow = Utc::now() + Duration::days(1);
    let state = SM2State::new();
    repo.update_kata_after_review(kata_id, &state, tomorrow, Utc::now())
        .unwrap();

    let due = repo.get_katas_due(Utc::now()).unwrap();
    assert_eq!(due.len(), 0);
}

#[test]
fn test_update_kata_after_review() {
    let repo = setup_test_repo();
    let new_kata = NewKata {
        name: "review_test".to_string(),
        category: "test".to_string(),
        description: "Testing review update".to_string(),
        base_difficulty: 3,
        parent_kata_id: None,
        variation_params: None,
    };

    let kata_id = repo.create_kata(&new_kata, Utc::now()).unwrap();

    let mut state = SM2State::new();
    state.update(QualityRating::Good);
    state.update(QualityRating::Good);

    let next_review = Utc::now() + Duration::days(6);
    let reviewed_at = Utc::now();

    repo.update_kata_after_review(kata_id, &state, next_review, reviewed_at)
        .unwrap();

    let kata = repo.get_kata_by_id(kata_id).unwrap().unwrap();
    assert!(kata.last_reviewed_at.is_some());
    assert!(kata.next_review_at.is_some());
    assert_eq!(kata.current_interval_days, 6);
    assert_eq!(kata.current_repetition_count, 2);
}

#[test]
fn test_update_kata_difficulty() {
    let repo = setup_test_repo();
    let new_kata = NewKata {
        name: "difficulty_test".to_string(),
        category: "test".to_string(),
        description: "Testing difficulty update".to_string(),
        base_difficulty: 3,
        parent_kata_id: None,
        variation_params: None,
    };

    let kata_id = repo.create_kata(&new_kata, Utc::now()).unwrap();
    repo.update_kata_difficulty(kata_id, 4.5).unwrap();

    let kata = repo.get_kata_by_id(kata_id).unwrap().unwrap();
    assert_eq!(kata.current_difficulty, 4.5);
}

#[test]
fn test_create_session() {
    let repo = setup_test_repo();
    let new_kata = NewKata {
        name: "session_kata".to_string(),
        category: "test".to_string(),
        description: "For session testing".to_string(),
        base_difficulty: 2,
        parent_kata_id: None,
        variation_params: None,
    };

    let kata_id = repo.create_kata(&new_kata, Utc::now()).unwrap();

    let session = NewSession {
        kata_id,
        started_at: Utc::now(),
        completed_at: Some(Utc::now()),
        test_results_json: Some("{\"passed\": true}".to_string()),
        num_passed: Some(5),
        num_failed: Some(1),
        num_skipped: Some(0),
        duration_ms: Some(2345),
        quality_rating: Some(2),
    };

    let session_id = repo.create_session(&session).unwrap();
    assert!(session_id > 0);
}

#[test]
fn test_get_recent_sessions() {
    let repo = setup_test_repo();
    let new_kata = NewKata {
        name: "recent_sessions_kata".to_string(),
        category: "test".to_string(),
        description: "For testing recent sessions".to_string(),
        base_difficulty: 3,
        parent_kata_id: None,
        variation_params: None,
    };

    let kata_id = repo.create_kata(&new_kata, Utc::now()).unwrap();

    for i in 0..5 {
        let session = NewSession {
            kata_id,
            started_at: Utc::now() - Duration::minutes(10 * i),
            completed_at: Some(Utc::now() - Duration::minutes(10 * i)),
            test_results_json: None,
            num_passed: Some(i as i32),
            num_failed: Some(0),
            num_skipped: Some(0),
            duration_ms: Some(1000),
            quality_rating: Some(2),
        };
        repo.create_session(&session).unwrap();
    }

    let recent = repo.get_recent_sessions(kata_id, 3).unwrap();
    assert_eq!(recent.len(), 3);
}

#[test]
fn test_get_success_counts() {
    let repo = setup_test_repo();
    let new_kata = NewKata {
        name: "count_kata".to_string(),
        category: "test".to_string(),
        description: "For testing success counts".to_string(),
        base_difficulty: 2,
        parent_kata_id: None,
        variation_params: None,
    };

    let kata_id = repo.create_kata(&new_kata, Utc::now()).unwrap();

    // create sessions with different quality ratings
    // ratings >= 1 should count as success
    for rating in [2, 1, 0, 3, 2] {
        let session = NewSession {
            kata_id,
            started_at: Utc::now(),
            completed_at: Some(Utc::now()),
            test_results_json: None,
            num_passed: None,
            num_failed: None,
            num_skipped: None,
            duration_ms: None,
            quality_rating: Some(rating),
        };
        repo.create_session(&session).unwrap();
    }

    let counts = repo.get_success_counts().unwrap();
    // 2, 1, 3, 2 count as success (4 total), 0 does not
    assert_eq!(counts.get(&kata_id), Some(&4));
}

#[test]
fn test_success_counts_multiple_katas() {
    let repo = setup_test_repo();

    let kata1_id = repo
        .create_kata(
            &NewKata {
                name: "kata1".to_string(),
                category: "test".to_string(),
                description: "First".to_string(),
                base_difficulty: 2,
                parent_kata_id: None,
                variation_params: None,
            },
            Utc::now(),
        )
        .unwrap();

    let kata2_id = repo
        .create_kata(
            &NewKata {
                name: "kata2".to_string(),
                category: "test".to_string(),
                description: "Second".to_string(),
                base_difficulty: 3,
                parent_kata_id: None,
                variation_params: None,
            },
            Utc::now(),
        )
        .unwrap();

    // kata1: 2 successes
    for _ in 0..2 {
        repo.create_session(&NewSession {
            kata_id: kata1_id,
            started_at: Utc::now(),
            completed_at: Some(Utc::now()),
            test_results_json: None,
            num_passed: None,
            num_failed: None,
            num_skipped: None,
            duration_ms: None,
            quality_rating: Some(2),
        })
        .unwrap();
    }

    // kata2: 3 successes
    for _ in 0..3 {
        repo.create_session(&NewSession {
            kata_id: kata2_id,
            started_at: Utc::now(),
            completed_at: Some(Utc::now()),
            test_results_json: None,
            num_passed: None,
            num_failed: None,
            num_skipped: None,
            duration_ms: None,
            quality_rating: Some(1),
        })
        .unwrap();
    }

    let counts = repo.get_success_counts().unwrap();
    assert_eq!(counts.get(&kata1_id), Some(&2));
    assert_eq!(counts.get(&kata2_id), Some(&3));
}

#[test]
fn test_add_and_load_dependency() {
    let repo = setup_test_repo();

    // create katas first to satisfy foreign key constraints
    let kata1_id = repo
        .create_kata(
            &NewKata {
                name: "kata1".to_string(),
                category: "test".to_string(),
                description: "First".to_string(),
                base_difficulty: 2,
                parent_kata_id: None,
                variation_params: None,
            },
            Utc::now(),
        )
        .unwrap();

    let kata2_id = repo
        .create_kata(
            &NewKata {
                name: "kata2".to_string(),
                category: "test".to_string(),
                description: "Second".to_string(),
                base_difficulty: 3,
                parent_kata_id: None,
                variation_params: None,
            },
            Utc::now(),
        )
        .unwrap();

    let kata3_id = repo
        .create_kata(
            &NewKata {
                name: "kata3".to_string(),
                category: "test".to_string(),
                description: "Third".to_string(),
                base_difficulty: 4,
                parent_kata_id: None,
                variation_params: None,
            },
            Utc::now(),
        )
        .unwrap();

    repo.add_dependency(kata2_id, kata1_id, 3).unwrap();
    repo.add_dependency(kata3_id, kata1_id, 1).unwrap();
    repo.add_dependency(kata3_id, kata2_id, 2).unwrap();

    let graph = repo.load_dependency_graph().unwrap();

    let counts = HashMap::new();
    assert!(!graph.is_unlocked(kata2_id, &counts));
    assert!(!graph.is_unlocked(kata3_id, &counts));
}

#[test]
fn test_dependency_constraint_enforcement() {
    let repo = setup_test_repo();

    // attempting to add dependency with non-existent kata fails
    // due to foreign key constraints (bundled sqlite has them enabled)
    let result = repo.add_dependency(999, 998, 1);
    assert!(result.is_err());
}

#[test]
fn test_kata_sm2_state_method() {
    let repo = setup_test_repo();
    let new_kata = NewKata {
        name: "state_test".to_string(),
        category: "test".to_string(),
        description: "Testing SM2 state extraction".to_string(),
        base_difficulty: 3,
        parent_kata_id: None,
        variation_params: None,
    };

    let kata_id = repo.create_kata(&new_kata, Utc::now()).unwrap();

    let mut state = SM2State::new();
    state.update(QualityRating::Easy);

    repo.update_kata_after_review(kata_id, &state, Utc::now(), Utc::now())
        .unwrap();

    let kata = repo.get_kata_by_id(kata_id).unwrap().unwrap();
    let extracted_state = kata.sm2_state();

    assert_eq!(extracted_state.ease_factor, state.ease_factor);
    assert_eq!(extracted_state.interval_days, state.interval_days);
    assert_eq!(extracted_state.repetition_count, state.repetition_count);
}

#[test]
fn test_kata_with_parent() {
    let repo = setup_test_repo();

    let parent_id = repo
        .create_kata(
            &NewKata {
                name: "parent_kata".to_string(),
                category: "test".to_string(),
                description: "Base kata".to_string(),
                base_difficulty: 3,
                parent_kata_id: None,
                variation_params: None,
            },
            Utc::now(),
        )
        .unwrap();

    let child_id = repo
        .create_kata(
            &NewKata {
                name: "child_kata".to_string(),
                category: "test".to_string(),
                description: "Variation of parent".to_string(),
                base_difficulty: 4,
                parent_kata_id: Some(parent_id),
                variation_params: Some("{\"variant\": \"hard\"}".to_string()),
            },
            Utc::now(),
        )
        .unwrap();

    let child = repo.get_kata_by_id(child_id).unwrap().unwrap();
    assert_eq!(child.parent_kata_id, Some(parent_id));
    assert!(child.variation_params.is_some());
}

#[test]
fn test_difficulty_bounds_in_constraint() {
    let repo = setup_test_repo();

    // base_difficulty must be between 1 and 5
    let invalid_low = NewKata {
        name: "too_easy".to_string(),
        category: "test".to_string(),
        description: "Invalid difficulty".to_string(),
        base_difficulty: 0,
        parent_kata_id: None,
        variation_params: None,
    };

    let result = repo.create_kata(&invalid_low, Utc::now());
    assert!(result.is_err());

    let invalid_high = NewKata {
        name: "too_hard".to_string(),
        category: "test".to_string(),
        description: "Invalid difficulty".to_string(),
        base_difficulty: 6,
        parent_kata_id: None,
        variation_params: None,
    };

    let result = repo.create_kata(&invalid_high, Utc::now());
    assert!(result.is_err());
}

#[test]
fn test_quality_rating_bounds_in_constraint() {
    let repo = setup_test_repo();
    let kata_id = repo
        .create_kata(
            &NewKata {
                name: "rating_test".to_string(),
                category: "test".to_string(),
                description: "Testing rating bounds".to_string(),
                base_difficulty: 2,
                parent_kata_id: None,
                variation_params: None,
            },
            Utc::now(),
        )
        .unwrap();

    // quality_rating must be between 0 and 3
    let invalid_session = NewSession {
        kata_id,
        started_at: Utc::now(),
        completed_at: Some(Utc::now()),
        test_results_json: None,
        num_passed: None,
        num_failed: None,
        num_skipped: None,
        duration_ms: None,
        quality_rating: Some(4),
    };

    let result = repo.create_session(&invalid_session);
    assert!(result.is_err());
}

#[test]
fn test_empty_database_queries() {
    let repo = setup_test_repo();

    assert_eq!(repo.get_all_katas().unwrap().len(), 0);
    assert_eq!(repo.get_katas_due(Utc::now()).unwrap().len(), 0);
    assert_eq!(repo.get_success_counts().unwrap().len(), 0);

    let graph = repo.load_dependency_graph().unwrap();
    assert_eq!(graph.get_all_kata_ids().len(), 0);
}
