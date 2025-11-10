//! Integration tests for the database layer.
//!
//! These tests verify the complete database workflow including migrations,
//! CRUD operations, constraint enforcement, and complex queries.

use chrono::{Duration, Utc};
use kata_sr::core::fsrs::{FsrsCard, FsrsParams, Rating};
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
    let card = FsrsCard::new();
    repo.update_kata_after_fsrs_review(kata_id, &card, yesterday, yesterday)
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
    let card = FsrsCard::new();
    repo.update_kata_after_fsrs_review(kata_id, &card, tomorrow, Utc::now())
        .unwrap();

    let due = repo.get_katas_due(Utc::now()).unwrap();
    assert_eq!(due.len(), 0);
}

#[test]
fn test_update_kata_after_fsrs_review() {
    let repo = setup_test_repo();
    let new_kata = NewKata {
        name: "review_test".to_string(),
        category: "test".to_string(),
        description: "Testing FSRS review update".to_string(),
        base_difficulty: 3,
        parent_kata_id: None,
        variation_params: None,
    };

    let kata_id = repo.create_kata(&new_kata, Utc::now()).unwrap();

    let mut card = FsrsCard::new();
    let params = FsrsParams::default();
    card.schedule(Rating::Good, &params, Utc::now());
    card.schedule(Rating::Good, &params, Utc::now());

    let next_review = Utc::now() + Duration::days(card.scheduled_days as i64);
    let reviewed_at = Utc::now();

    repo.update_kata_after_fsrs_review(kata_id, &card, next_review, reviewed_at)
        .unwrap();

    let kata = repo.get_kata_by_id(kata_id).unwrap().unwrap();
    assert!(kata.last_reviewed_at.is_some());
    assert!(kata.next_review_at.is_some());
    assert!(kata.fsrs_stability > 0.0);
    assert_eq!(kata.fsrs_reps, 2);
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
        quality_rating: Some(3), // Good (FSRS)
        code_attempt: None,
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
            quality_rating: Some(3), // Good (FSRS)
            code_attempt: None,
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

    // create sessions with different quality ratings (FSRS 1-4 scale)
    // ratings >= 2 (Hard/Good/Easy) should count as success
    for rating in [3, 2, 1, 4, 3] {
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
            code_attempt: None,
        };
        repo.create_session(&session).unwrap();
    }

    let counts = repo.get_success_counts().unwrap();
    // 3, 2, 4, 3 count as success (4 total), 1 (Again) does not
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
            quality_rating: Some(3), // Good (FSRS)
            code_attempt: None,
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
            quality_rating: Some(3), // Good (FSRS) // Hard (FSRS)
            code_attempt: None,
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
fn test_kata_fsrs_card_method() {
    let repo = setup_test_repo();
    let new_kata = NewKata {
        name: "state_test".to_string(),
        category: "test".to_string(),
        description: "Testing FSRS card extraction".to_string(),
        base_difficulty: 3,
        parent_kata_id: None,
        variation_params: None,
    };

    let kata_id = repo.create_kata(&new_kata, Utc::now()).unwrap();

    let mut card = FsrsCard::new();
    let params = FsrsParams::default();
    card.schedule(Rating::Easy, &params, Utc::now());

    repo.update_kata_after_fsrs_review(kata_id, &card, Utc::now(), Utc::now())
        .unwrap();

    let kata = repo.get_kata_by_id(kata_id).unwrap().unwrap();
    let extracted_card = kata.fsrs_card();

    assert_eq!(extracted_card.stability, card.stability);
    assert_eq!(extracted_card.difficulty, card.difficulty);
    assert_eq!(extracted_card.reps, card.reps);
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

    // quality_rating must be between 1 and 4 (FSRS scale)
    let invalid_session_low = NewSession {
        kata_id,
        started_at: Utc::now(),
        completed_at: Some(Utc::now()),
        test_results_json: None,
        num_passed: None,
        num_failed: None,
        num_skipped: None,
        duration_ms: None,
        quality_rating: Some(0),
        code_attempt: None,
    };

    let result = repo.create_session(&invalid_session_low);
    assert!(result.is_err());

    // Also test upper bound
    let invalid_session_high = NewSession {
        kata_id,
        started_at: Utc::now(),
        completed_at: Some(Utc::now()),
        test_results_json: None,
        num_passed: None,
        num_failed: None,
        num_skipped: None,
        duration_ms: None,
        quality_rating: Some(5),
        code_attempt: None,
    };

    let result = repo.create_session(&invalid_session_high);
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

#[test]
fn test_get_future_review_counts() {
    let repo = setup_test_repo();
    let today = Utc::now().date_naive();

    // Create katas with different future review dates
    let kata1 = NewKata {
        name: "kata_tomorrow".to_string(),
        category: "test".to_string(),
        description: "Due tomorrow".to_string(),
        base_difficulty: 3,
        parent_kata_id: None,
        variation_params: None,
    };
    let id1 = repo.create_kata(&kata1, Utc::now()).unwrap();

    let kata2 = NewKata {
        name: "kata_next_week".to_string(),
        category: "test".to_string(),
        description: "Due next week".to_string(),
        base_difficulty: 3,
        parent_kata_id: None,
        variation_params: None,
    };
    let id2 = repo.create_kata(&kata2, Utc::now()).unwrap();

    let kata3 = NewKata {
        name: "kata_same_day".to_string(),
        category: "test".to_string(),
        description: "Due same day as kata2".to_string(),
        base_difficulty: 3,
        parent_kata_id: None,
        variation_params: None,
    };
    let id3 = repo.create_kata(&kata3, Utc::now()).unwrap();

    // Set different next_review_at dates using FSRS
    let tomorrow = (Utc::now() + Duration::days(1)).date_naive();
    let next_week = (Utc::now() + Duration::days(7)).date_naive();

    let params = FsrsParams::default();

    // Schedule kata1 for tomorrow
    let mut card1 = FsrsCard::new();
    card1.schedule(Rating::Good, &params, Utc::now());
    let next_review1 = Utc::now() + Duration::days(1);
    repo.update_kata_after_fsrs_review(id1, &card1, next_review1, Utc::now()).unwrap();

    // Schedule kata2 for next week
    let mut card2 = FsrsCard::new();
    card2.schedule(Rating::Good, &params, Utc::now());
    let next_review2 = Utc::now() + Duration::days(7);
    repo.update_kata_after_fsrs_review(id2, &card2, next_review2, Utc::now()).unwrap();

    // Schedule kata3 for next week (same day as kata2)
    let mut card3 = FsrsCard::new();
    card3.schedule(Rating::Good, &params, Utc::now());
    let next_review3 = Utc::now() + Duration::days(7);
    repo.update_kata_after_fsrs_review(id3, &card3, next_review3, Utc::now()).unwrap();

    // Query future review counts
    let end_date = today + Duration::days(14);
    let counts = repo.get_future_review_counts(today, end_date).unwrap();

    // Should have 2 entries: one for tomorrow (1 kata) and one for next week (2 katas)
    assert!(counts.len() >= 2, "Expected at least 2 dates with scheduled reviews");

    // Find the count for tomorrow
    let tomorrow_count = counts.iter().find(|dc| dc.date == tomorrow);
    assert!(tomorrow_count.is_some(), "Should have count for tomorrow");
    assert_eq!(tomorrow_count.unwrap().count, 1, "Should have 1 kata due tomorrow");

    // Find the count for next week
    let next_week_count = counts.iter().find(|dc| dc.date == next_week);
    assert!(next_week_count.is_some(), "Should have count for next week");
    assert_eq!(next_week_count.unwrap().count, 2, "Should have 2 katas due next week");
}

#[test]
fn test_get_future_review_counts_empty() {
    let repo = setup_test_repo();
    let today = Utc::now().date_naive();
    let end_date = today + Duration::days(14);

    // Empty database should return empty vector
    let counts = repo.get_future_review_counts(today, end_date).unwrap();
    assert_eq!(counts.len(), 0, "Empty database should have no scheduled reviews");
}
