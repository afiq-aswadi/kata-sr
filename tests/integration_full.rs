//! Full end-to-end integration tests.
//!
//! These tests verify the complete workflow from database initialization
//! through kata management, sessions, FSRS-5 scheduling, and dependency graphs.
//! All tests use in-memory databases to avoid filesystem side effects.

use chrono::{Duration, Utc};
use kata_sr::core::analytics::Analytics;
use kata_sr::core::fsrs::{FsrsCard, FsrsParams, Rating};
use kata_sr::db::repo::{KataRepository, NewKata, NewSession};
use std::collections::HashMap;

fn setup_repo() -> KataRepository {
    let repo = KataRepository::new_in_memory().unwrap();
    repo.run_migrations().unwrap();
    repo
}

#[test]
fn test_end_to_end_kata_lifecycle() {
    let repo = setup_repo();

    // create a new kata
    let new_kata = NewKata {
        name: "attention_mechanism".to_string(),
        category: "transformers".to_string(),
        description: "Implement attention mechanism".to_string(),
        base_difficulty: 3,
        parent_kata_id: None,
        variation_params: None,
    };

    let kata_id = repo.create_kata(&new_kata, Utc::now()).unwrap();
    assert!(kata_id > 0);

    // verify kata was created with correct FSRS defaults
    let kata = repo.get_kata_by_id(kata_id).unwrap().unwrap();
    assert_eq!(kata.name, "attention_mechanism");
    assert_eq!(kata.fsrs_stability, 0.0);
    assert_eq!(kata.fsrs_difficulty, 0.0);
    assert_eq!(kata.fsrs_reps, 0);
    assert!(kata.next_review_at.is_none());

    // kata should be due for review (never reviewed)
    let due = repo.get_katas_due(Utc::now()).unwrap();
    assert_eq!(due.len(), 1);
    assert_eq!(due[0].id, kata_id);

    // create a session with Good rating (3 in FSRS 1-4 scale)
    let session = NewSession {
        kata_id,
        started_at: Utc::now(),
        completed_at: Some(Utc::now()),
        test_results_json: Some("{\"tests_passed\": 5}".to_string()),
        num_passed: Some(5),
        num_failed: Some(0),
        num_skipped: Some(0),
        duration_ms: Some(3000),
        quality_rating: Some(3), // FSRS Good = 3
    };

    let session_id = repo.create_session(&session).unwrap();
    assert!(session_id > 0);

    // update FSRS state
    let mut card = kata.fsrs_card();
    let params = FsrsParams::default();
    card.schedule(Rating::Good, &params, Utc::now());

    let next_review = Utc::now() + Duration::days(card.scheduled_days as i64);
    repo.update_kata_after_fsrs_review(kata_id, &card, next_review, Utc::now())
        .unwrap();

    // verify kata is no longer due
    let due = repo.get_katas_due(Utc::now()).unwrap();
    assert_eq!(due.len(), 0);

    // but should be due after the interval
    // FSRS Good rating for new card schedules further out than SM-2
    let future = Utc::now() + Duration::days(card.scheduled_days as i64 + 1);
    let due = repo.get_katas_due(future).unwrap();
    assert_eq!(due.len(), 1);
}

#[test]
fn test_fsrs_progression_through_multiple_reviews() {
    let repo = setup_repo();

    let new_kata = NewKata {
        name: "test_kata".to_string(),
        category: "test".to_string(),
        description: "FSRS-5 progression test".to_string(),
        base_difficulty: 2,
        parent_kata_id: None,
        variation_params: None,
    };

    let kata_id = repo.create_kata(&new_kata, Utc::now()).unwrap();

    let ratings = [
        Rating::Good,   // 3
        Rating::Good,   // 3
        Rating::Good,   // 3
        Rating::Hard,   // 2
        Rating::Again,  // 1
    ];

    let mut current_time = Utc::now();
    let params = FsrsParams::default();

    for (i, rating) in ratings.iter().enumerate() {
        // create session
        let session = NewSession {
            kata_id,
            started_at: current_time,
            completed_at: Some(current_time),
            test_results_json: None,
            num_passed: None,
            num_failed: None,
            num_skipped: None,
            duration_ms: Some(1000),
            quality_rating: Some(*rating as i32),
        };
        repo.create_session(&session).unwrap();

        // update FSRS state
        let kata = repo.get_kata_by_id(kata_id).unwrap().unwrap();
        let mut card = kata.fsrs_card();
        card.schedule(*rating, &params, current_time);

        let next_review = current_time + Duration::days(card.scheduled_days as i64);
        repo.update_kata_after_fsrs_review(kata_id, &card, next_review, current_time)
            .unwrap();

        // verify state after each review
        let kata = repo.get_kata_by_id(kata_id).unwrap().unwrap();
        match i {
            0 => {
                // First Good: should have stability and reps = 1
                assert!(kata.fsrs_stability > 0.0);
                assert_eq!(kata.fsrs_reps, 1);
            }
            1 => {
                // Second Good: stability increases, reps = 2
                assert_eq!(kata.fsrs_reps, 2);
            }
            2 => {
                // Third Good: continued progression
                assert_eq!(kata.fsrs_reps, 3);
            }
            3 => {
                // Hard: still progresses but slower
                assert_eq!(kata.fsrs_reps, 4);
            }
            4 => {
                // Again: lapses increases
                assert_eq!(kata.fsrs_lapses, 1);
            }
            _ => {}
        }

        current_time += Duration::days(card.scheduled_days as i64);
    }

    // verify we have 5 sessions recorded
    let sessions = repo.get_recent_sessions(kata_id, 10).unwrap();
    assert_eq!(sessions.len(), 5);
}

#[test]
fn test_dependency_graph_with_unlocking() {
    let repo = setup_repo();

    // create a prerequisite kata
    let prereq = NewKata {
        name: "basics".to_string(),
        category: "foundations".to_string(),
        description: "Basic concepts".to_string(),
        base_difficulty: 1,
        parent_kata_id: None,
        variation_params: None,
    };
    let prereq_id = repo.create_kata(&prereq, Utc::now()).unwrap();

    // create an advanced kata that depends on the prerequisite
    let advanced = NewKata {
        name: "advanced".to_string(),
        category: "foundations".to_string(),
        description: "Advanced concepts".to_string(),
        base_difficulty: 4,
        parent_kata_id: None,
        variation_params: None,
    };
    let advanced_id = repo.create_kata(&advanced, Utc::now()).unwrap();

    // add dependency: advanced requires basics to be completed 3 times
    repo.add_dependency(advanced_id, prereq_id, 3).unwrap();

    // load dependency graph
    let graph = repo.load_dependency_graph().unwrap();
    let success_counts = repo.get_success_counts().unwrap();

    // advanced should be locked initially
    assert!(!graph.is_unlocked(advanced_id, &success_counts));

    // get blocking dependencies
    let blocking = graph.get_blocking_dependencies(advanced_id, &success_counts);
    assert_eq!(blocking.len(), 1);
    assert_eq!(blocking[0].0, prereq_id);
    assert_eq!(blocking[0].1, 3);
    assert_eq!(blocking[0].2, 0);

    // complete prereq kata 3 times with Good rating
    for _ in 0..3 {
        let session = NewSession {
            kata_id: prereq_id,
            started_at: Utc::now(),
            completed_at: Some(Utc::now()),
            test_results_json: None,
            num_passed: None,
            num_failed: None,
            num_skipped: None,
            duration_ms: None,
            quality_rating: Some(3), // Good (FSRS)
        };
        repo.create_session(&session).unwrap();
    }

    // reload success counts
    let success_counts = repo.get_success_counts().unwrap();
    assert_eq!(success_counts.get(&prereq_id), Some(&3));

    // advanced should now be unlocked
    assert!(graph.is_unlocked(advanced_id, &success_counts));

    // no blocking dependencies
    let blocking = graph.get_blocking_dependencies(advanced_id, &success_counts);
    assert_eq!(blocking.len(), 0);
}

#[test]
fn test_complex_dependency_graph() {
    let repo = setup_repo();

    // create a learning path: basics -> intermediate1 & intermediate2 -> advanced
    let basics_id = repo
        .create_kata(
            &NewKata {
                name: "basics".to_string(),
                category: "ml".to_string(),
                description: "ML basics".to_string(),
                base_difficulty: 1,
                parent_kata_id: None,
                variation_params: None,
            },
            Utc::now(),
        )
        .unwrap();

    let inter1_id = repo
        .create_kata(
            &NewKata {
                name: "intermediate1".to_string(),
                category: "ml".to_string(),
                description: "Linear models".to_string(),
                base_difficulty: 2,
                parent_kata_id: None,
                variation_params: None,
            },
            Utc::now(),
        )
        .unwrap();

    let inter2_id = repo
        .create_kata(
            &NewKata {
                name: "intermediate2".to_string(),
                category: "ml".to_string(),
                description: "Neural nets".to_string(),
                base_difficulty: 2,
                parent_kata_id: None,
                variation_params: None,
            },
            Utc::now(),
        )
        .unwrap();

    let advanced_id = repo
        .create_kata(
            &NewKata {
                name: "advanced".to_string(),
                category: "ml".to_string(),
                description: "Transformers".to_string(),
                base_difficulty: 4,
                parent_kata_id: None,
                variation_params: None,
            },
            Utc::now(),
        )
        .unwrap();

    // setup dependencies
    repo.add_dependency(inter1_id, basics_id, 1).unwrap();
    repo.add_dependency(inter2_id, basics_id, 1).unwrap();
    repo.add_dependency(advanced_id, inter1_id, 2).unwrap();
    repo.add_dependency(advanced_id, inter2_id, 2).unwrap();

    let graph = repo.load_dependency_graph().unwrap();
    let mut success_counts = HashMap::new();

    // nothing unlocked except basics
    assert!(graph.is_unlocked(basics_id, &success_counts));
    assert!(!graph.is_unlocked(inter1_id, &success_counts));
    assert!(!graph.is_unlocked(inter2_id, &success_counts));
    assert!(!graph.is_unlocked(advanced_id, &success_counts));

    // complete basics once
    success_counts.insert(basics_id, 1);
    assert!(graph.is_unlocked(inter1_id, &success_counts));
    assert!(graph.is_unlocked(inter2_id, &success_counts));
    assert!(!graph.is_unlocked(advanced_id, &success_counts));

    // complete both intermediates once each
    success_counts.insert(inter1_id, 1);
    success_counts.insert(inter2_id, 1);
    assert!(!graph.is_unlocked(advanced_id, &success_counts));

    // complete one more time each
    success_counts.insert(inter1_id, 2);
    success_counts.insert(inter2_id, 2);
    assert!(graph.is_unlocked(advanced_id, &success_counts));
}

#[test]
fn test_data_persistence_across_connections() {
    use tempfile::NamedTempFile;

    // create a temporary database file
    let temp_file = NamedTempFile::new().unwrap();
    let db_path = temp_file.path();

    // first connection: create kata
    let kata_id = {
        let repo = KataRepository::new(db_path).unwrap();
        repo.run_migrations().unwrap();

        let new_kata = NewKata {
            name: "persistent_kata".to_string(),
            category: "test".to_string(),
            description: "Testing persistence".to_string(),
            base_difficulty: 3,
            parent_kata_id: None,
            variation_params: None,
        };

        repo.create_kata(&new_kata, Utc::now()).unwrap()
    };

    // second connection: verify data persists
    {
        let repo = KataRepository::new(db_path).unwrap();
        let kata = repo.get_kata_by_id(kata_id).unwrap().unwrap();
        assert_eq!(kata.name, "persistent_kata");
        assert_eq!(kata.category, "test");
    }

    // third connection: add session and verify
    {
        let repo = KataRepository::new(db_path).unwrap();

        let session = NewSession {
            kata_id,
            started_at: Utc::now(),
            completed_at: Some(Utc::now()),
            test_results_json: None,
            num_passed: Some(3),
            num_failed: Some(1),
            num_skipped: Some(0),
            duration_ms: Some(2000),
            quality_rating: Some(3), // Good (FSRS)
        };

        repo.create_session(&session).unwrap();
    }

    // fourth connection: verify session persists
    {
        let repo = KataRepository::new(db_path).unwrap();
        let sessions = repo.get_recent_sessions(kata_id, 10).unwrap();
        assert_eq!(sessions.len(), 1);
        assert_eq!(sessions[0].num_passed, Some(3));
    }
}

#[test]
fn test_adaptive_difficulty_workflow() {
    let repo = setup_repo();

    let new_kata = NewKata {
        name: "adaptive_kata".to_string(),
        category: "test".to_string(),
        description: "Testing difficulty adaptation".to_string(),
        base_difficulty: 3,
        parent_kata_id: None,
        variation_params: None,
    };

    let kata_id = repo.create_kata(&new_kata, Utc::now()).unwrap();

    // initial difficulty should match base
    let kata = repo.get_kata_by_id(kata_id).unwrap().unwrap();
    assert_eq!(kata.current_difficulty, 3.0);

    // simulate high success rate (should increase difficulty)
    for _ in 0..5 {
        let session = NewSession {
            kata_id,
            started_at: Utc::now(),
            completed_at: Some(Utc::now()),
            test_results_json: None,
            num_passed: None,
            num_failed: None,
            num_skipped: None,
            duration_ms: None,
            quality_rating: Some(3),
        };
        repo.create_session(&session).unwrap();
    }

    // increase difficulty
    repo.update_kata_difficulty(kata_id, 3.5).unwrap();

    let kata = repo.get_kata_by_id(kata_id).unwrap().unwrap();
    assert_eq!(kata.current_difficulty, 3.5);

    // simulate low success rate (should decrease difficulty)
    for _ in 0..5 {
        let session = NewSession {
            kata_id,
            started_at: Utc::now(),
            completed_at: Some(Utc::now()),
            test_results_json: None,
            num_passed: None,
            num_failed: None,
            num_skipped: None,
            duration_ms: None,
            quality_rating: Some(1), // Again (FSRS)
        };
        repo.create_session(&session).unwrap();
    }

    // decrease difficulty
    repo.update_kata_difficulty(kata_id, 2.8).unwrap();

    let kata = repo.get_kata_by_id(kata_id).unwrap().unwrap();
    assert_eq!(kata.current_difficulty, 2.8);
}

#[test]
fn test_kata_variations() {
    let repo = setup_repo();

    // create parent kata
    let parent_id = repo
        .create_kata(
            &NewKata {
                name: "base_kata".to_string(),
                category: "test".to_string(),
                description: "Base implementation".to_string(),
                base_difficulty: 2,
                parent_kata_id: None,
                variation_params: None,
            },
            Utc::now(),
        )
        .unwrap();

    // create variations
    let easy_variant_id = repo
        .create_kata(
            &NewKata {
                name: "easy_variant".to_string(),
                category: "test".to_string(),
                description: "Easier version".to_string(),
                base_difficulty: 1,
                parent_kata_id: Some(parent_id),
                variation_params: Some("{\"hints\": true, \"tests\": \"basic\"}".to_string()),
            },
            Utc::now(),
        )
        .unwrap();

    let hard_variant_id = repo
        .create_kata(
            &NewKata {
                name: "hard_variant".to_string(),
                category: "test".to_string(),
                description: "Harder version".to_string(),
                base_difficulty: 4,
                parent_kata_id: Some(parent_id),
                variation_params: Some(
                    "{\"hints\": false, \"tests\": \"comprehensive\"}".to_string(),
                ),
            },
            Utc::now(),
        )
        .unwrap();

    // verify relationships
    let easy = repo.get_kata_by_id(easy_variant_id).unwrap().unwrap();
    assert_eq!(easy.parent_kata_id, Some(parent_id));
    assert!(easy.variation_params.is_some());

    let hard = repo.get_kata_by_id(hard_variant_id).unwrap().unwrap();
    assert_eq!(hard.parent_kata_id, Some(parent_id));
    assert!(hard.variation_params.is_some());

    // all three katas should be independent in scheduling
    let all = repo.get_all_katas().unwrap();
    assert_eq!(all.len(), 3);
}

#[test]
fn test_analytics_daily_stats_accuracy() {
    let repo = setup_repo();

    // create two katas in different categories
    let kata1_id = repo
        .create_kata(
            &NewKata {
                name: "kata1".to_string(),
                category: "transformers".to_string(),
                description: "Test 1".to_string(),
                base_difficulty: 3,
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
                category: "algorithms".to_string(),
                description: "Test 2".to_string(),
                base_difficulty: 2,
                parent_kata_id: None,
                variation_params: None,
            },
            Utc::now(),
        )
        .unwrap();

    // create 3 successful sessions and 1 failed session
    for i in 0..3 {
        let kata_id = if i % 2 == 0 { kata1_id } else { kata2_id };
        repo.create_session(&NewSession {
            kata_id,
            started_at: Utc::now(),
            completed_at: Some(Utc::now()),
            test_results_json: None,
            num_passed: Some(5),
            num_failed: Some(0),
            num_skipped: Some(0),
            duration_ms: Some(100),
            quality_rating: Some(3), // Good (FSRS) // Good
        })
        .unwrap();
    }

    // one failed session
    repo.create_session(&NewSession {
        kata_id: kata1_id,
        started_at: Utc::now(),
        completed_at: Some(Utc::now()),
        test_results_json: None,
        num_passed: Some(2),
        num_failed: Some(3),
        num_skipped: Some(0),
        duration_ms: Some(100),
        quality_rating: Some(1), // Again (FSRS)
    })
    .unwrap();

    // compute and verify daily stats
    let analytics = Analytics::new(&repo);
    let today = Utc::now().date_naive();
    let stats = analytics.compute_daily_stats(today).unwrap();

    assert_eq!(stats.total_reviews, 4);
    assert_eq!(stats.total_successes, 3);
    assert_eq!(stats.success_rate, 0.75);
    assert_eq!(stats.streak_days, 1);

    // verify category breakdown
    let categories: HashMap<String, i32> = serde_json::from_str(&stats.categories_json).unwrap();
    assert_eq!(categories.get("transformers"), Some(&3));
    assert_eq!(categories.get("algorithms"), Some(&1));
}

#[test]
fn test_analytics_streak_calculation() {
    let repo = setup_repo();

    let kata_id = repo
        .create_kata(
            &NewKata {
                name: "streak_test".to_string(),
                category: "test".to_string(),
                description: "Streak test".to_string(),
                base_difficulty: 2,
                parent_kata_id: None,
                variation_params: None,
            },
            Utc::now(),
        )
        .unwrap();

    // create sessions on consecutive days
    let now = Utc::now();
    for i in 0..5 {
        let session_time = now - Duration::days(i);
        repo.create_session(&NewSession {
            kata_id,
            started_at: session_time,
            completed_at: Some(session_time),
            test_results_json: None,
            num_passed: Some(5),
            num_failed: Some(0),
            num_skipped: Some(0),
            duration_ms: Some(100),
            quality_rating: Some(3), // Good (FSRS)
        })
        .unwrap();
    }

    // streak should be 5 days
    let streak = repo.get_current_streak().unwrap();
    assert_eq!(streak, 5);

    // verify through analytics
    let analytics = Analytics::new(&repo);
    let today = Utc::now().date_naive();
    let computed_streak = analytics.compute_streak_up_to(today).unwrap();
    assert_eq!(computed_streak, 5);
}

#[test]
fn test_analytics_streak_with_gap() {
    let repo = setup_repo();

    let kata_id = repo
        .create_kata(
            &NewKata {
                name: "gap_test".to_string(),
                category: "test".to_string(),
                description: "Gap test".to_string(),
                base_difficulty: 2,
                parent_kata_id: None,
                variation_params: None,
            },
            Utc::now(),
        )
        .unwrap();

    // create session today
    repo.create_session(&NewSession {
        kata_id,
        started_at: Utc::now(),
        completed_at: Some(Utc::now()),
        test_results_json: None,
        num_passed: Some(5),
        num_failed: Some(0),
        num_skipped: Some(0),
        duration_ms: Some(100),
        quality_rating: Some(2),
    })
    .unwrap();

    // create session 3 days ago (gap of 2 days)
    let three_days_ago = Utc::now() - Duration::days(3);
    repo.create_session(&NewSession {
        kata_id,
        started_at: three_days_ago,
        completed_at: Some(three_days_ago),
        test_results_json: None,
        num_passed: Some(5),
        num_failed: Some(0),
        num_skipped: Some(0),
        duration_ms: Some(100),
        quality_rating: Some(2),
    })
    .unwrap();

    // streak should be 1 (only today counts)
    let streak = repo.get_current_streak().unwrap();
    assert_eq!(streak, 1);
}

#[test]
fn test_analytics_daily_stats_persistence() {
    let repo = setup_repo();

    let kata_id = repo
        .create_kata(
            &NewKata {
                name: "persist_test".to_string(),
                category: "test".to_string(),
                description: "Persistence test".to_string(),
                base_difficulty: 2,
                parent_kata_id: None,
                variation_params: None,
            },
            Utc::now(),
        )
        .unwrap();

    // create a session with Good rating (3) to count as success
    repo.create_session(&NewSession {
        kata_id,
        started_at: Utc::now(),
        completed_at: Some(Utc::now()),
        test_results_json: None,
        num_passed: Some(5),
        num_failed: Some(0),
        num_skipped: Some(0),
        duration_ms: Some(100),
        quality_rating: Some(3), // Good (FSRS)
    })
    .unwrap();

    // compute and save daily stats
    let analytics = Analytics::new(&repo);
    analytics.update_daily_stats().unwrap();

    // verify stats were persisted
    let today = Utc::now().date_naive().format("%Y-%m-%d").to_string();
    let stats = repo.get_daily_stats(&today).unwrap();

    assert!(stats.is_some());
    let stats = stats.unwrap();
    assert_eq!(stats.total_reviews, 1);
    assert_eq!(stats.total_successes, 1);
    assert_eq!(stats.success_rate, 1.0);
}

#[test]
fn test_analytics_success_rate_calculation() {
    let repo = setup_repo();

    let kata_id = repo
        .create_kata(
            &NewKata {
                name: "success_rate_test".to_string(),
                category: "test".to_string(),
                description: "Success rate test".to_string(),
                base_difficulty: 2,
                parent_kata_id: None,
                variation_params: None,
            },
            Utc::now(),
        )
        .unwrap();

    // create 7 sessions over 7 days with mixed success
    let now = Utc::now();
    let ratings = [3, 3, 3, 1, 2, 3, 4]; // 6 successes (>=2: Hard/Good/Easy) out of 7

    for (i, rating) in ratings.iter().enumerate() {
        let session_time = now - Duration::days(i as i64);
        repo.create_session(&NewSession {
            kata_id,
            started_at: session_time,
            completed_at: Some(session_time),
            test_results_json: None,
            num_passed: None,
            num_failed: None,
            num_skipped: None,
            duration_ms: Some(100),
            quality_rating: Some(*rating),
        })
        .unwrap();
    }

    // 7-day success rate should be 6/7 ≈ 0.857
    let success_rate = repo.get_success_rate_last_n_days(7).unwrap();
    assert!((success_rate - 6.0 / 7.0).abs() < 0.01);

    // verify through analytics
    let analytics = Analytics::new(&repo);
    let computed_rate = analytics.get_success_rate_last_n_days(7).unwrap();
    assert_eq!(success_rate, computed_rate);
}

#[test]
fn test_analytics_category_breakdown() {
    let repo = setup_repo();

    // create katas in multiple categories
    let categories = ["transformers", "algorithms", "fundamentals"];
    let mut kata_ids = Vec::new();

    for category in &categories {
        let kata_id = repo
            .create_kata(
                &NewKata {
                    name: format!("kata_{}", category),
                    category: category.to_string(),
                    description: "Test".to_string(),
                    base_difficulty: 2,
                    parent_kata_id: None,
                    variation_params: None,
                },
                Utc::now(),
            )
            .unwrap();
        kata_ids.push(kata_id);
    }

    // create sessions: 3 for transformers, 2 for algorithms, 1 for fundamentals
    for _ in 0..3 {
        repo.create_session(&NewSession {
            kata_id: kata_ids[0],
            started_at: Utc::now(),
            completed_at: Some(Utc::now()),
            test_results_json: None,
            num_passed: Some(5),
            num_failed: Some(0),
            num_skipped: Some(0),
            duration_ms: Some(100),
            quality_rating: Some(3), // Good (FSRS)
        })
        .unwrap();
    }

    for _ in 0..2 {
        repo.create_session(&NewSession {
            kata_id: kata_ids[1],
            started_at: Utc::now(),
            completed_at: Some(Utc::now()),
            test_results_json: None,
            num_passed: Some(5),
            num_failed: Some(0),
            num_skipped: Some(0),
            duration_ms: Some(100),
            quality_rating: Some(3), // Good (FSRS)
        })
        .unwrap();
    }

    repo.create_session(&NewSession {
        kata_id: kata_ids[2],
        started_at: Utc::now(),
        completed_at: Some(Utc::now()),
        test_results_json: None,
        num_passed: Some(5),
        num_failed: Some(0),
        num_skipped: Some(0),
        duration_ms: Some(100),
        quality_rating: Some(2),
    })
    .unwrap();

    // verify category breakdown
    let analytics = Analytics::new(&repo);
    let today = Utc::now().date_naive();
    let breakdown = analytics.get_category_breakdown(today).unwrap();

    assert_eq!(breakdown.get("transformers"), Some(&3));
    assert_eq!(breakdown.get("algorithms"), Some(&2));
    assert_eq!(breakdown.get("fundamentals"), Some(&1));
}
