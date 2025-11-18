//! SQL query constants to eliminate duplication across repository modules.

/// Full SELECT columns for kata queries.
pub const KATA_SELECT_COLUMNS: &str =
    "id, name, category, description, base_difficulty, current_difficulty,
                parent_kata_id, variation_params, next_review_at, last_reviewed_at,
                current_ease_factor, current_interval_days, current_repetition_count,
                COALESCE(fsrs_stability, 0.0), COALESCE(fsrs_difficulty, 0.0),
                COALESCE(fsrs_elapsed_days, 0), COALESCE(fsrs_scheduled_days, 0),
                COALESCE(fsrs_reps, 0), COALESCE(fsrs_lapses, 0),
                COALESCE(fsrs_state, 'New'), COALESCE(scheduler_type, 'SM2'),
                created_at,
                COALESCE(is_problematic, FALSE), problematic_notes, flagged_at";

/// Full SELECT columns for kata queries with 'k' table alias (for JOINs).
pub const KATA_SELECT_COLUMNS_ALIASED: &str =
    "k.id, k.name, k.category, k.description, k.base_difficulty, k.current_difficulty,
                k.parent_kata_id, k.variation_params, k.next_review_at, k.last_reviewed_at,
                k.current_ease_factor, k.current_interval_days, k.current_repetition_count,
                COALESCE(k.fsrs_stability, 0.0), COALESCE(k.fsrs_difficulty, 0.0),
                COALESCE(k.fsrs_elapsed_days, 0), COALESCE(k.fsrs_scheduled_days, 0),
                COALESCE(k.fsrs_reps, 0), COALESCE(k.fsrs_lapses, 0),
                COALESCE(k.fsrs_state, 'New'), COALESCE(k.scheduler_type, 'SM2'),
                k.created_at,
                COALESCE(k.is_problematic, FALSE), k.problematic_notes, k.flagged_at";

/// Full SELECT columns for session queries.
pub const SESSION_SELECT_COLUMNS: &str = "id, kata_id, started_at, completed_at, test_results_json,
                num_passed, num_failed, num_skipped, duration_ms, quality_rating, code_attempt";
