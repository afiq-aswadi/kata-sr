# FSRS-5 Implementation

## Overview

This implementation adds FSRS-5 (Free Spaced Repetition Scheduler) as a modern alternative to the SM-2 algorithm. FSRS-5 provides more accurate scheduling predictions based on a memory model that tracks retrievability and stability.

## Key Features

### 1. Core FSRS-5 Algorithm (`src/core/fsrs.rs`)

- **Full FSRS-5 Implementation**: 19-parameter model with default parameters optimized from millions of reviews
- **Card States**: New, Learning, Review, Relearning
- **Rating System**: 4-button (Again/Hard/Good/Easy) compatible with existing SM-2 ratings
- **Memory Model**: Tracks stability (half-life), difficulty (1-10), and retrievability

### 2. Parameter Optimization (`src/core/fsrs_optimizer.rs`)

- **Gradient Descent Optimizer**: Trains parameters on your review history
- **Minimum Requirements**: 50 reviews recommended for optimization
- **Configurable**: Learning rate, epochs, and min reviews can be customized
- **Automatic Fallback**: Uses default parameters if insufficient data

### 3. Database Support

#### New Tables

- **`fsrs_params`**: Stores optimized parameter sets with timestamps

#### New Columns in `katas` Table

- `fsrs_stability`: Memory stability in days
- `fsrs_difficulty`: Card difficulty (1-10)
- `fsrs_elapsed_days`: Days since last review
- `fsrs_scheduled_days`: Scheduled interval
- `fsrs_reps`: Number of reviews
- `fsrs_lapses`: Number of Again ratings
- `fsrs_state`: Learning state (New/Learning/Review/Relearning)
- `scheduler_type`: 'SM2' or 'FSRS'

### 4. Dual Algorithm Support

- **Backward Compatible**: Existing SM-2 data is preserved
- **Per-Kata Scheduling**: Each kata can use either SM-2 or FSRS
- **Easy Migration**: Simple methods to switch algorithms

## Usage

### Using FSRS for a Kata

```rust
use kata_sr::db::repo::KataRepository;
use kata_sr::core::fsrs::{FsrsCard, FsrsParams, Rating};
use chrono::Utc;

// Get kata and params
let repo = KataRepository::new("kata.db")?;
let kata = repo.get_kata_by_id(1)?.unwrap();
let params = repo.get_latest_fsrs_params()?
    .unwrap_or_else(|| FsrsParams::default());

// Schedule a review
let mut card = kata.fsrs_card();
let now = Utc::now();
card.schedule(Rating::Good, &params, now);

// Calculate next review time
let next_review = now + chrono::Duration::days(card.scheduled_days as i64);

// Save to database
repo.update_kata_after_fsrs_review(kata.id, &card, next_review, now)?;
```

### Optimizing Parameters

```rust
use kata_sr::core::fsrs_optimizer::FsrsOptimizer;

// Create optimizer with default settings
let optimizer = FsrsOptimizer::new();

// Or customize settings
let optimizer = FsrsOptimizer::with_config(
    0.01,   // learning_rate
    100,    // epochs
    50      // min_reviews
);

// Optimize on review history
let optimized_params = optimizer.optimize(&repo)?;

// Save to database
repo.save_fsrs_params(&optimized_params, Utc::now())?;
```

### Migrating Katas to FSRS

```rust
// Migrate a single kata
repo.set_kata_to_fsrs(kata_id)?;

// Migrate all katas
repo.migrate_all_to_fsrs()?;

// Revert to SM-2
repo.set_kata_to_sm2(kata_id)?;
```

## Database Migration

The migration is automatic and safe:

1. **Existing Databases**: FSRS columns are added automatically on first run
2. **New Databases**: All columns are created from the start
3. **Data Preservation**: SM-2 data remains intact
4. **Default Values**: FSRS columns default to safe initial values

## Algorithm Comparison

### SM-2 (Original)

- **Rating Scale**: 0-3 (Again/Hard/Good/Easy)
- **Parameters**: Ease factor (2.5 default)
- **State**: Simple interval and repetition count
- **Best For**: Simplicity, proven track record

### FSRS-5 (New)

- **Rating Scale**: 1-4 (Again/Hard/Good/Easy) - compatible mapping
- **Parameters**: 19 weights, optimizable
- **State**: Stability, difficulty, retrievability
- **Best For**: Accuracy, personalization, long-term retention

## Testing

All FSRS functionality is thoroughly tested:

```bash
# Run all FSRS tests
cargo test --lib fsrs

# Run specific test modules
cargo test core::fsrs::tests
cargo test core::fsrs_optimizer::tests
```

### Test Coverage

- ✅ Rating conversion (1-4 scale)
- ✅ Card state transitions
- ✅ Scheduling for New/Learning/Review/Relearning states
- ✅ Stability calculations
- ✅ Difficulty bounds (1-10)
- ✅ Forgetting curve
- ✅ Parameter optimization
- ✅ Training data extraction
- ✅ Gradient computation

## Implementation Details

### FSRS-5 Formulas

#### Initial Stability
```
S_0 = w[rating - 1]  // w[0], w[1], w[2], or w[3]
```

#### Initial Difficulty
```
D_0 = w[4] + w[5] * (rating - 3)
Clamped to [1.0, 10.0]
```

#### Forgetting Curve
```
R = (1 + elapsed_days / (9 * S))^(-1)
```

#### Difficulty Update
```
D_new = D + w[6] * (rating - 3) - w[7] * (D - w[4])
Clamped to [1.0, 10.0]
```

#### Stability After Recall
```
S_new = S * (1 + exp(w[8]) * (11 - D) * S^(-w[9]) *
              (exp((1 - R) * w[10]) - 1) * hard_penalty * easy_bonus)
```

#### Stability After Lapse
```
S_new = w[11] * D^(-w[12]) * ((S + 1)^w[13] - 1) * exp((1 - R) * w[14])
```

### Default Parameters

Based on analysis of millions of reviews:
```rust
[0.4072, 1.1829, 3.1262, 15.4722, 7.2102,
 0.5316, 1.0651, 0.0234, 1.616, 0.1544,
 1.0824, 1.9813, 0.0953, 0.2975, 2.2042,
 0.2407, 2.9466, 0.5034, 0.6567]
```

## Future Enhancements

Potential improvements for future versions:

1. **CLI Commands**: Add commands for optimization and migration
2. **TUI Integration**: Display FSRS stats in dashboard
3. **Parameter Versioning**: Track multiple parameter sets over time
4. **Advanced Optimization**: Use more sophisticated optimization algorithms
5. **Review Analytics**: Show FSRS-specific metrics (retrievability, stability trends)

## References

- [FSRS-5 Algorithm](https://github.com/open-spaced-repetition/fsrs4anki/wiki/The-Algorithm)
- [FSRS Rust Implementation](https://github.com/open-spaced-repetition/fsrs-rs)
- [Original FSRS Paper](https://www.nature.com/articles/s41562-024-01962-9)

## Support

For issues or questions:
- File an issue on GitHub
- Check the test suite for usage examples
- Review the inline documentation in the source code
