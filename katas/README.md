# Kata Exercises

This directory contains the Python-side resources used by the kata spaced-repetition system.  It is designed so you can iterate on individual exercises without needing the full Rust application running.

## Running the Test Suite Early

You can run the exercises test suite even before you implement any solutions:

```bash
uv run pytest
```

Each kata test file first tries to import the learner's implementation as `user_kata`.  When you are still working through the blanks, that module is not available yet.  In that case the tests automatically fall back to the checked-in reference implementation so the suite still runs.  This allows you to familiarize yourself with test expectations and guard-rails right away.

Once you start filling in the blanks, place your work in the template (or in the location that CLI tooling writes to) so the tests pick up `user_kata` and validate your solution rather than the reference.

## Development Notes

- `uv run pytest` executes both the framework tests under `tests/` and each individual kata's `test_kata.py` file.
- The `framework.py` helpers provide a couple of common assertions used across exercises (shape checks, numerical comparisons).
- If you want to inspect a reference solution, look at the `reference.py` file within the corresponding kata directory.  The tests only fall back to those implementations when `user_kata` is missing, so your own code will be exercised as soon as it is present.
