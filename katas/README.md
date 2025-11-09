# Kata Exercises

This directory contains the Python-side resources used by the kata spaced-repetition system. It is designed so you can iterate on individual exercises without needing the full Rust application running.

## Atomic Kata Philosophy

Each kata should be **atomic** - focused on practicing a single function or concept. This design principle ensures:

- Clear learning objectives (you know exactly what you're practicing)
- Faster iteration (complete a kata in 5-15 minutes)
- Better retention (focused practice is more effective than broad review)
- Easier scheduling (small chunks fit naturally into daily practice)

**Guidelines:**
- **One function per kata**: Each kata should implement one specific function or operation
- **Related concepts → separate katas**: Instead of "implement attention mechanism" with 5 TODOs, create separate katas: "attention_scores", "attention_weights", "attention_output", etc.
- **Use dependencies**: Link related katas via the `dependencies` field in manifest.toml
- **Single BLANK**: Templates should use `BLANK_START`/`BLANK_END` markers for a single function body

**Bad (multi-concept kata):**
```python
# template.py with multiple TODOs
def attention_scores(...):
    # TODO: implement scores

def attention_weights(...):
    # TODO: implement softmax

def attention_output(...):
    # TODO: apply weights
```

**Good (atomic katas):**
```python
# kata: attention_scores
def attention_scores(Q, K):
    # BLANK_START
    raise NotImplementedError
    # BLANK_END

# kata: attention_weights (depends on attention_scores)
def attention_weights(scores):
    # BLANK_START
    raise NotImplementedError
    # BLANK_END
```

## Kata Directory Structure

Each kata lives in `exercises/<kata_name>/` with these files:

```
exercises/
└── softmax/
    ├── __init__.py        # Empty marker file
    ├── manifest.toml      # Metadata (name, category, difficulty, dependencies)
    ├── template.py        # Starter code with BLANK_START/BLANK_END
    ├── reference.py       # Your solution implementation
    └── test_kata.py       # Pytest tests
```

### Template Pattern: BLANK_START/BLANK_END

Templates use special markers to indicate where users fill in code:

```python
import torch

def softmax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Compute softmax probabilities along specified dimension.

    Args:
        logits: Input tensor
        dim: Dimension to compute softmax over

    Returns:
        Probabilities that sum to 1.0 along dim
    """
    # BLANK_START
    raise NotImplementedError("Implement softmax using exp and normalization")
    # BLANK_END
```

**Rules:**
- Use exactly one `BLANK_START`/`BLANK_END` pair per template
- Place markers inside the function body (not at file/class level)
- Include type hints and docstrings outside the blank
- Default implementation: `raise NotImplementedError` with helpful message

## Running the Test Suite

You can run the exercises test suite even before you implement any solutions:

```bash
uv run pytest
```

Each kata test file first tries to import the learner's implementation as `user_kata`. When you are still working through the blanks, that module is not available yet. In that case the tests automatically fall back to the checked-in reference implementation so the suite still runs. This allows you to familiarize yourself with test expectations and guard-rails right away.

Once you start filling in the blanks, place your work in the template (or in the location that CLI tooling writes to) so the tests pick up `user_kata` and validate your solution rather than the reference.

## Development Notes

- `uv run pytest` executes both the framework tests under `tests/` and each individual kata's `test_kata.py` file
- The `framework.py` helpers provide common assertions used across exercises (shape checks, numerical comparisons)
- If you want to inspect a reference solution, look at the `reference.py` file within the corresponding kata directory
- Tests only fall back to reference implementations when `user_kata` is missing

## Creating New Katas

For detailed instructions on creating katas, see `CLAUDE.MD` in this directory. Quick reference:

1. Each kata is a single function/concept
2. Use dependencies to create learning paths
3. Templates have one BLANK_START/BLANK_END pair
4. Write 5-10 focused tests covering correctness and edge cases
5. Keep descriptions clear (what to implement, not how)
