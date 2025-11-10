You are helping the user create a new kata for the kata-spaced-repetition project.

## Your Task

Create a complete kata with all required files following the atomic kata design principles from `katas/CLAUDE.md`.

## Process

1. **Gather information** from the user about the kata:
   - Kata name (snake_case, e.g., "attention_scores")
   - Category (e.g., "transformers", "fundamentals", "algorithms")
   - Base difficulty (1-5 scale)
   - Description (what the user will implement, keep under 5 lines)
   - Dependencies (list of prerequisite kata names, can be empty)
   - Tags (optional, for searching/filtering)
   - Function signature (name, arguments with types, return type)
   - What the function should do (brief implementation description)

2. **Create directory structure**:
   ```
   katas/exercises/<kata_name>/
   ├── __init__.py         # Empty file
   ├── manifest.toml       # Metadata
   ├── template.py         # Starter code with BLANK_START/BLANK_END
   ├── reference.py        # Complete solution
   └── test_kata.py        # 5-10 pytest tests
   ```

3. **Generate files** following these requirements:

   **manifest.toml**:
   - Name must match directory name exactly
   - Include all metadata fields
   - Keep description concise and clear
   - List only direct dependencies (not transitive)

   **template.py**:
   - Import necessary modules (torch, jaxtyping, etc.)
   - Complete type hints using jaxtyping for tensors
   - Comprehensive docstring with Args, Returns, optional Example
   - Exactly ONE `BLANK_START`/`BLANK_END` pair
   - Inside BLANK: `raise NotImplementedError("helpful hint here")`

   **reference.py**:
   - Identical signature to template
   - Clean, correct implementation
   - Comments for non-obvious steps
   - This is the "answer" users can compare against

   **test_kata.py**:
   - Import pattern (try user_kata, fallback to reference)
   - 5-10 tests covering:
     - Shape verification (assert_shape)
     - Correctness on simple inputs (assert_close)
     - Edge cases (single element, boundary conditions)
     - Batch processing
     - Realistic dimensions
   - Each test has descriptive docstring
   - Use framework helpers: `assert_shape`, `assert_close`

   **__init__.py**:
   - Empty file (required for Python module)

4. **Validate** the kata:
   - Check all files are created
   - Verify template has exactly one BLANK_START/BLANK_END pair
   - Confirm test import pattern is correct
   - Ensure manifest name matches directory name

5. **Next steps** for the user:
   - Run `kata-sr debug reimport` to load kata into database
   - Optionally run tests against reference: `cd katas && uv run pytest exercises/<kata_name>/test_kata.py`
   - Kata will appear in TUI after reimport

## Important Principles

### Atomic Katas
- Each kata implements exactly ONE function or concept
- If a concept requires multiple functions, create multiple katas
- Keep katas focused: 5-15 minutes to complete
- Use dependencies to create learning paths

### BLANK Markers
- Exactly ONE `BLANK_START`/`BLANK_END` pair per template
- Must be inside function body (not at module/class level)
- User fills in the implementation between markers

### Type Hints
- Use `jaxtyping` for tensor shapes: `Float[torch.Tensor, "batch seq dim"]`
- Include all argument and return types
- Use lowercase generics: `list`, `dict`, `tuple`
- Use `Optional[T]` for optional arguments

### Test Quality
- 5-10 tests minimum
- Cover shapes, correctness, edge cases
- Use descriptive test names: `test_output_shape`, `test_numerical_stability`
- Each test has one-line docstring explaining what it verifies

## Example Interaction

User: "I want to create a kata for implementing ReLU activation"

You: "Great! Let me gather some details:
- Name: relu
- Category: fundamentals
- Difficulty: 1
- Description: Implement ReLU activation: max(0, x)
- Dependencies: none
- Function: `def relu(x: torch.Tensor) -> torch.Tensor`

Is this correct?"

[After confirmation, create all 5 files with complete implementations]

## Tips

- Ask clarifying questions if the kata scope is unclear
- Suggest splitting into multiple katas if concept is too broad
- Recommend appropriate difficulty (1=basic, 5=complex)
- Verify dependencies exist before adding them
- Follow the patterns from existing katas in `katas/exercises/`

Now, ask the user what kata they want to create and gather the necessary information!
