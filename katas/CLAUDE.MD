# Kata Creation Guidelines for AI Agents

This document provides comprehensive instructions for AI agents implementing kata exercises for the spaced repetition system.

## Core Philosophy: Atomic Katas

**Each kata must be atomic** - focused on implementing exactly one function or concept.

### Why Atomic?

1. **Clear learning objectives**: User knows precisely what they're practicing
2. **Effective spaced repetition**: Small, focused chunks optimize memory retention
3. **Faster iteration**: Complete a kata in 5-15 minutes, not 30-60 minutes
4. **Granular scheduling**: Each concept scheduled independently based on performance
5. **Lower cognitive load**: Implement one thing well rather than juggle multiple concepts

### Atomic vs Multi-Concept Katas

**❌ Bad (Multi-concept kata):**
```python
# template.py - TOO BROAD
class MultiHeadAttention:
    def __init__(self, d_model, n_heads):
        # TODO: initialize weight matrices

    def split_heads(self, x):
        # TODO: reshape for multi-head

    def attention(self, Q, K, V):
        # TODO: scaled dot-product attention

    def forward(self, x):
        # TODO: full multi-head attention
```

Problems:
- 4 separate concepts mixed together
- Hard to isolate what user struggled with
- All-or-nothing scheduling (can't review just one part)
- Takes 30+ minutes to complete

**✅ Good (Atomic katas):**

```python
# kata: attention_scores
def attention_scores(Q: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """Compute Q @ K.T / sqrt(d_k)"""
    # BLANK_START
    raise NotImplementedError
    # BLANK_END

# kata: attention_weights (depends on softmax)
def attention_weights(scores: torch.Tensor) -> torch.Tensor:
    """Apply softmax to attention scores"""
    # BLANK_START
    raise NotImplementedError
    # BLANK_END

# kata: attention_output
def attention_output(weights: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """Apply attention weights to values: weights @ V"""
    # BLANK_START
    raise NotImplementedError
    # BLANK_END

# kata: multihead_reshape (depends on attention_output)
def multihead_reshape(x: torch.Tensor, n_heads: int) -> torch.Tensor:
    """Reshape (batch, seq, d_model) → (batch, seq, n_heads, d_head)"""
    # BLANK_START
    raise NotImplementedError
    # BLANK_END
```

Benefits:
- Each kata: 5-10 minutes
- Clear what user is practicing
- Dependencies create learning path
- Can review weak areas independently

## File Structure Requirements

Every kata must have exactly 4 files in `exercises/<kata_name>/`:

```
exercises/
└── attention_scores/
    ├── __init__.py         # Empty file (required for Python module)
    ├── manifest.toml       # Metadata (name, category, difficulty, dependencies)
    ├── template.py         # User-facing code with BLANK_START/BLANK_END
    ├── reference.py        # Your complete solution
    └── test_kata.py        # 5-10 pytest tests
```

### 1. `__init__.py`

Always empty. Required for Python module imports.

```python
# Empty file
```

### 2. `manifest.toml`

Metadata defining the kata:

```toml
[kata]
name = "attention_scores"                    # Must match directory name (snake_case)
category = "transformers"                    # Grouping: algorithms, transformers, pytorch, etc.
base_difficulty = 3                          # 1-5 scale (start conservative)
description = """
Compute scaled attention scores from query and key matrices.

Implement the core attention scoring mechanism:
    scores = Q @ K.T / sqrt(d_k)

This forms the foundation of the attention mechanism.
"""
dependencies = []                            # List of prerequisite kata names
tags = ["attention", "matrix-ops"]           # Optional: for searching/filtering
```

**Guidelines:**
- **name**: Must exactly match directory name (lowercase with underscores)
- **category**: Use existing categories when possible (see other katas)
- **base_difficulty**:
  - 1-2: Basic operations (softmax, relu, mean)
  - 3: Moderate complexity (attention scores, layer norm)
  - 4-5: Complex patterns (full attention, custom autograd)
- **description**:
  - First line: What the user implements (one sentence)
  - Optional: Mathematical formula or key insight
  - Keep under 5 lines
- **dependencies**: Only list direct prerequisites (not transitive dependencies)

### 3. `template.py`

User-facing starter code with exactly ONE `BLANK_START`/`BLANK_END` pair.

```python
import torch

def attention_scores(
    Q: torch.Tensor,
    K: torch.Tensor,
    scale: bool = True
) -> torch.Tensor:
    """
    Compute scaled dot-product attention scores.

    Args:
        Q: Query matrix (batch, seq_q, d_k)
        K: Key matrix (batch, seq_k, d_k)
        scale: Whether to scale by sqrt(d_k)

    Returns:
        Attention scores (batch, seq_q, seq_k)

    Example:
        >>> Q = torch.randn(2, 10, 64)
        >>> K = torch.randn(2, 15, 64)
        >>> scores = attention_scores(Q, K)
        >>> scores.shape
        torch.Size([2, 10, 15])
    """
    # BLANK_START
    raise NotImplementedError("Compute Q @ K.transpose(-2, -1), then scale if needed")
    # BLANK_END
```

**Requirements:**
- Exactly ONE `BLANK_START`/`BLANK_END` pair (per file)
- Markers must be inside function body (not at module/class level)
- Include complete type hints (all arguments and return type)
- Include comprehensive docstring with:
  - One-line summary
  - Args section with shapes
  - Returns section with shape
  - Optional: Example usage
- Default implementation: `raise NotImplementedError` with helpful hint
- Keep imports minimal (only what's needed)

**Type Hint Guidelines:**
- Use proper tensor type hints: `torch.Tensor`, `jnp.ndarray`, `np.ndarray`
- Include shape information in docstrings, not type hints
- Use `Optional[T]` for optional args (not `T | None` for broader compatibility)
- For generic types, use lowercase: `list`, `dict`, `tuple`

### 4. `reference.py`

Your complete, correct solution. This is used:
- As fallback during test development
- For answer checking (user can compare)
- For validation (tests run against this first)

```python
import torch

def attention_scores(
    Q: torch.Tensor,
    K: torch.Tensor,
    scale: bool = True
) -> torch.Tensor:
    """
    Compute scaled dot-product attention scores.

    Args:
        Q: Query matrix (batch, seq_q, d_k)
        K: Key matrix (batch, seq_k, d_k)
        scale: Whether to scale by sqrt(d_k)

    Returns:
        Attention scores (batch, seq_q, seq_k)
    """
    # Compute Q @ K^T
    scores = Q @ K.transpose(-2, -1)

    # Scale by sqrt(d_k) if requested
    if scale:
        d_k = Q.shape[-1]
        scores = scores / (d_k ** 0.5)

    return scores
```

**Requirements:**
- Identical signature to template (same function name, args, types)
- Identical imports to template
- Clean, readable implementation (this is the "correct answer")
- Include comments for non-obvious steps
- No testing code (pure implementation)

### 5. `test_kata.py`

Comprehensive pytest tests (5-10 tests covering correctness and edge cases).

```python
import pytest
import torch
from framework import assert_shape, assert_close

# Import user implementation or fall back to reference
try:
    from user_kata import attention_scores
except ImportError:
    from .reference import attention_scores


def test_output_shape():
    """Verify output has correct shape (batch, seq_q, seq_k)"""
    Q = torch.randn(2, 10, 64)
    K = torch.randn(2, 15, 64)
    scores = attention_scores(Q, K)
    assert_shape(scores, (2, 10, 15))


def test_scaling():
    """Scores should be scaled by sqrt(d_k)"""
    Q = torch.randn(1, 5, 32)
    K = torch.randn(1, 5, 32)

    scores_scaled = attention_scores(Q, K, scale=True)
    scores_unscaled = attention_scores(Q, K, scale=False)

    expected_scale = 32 ** 0.5
    assert_close(scores_scaled * expected_scale, scores_unscaled, rtol=1e-5)


def test_single_sequence():
    """Handle single sequence (batch=1, seq=1)"""
    Q = torch.randn(1, 1, 16)
    K = torch.randn(1, 1, 16)
    scores = attention_scores(Q, K)
    assert_shape(scores, (1, 1, 1))


def test_different_seq_lengths():
    """Q and K can have different sequence lengths"""
    Q = torch.randn(1, 10, 64)
    K = torch.randn(1, 20, 64)
    scores = attention_scores(Q, K)
    assert_shape(scores, (1, 10, 20))


def test_correctness_simple():
    """Verify correctness on simple example"""
    # Simple case: Q=K, so scores should be symmetric
    x = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])  # (1, 2, 2)
    scores = attention_scores(x, x, scale=False)

    expected = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])  # Identity
    assert_close(scores, expected, rtol=1e-5)


def test_batch_dimension():
    """Verify batching works correctly"""
    Q = torch.randn(8, 10, 64)
    K = torch.randn(8, 10, 64)
    scores = attention_scores(Q, K)
    assert_shape(scores, (8, 10, 10))


def test_large_dimensions():
    """Handle realistic large dimensions"""
    Q = torch.randn(16, 512, 768)
    K = torch.randn(16, 512, 768)
    scores = attention_scores(Q, K)
    assert_shape(scores, (16, 512, 512))
```

**Test Structure Guidelines:**

1. **Import pattern** (first 4 lines):
   ```python
   try:
       from user_kata import <function_name>
   except ImportError:
       from .reference import <function_name>
   ```

2. **Test count**: 5-10 tests (prefer 7-8)

3. **Test categories** (include at least one of each):
   - **Shape tests**: Verify output dimensions
   - **Correctness tests**: Check actual values on simple inputs
   - **Edge cases**: Single element, empty, boundary conditions
   - **Batch tests**: Verify batching works
   - **Argument tests**: Test optional arguments, defaults
   - **Large scale**: Realistic dimensions

4. **Test naming**: Use descriptive names with `test_` prefix

5. **Assertions**: Use framework helpers when possible
   - `assert_shape(tensor, expected_shape)` for shapes
   - `assert_close(actual, expected, rtol, atol)` for numerical comparison
   - Regular `assert` for boolean conditions

6. **Docstrings**: One-line explanation of what the test verifies

## Dependencies and Learning Paths

Use dependencies to create structured learning progressions.

### Dependency Rules

1. **Direct dependencies only**: List immediate prerequisites, not transitive
   ```toml
   # Good
   [kata]
   name = "attention_output"
   dependencies = ["attention_weights"]  # Only direct dependency

   # Bad
   [kata]
   name = "attention_output"
   dependencies = ["attention_weights", "softmax"]  # softmax is transitive
   ```

2. **Minimal dependencies**: Only require what's actually needed
   - If a kata doesn't use another's implementation, don't depend on it
   - Use dependencies for conceptual prerequisites, not just related topics

3. **Avoid circular dependencies**: Graph must be acyclic

### Example Learning Path: Attention Mechanism

```
softmax (difficulty: 1, deps: [])
    ↓
attention_weights (difficulty: 2, deps: [softmax])
    ↓
attention_scores (difficulty: 2, deps: [])
    ↓
attention_output (difficulty: 3, deps: [attention_weights])
    ↓
multihead_split (difficulty: 3, deps: [])
    ↓
multihead_attention (difficulty: 4, deps: [attention_output, multihead_split])
```

## Implementation Checklist

When implementing a kata, verify:

- [ ] Directory name is snake_case (e.g., `attention_scores`)
- [ ] All 5 files present: `__init__.py`, `manifest.toml`, `template.py`, `reference.py`, `test_kata.py`
- [ ] `manifest.toml` has correct name matching directory
- [ ] Template has exactly ONE `BLANK_START`/`BLANK_END` pair
- [ ] Template has complete type hints and docstring
- [ ] Reference has identical signature to template
- [ ] Reference implementation is correct
- [ ] Tests have try/except import pattern
- [ ] Tests cover shapes, correctness, and edge cases (5-10 tests)
- [ ] Tests pass against reference implementation
- [ ] Dependencies are minimal and direct only
- [ ] Difficulty rating is appropriate (1-5 scale)
- [ ] Description is clear and concise (<5 lines)

## Common Patterns

### Pattern 1: Mathematical Operations

For mathematical functions (softmax, layer_norm, etc.):

```python
# template.py
def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Compute softmax: exp(x) / sum(exp(x))

    Args:
        x: Input tensor (any shape)
        dim: Dimension to normalize over

    Returns:
        Probabilities summing to 1.0 along dim
    """
    # BLANK_START
    raise NotImplementedError("Use torch.exp and normalization")
    # BLANK_END
```

Tests should verify:
- Output shape matches input shape
- Values sum to 1.0 along specified dimension
- Values are in [0, 1]
- Numerical stability (doesn't overflow/underflow)

### Pattern 2: Neural Network Modules

For nn.Module implementations:

```python
# template.py
import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        # BLANK_START
        raise NotImplementedError("Initialize gamma and beta parameters")
        # BLANK_END

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize over last dimension and apply affine transform"""
        # Implementation goes in reference.py, not template
        return x  # Placeholder
```

**Note**: For classes, put BLANK in `__init__`, provide complete `forward` in reference only.

### Pattern 3: Tensor Operations

For complex tensor manipulations (einsum, advanced indexing):

```python
# template.py
def batch_outer_product(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute outer product for each batch: x[i] ⊗ y[i]

    Args:
        x: Tensor (batch, n)
        y: Tensor (batch, m)

    Returns:
        Outer products (batch, n, m)
    """
    # BLANK_START
    raise NotImplementedError("Use broadcasting or einsum")
    # BLANK_END
```

Tests should verify:
- Shape correctness
- Batch independence (result[i] only depends on x[i], y[i])
- Correctness on simple examples

## Best Practices

### Writing Descriptions

**Good descriptions:**
- Start with what the user implements: "Compute scaled attention scores"
- Include key formula if mathematical: "scores = Q @ K.T / sqrt(d_k)"
- Mention purpose/context: "This forms the foundation of the attention mechanism"
- Keep under 5 lines

**Bad descriptions:**
- Too vague: "Implement attention" (what part?)
- Too detailed: Step-by-step instructions (belongs in hints)
- Implementation details: "Use torch.matmul then divide" (let user figure it out)

### Setting Difficulty

- **1**: Basic operations (relu, mean, concatenate)
- **2**: Single-step algorithms (softmax, layer norm, simple attention)
- **3**: Multi-step operations (attention with masking, batch norm)
- **4**: Complex implementations (multi-head attention, custom autograd)
- **5**: Advanced patterns (transformer blocks, complex architectures)

Start conservative - SM-2 will adjust based on user performance.

### Writing Hints

In `NotImplementedError` messages, provide direction without giving away solution:

**Good hints:**
- "Use torch.exp and normalize by sum"
- "Compute Q @ K.transpose(-2, -1)"
- "Scale by sqrt(d_k) for numerical stability"

**Bad hints:**
- "return torch.exp(x) / torch.exp(x).sum(dim)" (too specific)
- "Implement softmax" (too vague)

### Test Design

**Prioritize:**
1. Shape tests (always include 2-3)
2. Correctness on simple inputs (identity, zeros, known values)
3. Edge cases (single element, empty dimensions)
4. Realistic dimensions (batch sizes, sequence lengths)

**Avoid:**
- Testing implementation details (don't assume specific method)
- Brittle tests (exact float comparisons without tolerance)
- Testing more than the kata's scope

## Example: Complete Atomic Kata

Here's a complete example of a well-designed atomic kata:

**Directory**: `exercises/relu/`

**`manifest.toml`**:
```toml
[kata]
name = "relu"
category = "fundamentals"
base_difficulty = 1
description = """
Implement ReLU activation: max(0, x)

Rectified Linear Unit - the most common activation function.
"""
dependencies = []
```

**`template.py`**:
```python
import torch

def relu(x: torch.Tensor) -> torch.Tensor:
    """
    Apply ReLU activation element-wise.

    Args:
        x: Input tensor (any shape)

    Returns:
        Activated tensor (same shape as x)
    """
    # BLANK_START
    raise NotImplementedError("Return max(0, x) element-wise")
    # BLANK_END
```

**`reference.py`**:
```python
import torch

def relu(x: torch.Tensor) -> torch.Tensor:
    """
    Apply ReLU activation element-wise.

    Args:
        x: Input tensor (any shape)

    Returns:
        Activated tensor (same shape as x)
    """
    return torch.maximum(x, torch.zeros_like(x))
```

**`test_kata.py`**:
```python
import pytest
import torch
from framework import assert_shape, assert_close

try:
    from user_kata import relu
except ImportError:
    from .reference import relu


def test_output_shape():
    """Output shape matches input shape"""
    x = torch.randn(10, 20)
    y = relu(x)
    assert_shape(y, (10, 20))


def test_positive_unchanged():
    """Positive values pass through unchanged"""
    x = torch.tensor([1.0, 2.0, 3.0])
    y = relu(x)
    assert_close(y, x)


def test_negative_zeroed():
    """Negative values become zero"""
    x = torch.tensor([-1.0, -2.0, -3.0])
    expected = torch.tensor([0.0, 0.0, 0.0])
    y = relu(x)
    assert_close(y, expected)


def test_mixed_values():
    """Correctly handles mix of positive and negative"""
    x = torch.tensor([-1.0, 0.0, 1.0, -2.0, 2.0])
    expected = torch.tensor([0.0, 0.0, 1.0, 0.0, 2.0])
    y = relu(x)
    assert_close(y, expected)


def test_zero_threshold():
    """Zero maps to zero (boundary case)"""
    x = torch.tensor([0.0])
    y = relu(x)
    assert_close(y, torch.tensor([0.0]))


def test_multidimensional():
    """Works with multidimensional tensors"""
    x = torch.randn(4, 5, 6)
    y = relu(x)
    assert_shape(y, (4, 5, 6))
    assert torch.all((y >= 0) & ((y == 0) | (y == x)))
```

This kata demonstrates all the principles:
- Atomic (one function: ReLU)
- Complete type hints and docstring
- Single BLANK_START/BLANK_END
- Clean reference implementation
- Comprehensive tests (6 tests covering all cases)
- Appropriate difficulty (1 for basic operation)
- Clear, concise description

## FAQ

**Q: What if a concept naturally requires multiple functions?**

A: Split into multiple katas. Example: Instead of "attention mechanism" kata, create:
- `attention_scores` (Q @ K.T)
- `attention_weights` (softmax)
- `attention_output` (weights @ V)

**Q: Should I include multiple BLANK_START/BLANK_END pairs?**

A: No. Exactly one pair per template. If you need multiple, split into multiple katas.

**Q: What about class-based implementations (nn.Module)?**

A: Put BLANK in `__init__` for parameter initialization. Provide complete `forward` in reference only. Or split into separate katas (one for init, one for forward).

**Q: How do I handle optional advanced features?**

A: Use optional arguments with defaults. Test both basic and advanced usage.

**Q: Should every kata have dependencies?**

A: No. Many fundamental katas (softmax, relu, etc.) have no dependencies. Only add dependencies when the concept genuinely builds on another.

## Summary

When creating katas, remember:

1. **Atomic**: One function/concept per kata
2. **Single BLANK**: Exactly one `BLANK_START`/`BLANK_END` pair
3. **Complete types**: Full type hints and comprehensive docstrings
4. **Test thoroughly**: 5-10 tests covering shapes, correctness, edges
5. **Clear descriptions**: What to implement, not how
6. **Minimal dependencies**: Only direct prerequisites
7. **Appropriate difficulty**: 1-5 scale, start conservative

Following these guidelines ensures consistent, high-quality katas that optimize the spaced repetition learning experience.
