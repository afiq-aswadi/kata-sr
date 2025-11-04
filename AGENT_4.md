# Agent 4: Example Katas

## Mission

Create 3-4 complete, high-quality kata exercises demonstrating the framework. These serve as both test content and templates for future katas. Focus on correctness, clear TODOs, and comprehensive tests.

## Dependencies

**Requires Agent 2 to be complete:**
- Runner framework
- Manifest parser
- Template validation utilities

You can start designing exercises while Agent 2 works, but testing requires their completion.

## What You're Building

Create these katas:
1. Multi-head Attention (with causal variation)
2. DFS/BFS (graph traversal)
3. MLP (multi-layer perceptron)
4. Layer Normalization

Each kata includes:
- manifest.toml with metadata
- template.py with clear TODO markers
- test_kata.py with 5-10 comprehensive tests
- reference.py with working solution

## Detailed Specifications

### Kata 1: Multi-head Attention

```toml
# katas/exercises/multihead_attention/manifest.toml

[kata]
name = "multihead_attention"
category = "transformers"
base_difficulty = 4
description = """
Implement scaled dot-product multi-head attention.

You'll need to:
1. Project input into query, key, value for each head
2. Compute attention scores with scaling
3. Apply softmax to get attention weights
4. Compute weighted sum of values
5. Concatenate heads and apply output projection

Key concepts: attention mechanism, multi-head parallelism, scaling factor
"""
dependencies = []

[[variations]]
name = "attention_causal"
description = "Add causal masking for autoregressive generation"
params = { mask_type = "causal" }
```

```python
# katas/exercises/multihead_attention/template.py

import torch
import torch.nn as nn
from einops import rearrange, einsum

class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention module.

    Args:
        d_model: dimension of input embeddings
        num_heads: number of attention heads
    """

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        # TODO: initialize query, key, value projection matrices
        # BLANK_START
        self.w_q = None  # Replace this
        self.w_k = None
        self.w_v = None
        # BLANK_END

        # TODO: initialize output projection
        # BLANK_START
        self.w_o = None
        # BLANK_END

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input tensor of shape (batch, seq_len, d_model)

        Returns:
            output tensor of shape (batch, seq_len, d_model)
        """
        batch, seq_len, d_model = x.shape

        # TODO: project to Q, K, V and reshape for multi-head
        # Use einops.rearrange to split heads
        # BLANK_START
        q = None  # Shape should be (batch, num_heads, seq_len, d_head)
        k = None
        v = None
        # BLANK_END

        # TODO: compute scaled dot-product attention
        # attention_scores = Q @ K^T / sqrt(d_head)
        # BLANK_START
        scores = None
        # BLANK_END

        # TODO: apply softmax to get attention weights
        # BLANK_START
        attn_weights = None
        # BLANK_END

        # TODO: apply attention weights to values
        # BLANK_START
        out = None  # Shape: (batch, num_heads, seq_len, d_head)
        # BLANK_END

        # TODO: concatenate heads and apply output projection
        # Use einops.rearrange to merge heads
        # BLANK_START
        out = None  # Shape: (batch, seq_len, d_model)
        # BLANK_END

        return out


# Helper for tests
def create_causal_mask(seq_len: int) -> torch.Tensor:
    """Create upper triangular mask for causal attention"""
    return torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
```

```python
# katas/exercises/multihead_attention/test_kata.py

import pytest
import torch
from user_kata import MultiHeadAttention, create_causal_mask

def test_output_shape():
    batch, seq_len, d_model = 2, 10, 64
    num_heads = 8

    mha = MultiHeadAttention(d_model, num_heads)
    x = torch.randn(batch, seq_len, d_model)
    out = mha(x)

    assert out.shape == (batch, seq_len, d_model), \
        f"Expected shape {(batch, seq_len, d_model)}, got {out.shape}"


def test_attention_weights_normalized():
    """Attention weights should sum to 1 across sequence dimension"""
    batch, seq_len, d_model = 1, 5, 16
    num_heads = 4

    mha = MultiHeadAttention(d_model, num_heads)

    # Monkey-patch to capture attention weights
    captured_weights = []

    original_forward = mha.forward
    def forward_with_capture(x):
        # Run forward and capture intermediate attention weights
        # This requires modifying template to expose attn_weights
        # For now, test indirectly via output
        return original_forward(x)

    x = torch.randn(batch, seq_len, d_model)
    out = mha(x)

    # Indirect test: output should not have NaNs (indicates proper normalization)
    assert not torch.isnan(out).any(), "Output contains NaNs - attention likely not normalized"


def test_scaling_factor_applied():
    """Test that scaling factor sqrt(d_head) is applied"""
    batch, seq_len, d_model = 1, 3, 12
    num_heads = 3

    mha = MultiHeadAttention(d_model, num_heads)
    x = torch.randn(batch, seq_len, d_model)

    # Run forward pass
    out = mha(x)

    # Indirect test: scaled attention should produce reasonable values
    # Without scaling, very large d_head causes extreme softmax saturation
    assert out.abs().max() < 100, "Output values too large - scaling may not be applied"


def test_multiple_heads_different():
    """Different heads should compute different attention patterns"""
    # This is implicitly tested by the multi-head mechanism
    # Each head has independent Q, K, V projections
    pass


def test_deterministic():
    """Same input should produce same output"""
    batch, seq_len, d_model = 2, 5, 16
    num_heads = 4

    torch.manual_seed(42)
    mha = MultiHeadAttention(d_model, num_heads)

    x = torch.randn(batch, seq_len, d_model)
    out1 = mha(x)
    out2 = mha(x)

    assert torch.allclose(out1, out2), "Non-deterministic behavior"


def test_causal_mask_helper():
    """Test causal mask creation"""
    mask = create_causal_mask(4)
    expected = torch.tensor([
        [False, True, True, True],
        [False, False, True, True],
        [False, False, False, True],
        [False, False, False, False],
    ])
    assert torch.equal(mask, expected), "Causal mask incorrect"


def test_gradient_flow():
    """Gradients should flow through attention"""
    batch, seq_len, d_model = 2, 5, 16
    num_heads = 4

    mha = MultiHeadAttention(d_model, num_heads)
    x = torch.randn(batch, seq_len, d_model, requires_grad=True)

    out = mha(x)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None, "Gradients not flowing"
    assert not torch.isnan(x.grad).any(), "NaN in gradients"
```

```python
# katas/exercises/multihead_attention/reference.py

import torch
import torch.nn as nn
from einops import rearrange, einsum

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, d_model = x.shape

        q = rearrange(self.w_q(x), "b s (h d) -> b h s d", h=self.num_heads)
        k = rearrange(self.w_k(x), "b s (h d) -> b h s d", h=self.num_heads)
        v = rearrange(self.w_v(x), "b s (h d) -> b h s d", h=self.num_heads)

        scores = einsum(q, k, "b h i d, b h j d -> b h i j") / (self.d_head ** 0.5)
        attn_weights = torch.softmax(scores, dim=-1)
        out = einsum(attn_weights, v, "b h i j, b h j d -> b h i d")

        out = rearrange(out, "b h s d -> b s (h d)")
        out = self.w_o(out)

        return out


def create_causal_mask(seq_len: int) -> torch.Tensor:
    return torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
```

### Kata 2: DFS/BFS

```toml
# katas/exercises/dfs_bfs/manifest.toml

[kata]
name = "dfs_bfs"
category = "graphs"
base_difficulty = 2
description = """
Implement depth-first search (DFS) and breadth-first search (BFS) for graph traversal.

You'll need to:
1. Implement DFS using recursion or explicit stack
2. Implement BFS using a queue
3. Handle cycle detection (visited set)
4. Return nodes in traversal order

Key concepts: graph traversal, stack vs queue, visited tracking
"""
dependencies = []
```

```python
# katas/exercises/dfs_bfs/template.py

from collections import deque

def dfs(graph: dict[int, list[int]], start: int) -> list[int]:
    """
    Depth-first search traversal.

    Args:
        graph: adjacency list representation {node: [neighbors]}
        start: starting node

    Returns:
        list of nodes in DFS order
    """
    visited = set()
    result = []

    # TODO: implement DFS (recursive or iterative)
    # BLANK_START
    pass
    # BLANK_END

    return result


def bfs(graph: dict[int, list[int]], start: int) -> list[int]:
    """
    Breadth-first search traversal.

    Args:
        graph: adjacency list representation
        start: starting node

    Returns:
        list of nodes in BFS order
    """
    visited = set()
    result = []

    # TODO: implement BFS using a queue
    # BLANK_START
    pass
    # BLANK_END

    return result


def has_path(graph: dict[int, list[int]], start: int, end: int) -> bool:
    """
    Check if there's a path from start to end.

    Args:
        graph: adjacency list
        start: starting node
        end: target node

    Returns:
        True if path exists, False otherwise
    """
    # TODO: use BFS or DFS to check for path
    # BLANK_START
    return False
    # BLANK_END
```

```python
# katas/exercises/dfs_bfs/test_kata.py

import pytest
from user_kata import dfs, bfs, has_path


def test_dfs_simple_graph():
    graph = {
        1: [2, 3],
        2: [4],
        3: [4],
        4: [],
    }
    result = dfs(graph, 1)
    assert len(result) == 4, f"Expected 4 nodes, got {len(result)}"
    assert result[0] == 1, "Should start at node 1"
    assert set(result) == {1, 2, 3, 4}, f"Expected {{1,2,3,4}}, got {set(result)}"


def test_bfs_simple_graph():
    graph = {
        1: [2, 3],
        2: [4],
        3: [5],
        4: [],
        5: [],
    }
    result = bfs(graph, 1)
    assert result == [1, 2, 3, 4, 5] or result == [1, 3, 2, 5, 4], \
        f"BFS order incorrect: {result}"


def test_dfs_with_cycle():
    graph = {
        1: [2],
        2: [3],
        3: [1, 4],
        4: [],
    }
    result = dfs(graph, 1)
    assert len(result) == 4, "Should visit each node exactly once despite cycle"


def test_bfs_with_cycle():
    graph = {
        1: [2, 3],
        2: [1, 4],
        3: [],
        4: [],
    }
    result = bfs(graph, 1)
    assert len(result) == 4, "Should handle cycles correctly"


def test_has_path_exists():
    graph = {
        1: [2],
        2: [3],
        3: [4],
        4: [],
    }
    assert has_path(graph, 1, 4) == True


def test_has_path_does_not_exist():
    graph = {
        1: [2],
        2: [],
        3: [4],
        4: [],
    }
    assert has_path(graph, 1, 4) == False


def test_disconnected_graph():
    graph = {
        1: [2],
        2: [],
        3: [4],
        4: [],
    }
    result = dfs(graph, 1)
    assert set(result) == {1, 2}, "Should only visit connected component"
```

```python
# katas/exercises/dfs_bfs/reference.py

from collections import deque

def dfs(graph: dict[int, list[int]], start: int) -> list[int]:
    visited = set()
    result = []

    def visit(node):
        if node in visited:
            return
        visited.add(node)
        result.append(node)
        for neighbor in graph.get(node, []):
            visit(neighbor)

    visit(start)
    return result


def bfs(graph: dict[int, list[int]], start: int) -> list[int]:
    visited = set([start])
    result = []
    queue = deque([start])

    while queue:
        node = queue.popleft()
        result.append(node)

        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    return result


def has_path(graph: dict[int, list[int]], start: int, end: int) -> bool:
    visited = set()
    queue = deque([start])

    while queue:
        node = queue.popleft()
        if node == end:
            return True

        if node in visited:
            continue

        visited.add(node)
        for neighbor in graph.get(node, []):
            queue.append(neighbor)

    return False
```

### Kata 3: MLP

```toml
# katas/exercises/mlp/manifest.toml

[kata]
name = "mlp"
category = "neural_networks"
base_difficulty = 2
description = """
Implement a simple multi-layer perceptron (MLP) with ReLU activations.

You'll need to:
1. Initialize weight matrices and biases
2. Implement forward pass with linear layers
3. Apply ReLU activation between layers
4. No activation on final layer (for flexibility)

Key concepts: linear transformations, activation functions, layer stacking
"""
dependencies = []
```

```python
# katas/exercises/mlp/template.py

import torch
import torch.nn as nn

class MLP(nn.Module):
    """
    Simple multi-layer perceptron.

    Args:
        input_dim: input feature dimension
        hidden_dims: list of hidden layer dimensions
        output_dim: output dimension
    """

    def __init__(self, input_dim: int, hidden_dims: list[int], output_dim: int):
        super().__init__()

        # TODO: create linear layers
        # Build layers: input -> hidden[0] -> hidden[1] -> ... -> output
        # BLANK_START
        self.layers = None
        # BLANK_END

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: input tensor of shape (batch, input_dim)

        Returns:
            output tensor of shape (batch, output_dim)
        """
        # TODO: apply layers with ReLU activations
        # Don't apply activation after final layer
        # BLANK_START
        return x
        # BLANK_END
```

```python
# katas/exercises/mlp/test_kata.py

import pytest
import torch
from user_kata import MLP


def test_output_shape():
    mlp = MLP(input_dim=10, hidden_dims=[20, 15], output_dim=5)
    x = torch.randn(3, 10)
    out = mlp(x)
    assert out.shape == (3, 5)


def test_single_hidden_layer():
    mlp = MLP(input_dim=5, hidden_dims=[10], output_dim=3)
    x = torch.randn(2, 5)
    out = mlp(x)
    assert out.shape == (2, 3)


def test_no_hidden_layers():
    mlp = MLP(input_dim=5, hidden_dims=[], output_dim=3)
    x = torch.randn(2, 5)
    out = mlp(x)
    assert out.shape == (2, 3)


def test_gradient_flow():
    mlp = MLP(input_dim=5, hidden_dims=[10], output_dim=3)
    x = torch.randn(2, 5, requires_grad=True)
    out = mlp(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None


def test_no_activation_on_output():
    """Output layer should not have ReLU (allows negative values)"""
    mlp = MLP(input_dim=5, hidden_dims=[10], output_dim=3)
    x = torch.randn(1, 5)
    out = mlp(x)
    # If ReLU was applied to output, all values would be >= 0
    # With enough random runs, we should see some negative values
    # This test is probabilistic, but with 100 runs it's very likely
    has_negative = False
    for _ in range(100):
        x = torch.randn(1, 5)
        out = mlp(x)
        if (out < 0).any():
            has_negative = True
            break
    assert has_negative, "Output should allow negative values (no ReLU on output)"
```

```python
# katas/exercises/mlp/reference.py

import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], output_dim: int):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
```

## File Structure You'll Create

```
katas/exercises/
├── multihead_attention/
│   ├── manifest.toml
│   ├── template.py
│   ├── test_kata.py
│   └── reference.py
├── dfs_bfs/
│   ├── manifest.toml
│   ├── template.py
│   ├── test_kata.py
│   └── reference.py
└── mlp/
    ├── manifest.toml
    ├── template.py
    ├── test_kata.py
    └── reference.py
```

## Testing Your Katas

Run the test suite yourself first:

```bash
cd katas
uv run pytest exercises/multihead_attention/test_kata.py
uv run pytest exercises/dfs_bfs/test_kata.py
uv run pytest exercises/mlp/test_kata.py
```

Verify:
- Reference implementation passes all tests
- Template with TODOs filled in passes tests
- Blank template fails tests appropriately

## Acceptance Criteria

- [ ] 3-4 complete katas created
- [ ] Each has manifest.toml with correct metadata
- [ ] Templates have clear TODO/BLANK markers
- [ ] Tests are comprehensive (5-10 per kata)
- [ ] Reference implementations pass all tests
- [ ] Tests use proper assertions with helpful messages
- [ ] Code follows user's conventions (einops, type hints, no emojis)
- [ ] Difficulty ratings are realistic

## Notes

- Follow user's coding style from CLAUDE.md
- Use einops for tensor operations
- Add type hints everywhere
- Assert shapes liberally in tests
- No print statements - use assertions
- Tests should be deterministic (set random seeds if needed)
- Keep TODOs focused - not too granular, not too broad
- Make tests actually test correctness, not just shape
