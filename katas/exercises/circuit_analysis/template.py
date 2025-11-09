"""
Circuit Analysis

Analyze patching results to identify the causal circuit.
"""

import torch
from typing import Dict, List, Tuple


def analyze_circuit(
    patching_results: torch.Tensor,
    threshold: float = 0.1
) -> Dict:
    """
    Identify important circuit components from patching heatmap.

    This function analyzes a [n_layers, n_heads] heatmap of patching
    effects and returns the heads that matter most for the task.

    Args:
        patching_results: Tensor [n_layers, n_heads] with patching effects
        threshold: Minimum effect to consider a head important

    Returns:
        Dictionary with:
        - important_heads: List of (layer, head) tuples above threshold
        - max_effect: Maximum patching effect found
        - max_head: (layer, head) with maximum effect

    Example:
        >>> results = scan_all_heads(model, clean, corrupt, mary, john)
        >>> circuit = analyze_circuit(results, threshold=0.15)
        >>> print(f"Found {len(circuit['important_heads'])} important heads")
        >>> print(f"Most important: layer {circuit['max_head'][0]}, "
        ...       f"head {circuit['max_head'][1]}")
    """
    # BLANK_START
    raise NotImplementedError(
        "1. Find heads where patching_results > threshold\n"
        "2. Find maximum value and its location\n"
        "3. Convert indices to (layer, head) tuples\n"
        "4. Return dict with important_heads, max_effect, max_head"
    )
    # BLANK_END
