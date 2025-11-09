"""
Circuit Analysis - Reference Implementation
"""

import torch
from typing import Dict, List, Tuple


def analyze_circuit(
    patching_results: torch.Tensor,
    threshold: float = 0.1
) -> Dict:
    """
    Identify important circuit components from patching heatmap.
    """
    # Find important heads (above threshold)
    important_mask = patching_results > threshold
    important_indices = torch.nonzero(important_mask)
    important_heads = [(idx[0].item(), idx[1].item()) for idx in important_indices]

    # Find maximum effect
    max_value = patching_results.max()
    max_idx = patching_results.argmax()
    n_heads = patching_results.shape[1]
    max_layer = max_idx // n_heads
    max_head = max_idx % n_heads

    return {
        "important_heads": important_heads,
        "max_effect": max_value.item(),
        "max_head": (max_layer.item(), max_head.item())
    }
