"""
utils/metrics.py
================
Evaluation metric helpers.
"""

import torch
import numpy as np


def compute_accuracy(logits: torch.Tensor, targets: torch.Tensor, top_k: int = 1) -> float:
    """Top-k accuracy from raw logits."""
    with torch.no_grad():
        _, pred = logits.topk(top_k, dim=1)
        correct = pred.t().eq(targets.view(1, -1).expand_as(pred.t()))
        return correct[:top_k].reshape(-1).float().mean().item()


def gate_statistics(gates: np.ndarray, threshold: float = 0.01) -> dict:
    """Summary statistics for a 1-D array of gate values."""
    return {
        "sparsity":       float((gates < threshold).mean()),
        "mean":           float(gates.mean()),
        "std":            float(gates.std()),
        "min":            float(gates.min()),
        "max":            float(gates.max()),
        "frac_below_0.1": float((gates < 0.1).mean()),
        "frac_above_0.9": float((gates > 0.9).mean()),
        "n_pruned":       int((gates < threshold).sum()),
        "n_total":        len(gates),
    }
