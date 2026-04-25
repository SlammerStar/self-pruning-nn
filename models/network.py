"""
models/network.py
=================
Self-Pruning Feed-Forward Network for CIFAR-10 image classification.

Architecture
------------
    Input  (3 × 32 × 32) → flatten → 3072
    Layer 1: PrunableLinear(3072 → 1024) + BN + GELU + Dropout
    Layer 2: PrunableLinear(1024 →  512) + BN + GELU + Dropout
    Layer 3: PrunableLinear( 512 →  256) + BN + GELU + Dropout
    Output:  PrunableLinear( 256 →   10)      (logits)

Every linear projection is a PrunableLinear, so ALL weight connections
participate in the learned sparsification.
"""

from __future__ import annotations
from typing import List

import torch
import torch.nn as nn

from models.prunable_layer import PrunableLinear


class SelfPruningNetwork(nn.Module):
    """
    Feed-forward MLP whose every linear layer is prunable.

    Parameters
    ----------
    input_dim    : flattened input size (3072 for CIFAR-10)
    hidden_dims  : hidden layer widths
    num_classes  : number of output classes
    dropout_rate : dropout probability after each hidden activation
    """

    def __init__(
        self,
        input_dim:    int       = 3072,
        hidden_dims:  List[int] = None,
        num_classes:  int       = 10,
        dropout_rate: float     = 0.3,
    ):
        super().__init__()
        hidden_dims = hidden_dims or [1024, 512, 256]

        self.input_dim   = input_dim
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes

        # ── Build hidden blocks ─────────────────────────────────────────────
        layers: List[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers += [
                PrunableLinear(prev, h),
                nn.BatchNorm1d(h),
                nn.GELU(),
                nn.Dropout(p=dropout_rate),
            ]
            prev = h

        self.features   = nn.Sequential(*layers)
        self.classifier = PrunableLinear(prev, num_classes)

        # Convenient reference list — used by loss and metrics
        self._prunable: List[PrunableLinear] = [
            m for m in self.modules() if isinstance(m, PrunableLinear)
        ]

    # ── Forward ─────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)          # flatten: (B,C,H,W) → (B, 3072)
        return self.classifier(self.features(x))

    # ── Gate / sparsity helpers ─────────────────────────────────────────────
    @torch.no_grad()
    def all_gates(self) -> torch.Tensor:
        """Concatenate all gate values across every PrunableLinear into 1-D tensor."""
        return torch.cat([l.get_gates().view(-1) for l in self._prunable]).cpu()

    def sparsity(self, threshold: float = 0.01) -> float:
        """Fraction of weights whose gate < threshold (considered pruned)."""
        g = self.all_gates()
        return (g < threshold).float().mean().item()

    def param_counts(self) -> dict:
        """Return total, prunable, active, and pruned weight counts."""
        prunable = sum(l.weight.numel() for l in self._prunable)
        active   = int((self.all_gates() >= 0.01).sum())
        return {
            "total_params":    sum(p.numel() for p in self.parameters()),
            "prunable_weights": prunable,
            "active_weights":  active,
            "pruned_weights":  prunable - active,
        }

    def extra_repr(self) -> str:
        return (
            f"input_dim={self.input_dim}, "
            f"hidden_dims={self.hidden_dims}, "
            f"num_classes={self.num_classes}"
        )
