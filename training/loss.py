"""
training/loss.py
================
Composite loss for the Self-Pruning Neural Network.

    Total Loss = CrossEntropyLoss(logits, targets)
               + λ * SparsityLoss

SparsityLoss = MEAN of all sigmoid(gate_scores) across every PrunableLinear.

Why MEAN (not SUM)?
-------------------
Dividing by the total number of gate parameters keeps SparsityLoss in (0, 1),
making λ scale-independent of network size. The same λ value produces
comparable sparsity pressure on small and large networks.

Why does this L1-style penalty induce sparsity?
-----------------------------------------------
sigmoid(g) > 0 always, so mean(sigmoid(g)) is equivalent to an L1 norm
over the sigmoid-activated gate values.

L1 gradient w.r.t. gate_score g:
    ∂/∂g [sigmoid(g)] = sigmoid(g) · (1 − sigmoid(g))

This is always positive, giving a persistent downward gradient on every
gate_score regardless of its current value. This "constant pull toward −∞"
is what makes L1 (not L2) produce exactly-zero gates:
  • L2 penalty: gradient ∝ gate_score → shrinks as gate_score → 0, stalls
  • L1 penalty: gradient ≈ constant → keeps pushing until gate hits 0

The competition between:
  (a) CE loss: wants gates OPEN to allow information flow for accuracy
  (b) L1 penalty: wants gates CLOSED to minimise sparsity cost
produces a natural threshold — gates that don't contribute to accuracy
get zeroed; important gates stay open.
"""

import torch
import torch.nn as nn
from typing import Tuple

from models.network import SelfPruningNetwork


class SparsityRegularisedLoss(nn.Module):
    """
    Total Loss = CrossEntropy + λ * mean(sigmoid(gate_scores))

    Parameters
    ----------
    lam : float
        Sparsity regularisation coefficient.
        λ = 0.001  →  mild pruning, high accuracy preserved
        λ = 0.01   →  balanced trade-off
        λ = 0.1    →  aggressive pruning, some accuracy cost
    """

    def __init__(self, lam: float = 0.01):
        super().__init__()
        self.lam     = lam
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(
        self,
        logits:  torch.Tensor,
        targets: torch.Tensor,
        model:   SelfPruningNetwork,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute total loss and return all three components for logging.

        Parameters
        ----------
        logits  : (B, C) raw model outputs
        targets : (B,)   ground-truth class indices
        model   : the SelfPruningNetwork (needed to access gate_scores)

        Returns
        -------
        total_loss         : scalar — used for backward()
        classification_loss: CE component
        sparsity_loss      : mean gate component (for logging / analysis)
        """
        classification_loss = self.ce_loss(logits, targets)

        # Collect all gate values WITH gradient (not detached)
        all_gates = torch.cat([
            torch.sigmoid(layer.gate_scores).view(-1)
            for layer in model._prunable
        ])
        sparsity_loss = all_gates.mean()  # ∈ (0, 1)

        total_loss = classification_loss + self.lam * sparsity_loss
        return total_loss, classification_loss, sparsity_loss
