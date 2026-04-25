"""
models/prunable_layer.py
========================
Custom PyTorch linear layer with per-weight learnable sigmoid gates.

Core mechanism
--------------
Each scalar weight w_ij is paired with a learnable gate score g_ij.
During the forward pass:

    gate_ij        = sigmoid(g_ij)          ∈ (0, 1)
    pruned_w_ij    = w_ij  *  gate_ij
    output         = pruned_W @ x  +  bias

When the total loss includes  λ · mean(sigmoid(gate_scores)),
the optimiser is incentivised to push gate_scores negative
→ sigmoid → 0  →  connection effectively removed.

Why L1 (mean of sigmoid) causes sparsity
-----------------------------------------
sigmoid(g) is always positive, so mean(sigmoid(g)) = mean(|sigmoid(g)|),
i.e. it is an L1 norm over (0,1)-valued gate activations.

The gradient of the L1 penalty w.r.t. gate_score is:
    d/d(g) [ sigmoid(g) ] = sigmoid(g) * (1 - sigmoid(g))

This is always > 0 and approaches 0 only at ±∞, providing a
*persistent downward pull* on every gate throughout training.
Contrast with L2: d/d(g)[sigmoid(g)²] shrinks as g→-∞, allowing
near-zero but non-zero values. L1 maintains constant pressure,
producing truly sparse (exactly-zero-gate) solutions.

Gradient flow
-------------
Both `weight` and `gate_scores` are nn.Parameter leaf tensors.
`pruned_weight = weight * sigmoid(gate_scores)` is a differentiable
product, so autograd propagates gradients to both parameters
with no custom backward needed.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PrunableLinear(nn.Module):
    """
    Drop-in replacement for nn.Linear with learnable per-weight gates.

    Parameters
    ----------
    in_features  : int  — input dimensionality
    out_features : int  — output dimensionality
    bias         : bool — whether to include a bias term (default True)

    Attributes
    ----------
    weight      : Parameter (out, in)  — standard weight matrix
    gate_scores : Parameter (out, in)  — raw gate logits before sigmoid
    bias        : Parameter (out,)     — bias vector (if bias=True)
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features

        # ── Learnable weight (same init as nn.Linear) ──────────────────────
        self.weight = nn.Parameter(torch.empty(out_features, in_features))

        # ── Learnable gate logits ───────────────────────────────────────────
        # Init at 0 → sigmoid(0) = 0.5  (gates start fully half-open)
        # This lets the optimiser push them either toward 1 (keep) or 0 (prune)
        self.gate_scores = nn.Parameter(torch.zeros(out_features, in_features))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        # Kaiming uniform — identical to nn.Linear default
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
            nn.init.uniform_(self.bias, -bound, bound)
        # gate_scores already zeroed in __init__

    # ── Forward pass ────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        1. Compute gates = sigmoid(gate_scores)         ∈ (0, 1)
        2. Mask weights:  pruned_weight = weight * gates
        3. Apply linear:  F.linear(x, pruned_weight, bias)
        """
        gates         = torch.sigmoid(self.gate_scores)
        pruned_weight = self.weight * gates
        return F.linear(x, pruned_weight, self.bias)

    # ── Helpers ─────────────────────────────────────────────────────────────
    @torch.no_grad()
    def get_gates(self) -> torch.Tensor:
        """Return detached gate values (for inspection / metrics)."""
        return torch.sigmoid(self.gate_scores)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}"
        )
