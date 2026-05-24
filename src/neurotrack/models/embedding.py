"""Hit-embedding MLP (HitEmbedNet).

Per system_plan §4.3: an L2-normalised MLP that maps the 24-D unified
feature vector of each hit to a low-dimensional metric-learning space
(default ``D=12``).  Hits of the same particle are pulled close; hits
of different particles are pushed apart by at least ``margin`` (handled
in the loss, not the model).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class HitEmbedNet(nn.Module):
    """Plain MLP with LayerNorm + GELU + L2-normalised output.

    Tiny model: ~50k parameters at the defaults.  Stays comfortably under
    100 MB VRAM at any reasonable event size.
    """

    def __init__(
        self,
        in_dim: int = 24,
        hidden_dim: int = 128,
        out_dim: int = 12,
        num_layers: int = 4,
        dropout: float = 0.0,
        l2_normalise: bool = True,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        layers: list[nn.Module] = []
        prev = in_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(prev, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = hidden_dim
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)
        self.l2_normalise = l2_normalise
        self.out_dim = out_dim

    def forward(self, x: Tensor) -> Tensor:  # (N, F_in) -> (N, D)
        emb: Tensor = self.net(x)
        if self.l2_normalise:
            emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
        return emb


__all__ = ["HitEmbedNet"]
