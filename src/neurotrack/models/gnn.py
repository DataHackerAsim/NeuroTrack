"""Interaction-Network edge-classifier GNN.

Per system_plan §4.3 we implement the Battaglia-style Interaction Network:

* Each iteration runs a *relational* MLP that updates edge features from
  ``(src_node, dst_node, edge_feature)`` and an *object* MLP that updates
  node features from the sum of incident updated edge features.
* After ``num_iter`` iterations a final edge MLP emits one logit per
  candidate edge: "do these two hits belong to the same particle?"

The implementation is pure PyTorch (no PyG MessagePassing dependency) so
that gradient checkpointing and ``torch.compile`` are straightforward.

VRAM budget at the defaults (hidden=64, num_iter=8) on 8 GB
target: well under 2 GB for events of size N=2k, E≤16k.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint


def _mlp(
    sizes: list[int], *, activation: type[nn.Module] = nn.GELU,
    norm: bool = True,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            if norm:
                layers.append(nn.LayerNorm(sizes[i + 1]))
            layers.append(activation())
    return nn.Sequential(*layers)


class InteractionNetwork(nn.Module):
    """Edge-classifier GNN producing one logit per edge.

    Parameters
    ----------
    node_dim
        Number of raw node features (default 24 -- the unified feature set).
    edge_dim
        Number of raw edge features (default 7 -- see
        :func:`neurotrack.graph.construction.build_edge_features`).
    hidden_dim
        Internal width of the relational / object MLPs.
    num_iter
        Message-passing iterations.
    use_checkpoint
        Wrap each iteration in :func:`torch.utils.checkpoint.checkpoint`.
        Trades ~30 % compute for ~50 % activation memory.
    """

    def __init__(
        self,
        node_dim: int = 24,
        edge_dim: int = 7,
        hidden_dim: int = 64,
        num_iter: int = 8,
        use_checkpoint: bool = True,
    ) -> None:
        super().__init__()
        self.num_iter = num_iter
        self.use_checkpoint = use_checkpoint

        self.node_encoder = _mlp([node_dim, hidden_dim, hidden_dim])
        self.edge_encoder = _mlp([edge_dim, hidden_dim, hidden_dim])

        # Relational MLP: (h_src, h_dst, e) -> e'
        self.edge_mlp = _mlp([3 * hidden_dim, hidden_dim, hidden_dim])
        # Object MLP: (h, agg_e) -> h'
        self.node_mlp = _mlp([2 * hidden_dim, hidden_dim, hidden_dim])

        # Final readout: (h_src, h_dst, e) -> 1
        self.readout = _mlp([3 * hidden_dim, hidden_dim, 1], norm=False)

    # ------------------------------------------------------------------
    # One message-passing iteration
    # ------------------------------------------------------------------
    def _step(
        self, h: Tensor, e: Tensor, src: Tensor, dst: Tensor,
    ) -> tuple[Tensor, Tensor]:
        h_src = h[src]
        h_dst = h[dst]
        e_new = self.edge_mlp(torch.cat([h_src, h_dst, e], dim=-1)) + e

        # Aggregate updated edge features into nodes (sum over incident edges).
        n_nodes = h.shape[0]
        agg = h.new_zeros((n_nodes, e_new.shape[-1]))
        agg.index_add_(0, dst, e_new)
        agg.index_add_(0, src, e_new)
        h_new = self.node_mlp(torch.cat([h, agg], dim=-1)) + h
        return h_new, e_new

    # ------------------------------------------------------------------
    def forward(
        self, x: Tensor, edge_index: Tensor, edge_attr: Tensor,
    ) -> Tensor:
        if edge_index.numel() == 0:
            return x.new_empty((0,))
        h = self.node_encoder(x)
        e = self.edge_encoder(edge_attr)
        src = edge_index[0].long()
        dst = edge_index[1].long()

        for _ in range(self.num_iter):
            if self.use_checkpoint and self.training:
                h, e = checkpoint(
                    self._step, h, e, src, dst, use_reentrant=False,
                )
            else:
                h, e = self._step(h, e, src, dst)

        h_src = h[src]
        h_dst = h[dst]
        logits: Tensor = self.readout(torch.cat([h_src, h_dst, e], dim=-1)).squeeze(-1)
        return logits


__all__ = ["InteractionNetwork"]
