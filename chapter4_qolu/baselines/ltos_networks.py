"""Networks of the LToS baseline: the DGN low level and the DDPG high level.

Yi et al. (2021) give the *structure*; three numbers - the encoder-layer,
graph-convolution-layer and attention-head counts - are standard DGN defaults
rather than the values of Yi et al.'s Table 4, marked ``Design note`` on the
fields of :class:`chapter4_qolu.baselines.config.LToSConfig`:

* one encoder layer;
* two graph-convolution (multi-head attention) layers, with the encoder output
  and both layer outputs concatenated before the Q head, which is how DGN
  (Jiang et al., ICLR 2020) builds its Q input;
* eight attention heads.

Everything else follows the published traffic 6x6 column: 32-unit ReLU MLPs, a
softmax output on the high-level actor, and the "selfishness" initialiser of
Appendix C.

Shape convention throughout: ``(batch, districts, ...)`` where *batch* is a
mini-batch of whole-lattice timesteps, which is how the shared-seed
(synchronised) replay sampling of Appendix C is realised here - see
:mod:`chapter4_qolu.baselines.ltos`.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from chapter4_qolu.baselines.districts import N_SLOTS

__all__ = ["DGNQNetwork", "MultiHeadGraphAttention", "SharingActor"]


class MultiHeadGraphAttention(nn.Module):
    """One DGN graph-convolution layer: multi-head attention over ``N_i``."""

    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        if dim < 1 or heads < 1:
            raise ValueError("dim and heads must be positive")
        if dim % heads != 0:
            raise ValueError(f"dim {dim} must be divisible by heads {heads}")
        self.dim = int(dim)
        self.heads = int(heads)
        self.head_dim = self.dim // self.heads
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.output = nn.Linear(dim, dim)

    def forward(
        self,
        features: torch.Tensor,
        neighbour_index: torch.Tensor,
        neighbour_mask: torch.Tensor,
    ) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError("features must have shape (batch, districts, dim)")
        batch, districts, _ = features.shape
        slots = neighbour_index.shape[1]

        neighbours = features[:, neighbour_index]  # (B, K, S, D)
        queries = self.query(features).view(batch, districts, self.heads, self.head_dim)
        keys = self.key(neighbours).view(
            batch, districts, slots, self.heads, self.head_dim
        )
        values = self.value(neighbours).view(
            batch, districts, slots, self.heads, self.head_dim
        )

        logits = torch.einsum("bkhd,bkshd->bkhs", queries, keys) / math.sqrt(
            self.head_dim
        )
        mask = neighbour_mask.view(1, districts, 1, slots)
        logits = logits.masked_fill(mask == 0.0, float("-inf"))
        attention = torch.softmax(logits, dim=-1)
        mixed = torch.einsum("bkhs,bkshd->bkhd", attention, values)
        return torch.relu(self.output(mixed.reshape(batch, districts, self.dim)))


class DGNQNetwork(nn.Module):
    """``pi_i : (o_i, w_i^in) -> a_i`` as a Q-function over convert / skip.

    The encoder sees ``[o_i ; w_i^in]`` because LToS's low-level policy is
    explicitly conditioned on the incoming sharing weights; the Q head then
    scores each *candidate parcel* inside the district, which is the discrete
    decision this environment offers.
    """

    def __init__(
        self,
        node_dim: int,
        parcel_dim: int,
        hidden_units: int = 32,
        heads: int = 8,
        attention_layers: int = 2,
        encoder_layers: int = 1,
        n_actions: int = 2,
    ) -> None:
        super().__init__()
        if attention_layers < 1 or encoder_layers < 1:
            raise ValueError("attention_layers and encoder_layers must be positive")
        layers: list[nn.Module] = []
        width = node_dim + N_SLOTS
        for _ in range(encoder_layers):
            layers.append(nn.Linear(width, hidden_units))
            layers.append(nn.ReLU(inplace=True))
            width = hidden_units
        self.encoder = nn.Sequential(*layers)
        self.attention = nn.ModuleList(
            MultiHeadGraphAttention(hidden_units, heads)
            for _ in range(attention_layers)
        )
        # DGN feeds the concatenation of the encoder and every convolution
        # layer to the Q head.
        head_input = hidden_units * (1 + attention_layers) + parcel_dim
        self.head = nn.Sequential(
            nn.Linear(head_input, hidden_units),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_units, n_actions),
        )

    def district_embeddings(
        self,
        node_features: torch.Tensor,
        sharing_in: torch.Tensor,
        neighbour_index: torch.Tensor,
        neighbour_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encoder output concatenated with every attention layer's output."""
        hidden = self.encoder(torch.cat([node_features, sharing_in], dim=-1))
        outputs = [hidden]
        for layer in self.attention:
            hidden = layer(hidden, neighbour_index, neighbour_mask)
            outputs.append(hidden)
        return torch.cat(outputs, dim=-1)

    def forward(
        self,
        node_features: torch.Tensor,
        sharing_in: torch.Tensor,
        parcel_features: torch.Tensor,
        parcel_districts: torch.Tensor,
        neighbour_index: torch.Tensor,
        neighbour_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Q-values ``(batch, parcels, n_actions)`` for the sampled candidates."""
        embeddings = self.district_embeddings(
            node_features, sharing_in, neighbour_index, neighbour_mask
        )
        gathered = torch.gather(
            embeddings,
            1,
            parcel_districts.unsqueeze(-1).expand(-1, -1, embeddings.shape[-1]),
        )
        return self.head(torch.cat([gathered, parcel_features], dim=-1))


class SharingActor(nn.Module):
    """``phi_i : o_i -> w_i^out``, a softmax over ``N_i`` (the high level).

    The last layer implements Appendix C's **selfishness initialiser**: its
    weights start at zero and its bias is set so that the softmax over a
    district's present slots begins at ``selfishness`` on the self-loop, with
    the remainder split evenly over the neighbours.

    Design note: the bias is a single shared vector (the actor's parameters are
    shared across districts, see :mod:`chapter4_qolu.baselines.ltos`), so it is
    solved for the *interior* neighbour count ``interior_neighbours``.  Boundary
    districts, which have fewer neighbours to share with, therefore start
    slightly more selfish than the target - they cannot start *less* selfish,
    which is the property Appendix C actually relies on.
    """

    def __init__(
        self,
        node_dim: int,
        hidden_units: int = 32,
        selfishness: float = 0.8,
        interior_neighbours: int = 4,
    ) -> None:
        super().__init__()
        if not 0.0 < selfishness < 1.0:
            raise ValueError("selfishness must lie in (0, 1)")
        if interior_neighbours < 1:
            raise ValueError("interior_neighbours must be positive")
        self.body = nn.Sequential(
            nn.Linear(node_dim, hidden_units),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(inplace=True),
        )
        self.logits = nn.Linear(hidden_units, N_SLOTS)
        with torch.no_grad():
            self.logits.weight.zero_()
            bias = torch.zeros(N_SLOTS)
            bias[0] = math.log(
                selfishness / (1.0 - selfishness) * float(interior_neighbours)
            )
            self.logits.bias.copy_(bias)

    def forward(
        self, node_features: torch.Tensor, neighbour_mask: torch.Tensor
    ) -> torch.Tensor:
        """Row-stochastic outgoing weights, zero on absent neighbours."""
        logits = self.logits(self.body(node_features))
        mask = neighbour_mask.view(1, *neighbour_mask.shape)
        return torch.softmax(logits.masked_fill(mask == 0.0, float("-inf")), dim=-1)
