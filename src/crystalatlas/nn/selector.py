"""Arithmetic-completion-masked neural cover selector.

The selector implements the score parameterization in equations (3.4)--(3.6).
It never decides crystallographic legality: exact completion masks produced by
the frozen catalogue/arithmetic layer are mandatory at each sampled decision.
"""

from __future__ import annotations

from dataclasses import dataclass

from .._optional import require_torch
from .composition import CompositionEncoder, apply_legal_mask, make_element_embedding

torch = require_torch()
Tensor = torch.Tensor
nn = torch.nn


@dataclass(frozen=True)
class CoverSelectorConfig:
    num_nu_choices: int
    num_space_groups: int = 230
    row_feature_dim: int = 32
    model_dim: int = 192
    num_heads: int = 6
    num_layers: int = 3
    feedforward_dim: int = 384
    max_atomic_number: int = 118
    dropout: float = 0.0
    group_descriptor_dim: int = 0
    """Operation-derived group descriptors (review m1); 0 keeps the bare embedding."""
    row_geometry_features: bool = False
    """Exact-chart row block of ``row_geometry_descriptor`` (review 2026-10-29, M1).

    ``row_feature_dim`` must include its ``ROW_GEOMETRY_DESCRIPTOR_DIM`` entries."""
    element_embedding: str = "learned-v1"
    """``learned-v1`` (free table) or ``periodic-v1`` (periodic-table sharing)."""


class CountVectorEncoder(nn.Module):
    """Permutation-invariant embedding of an element-count action vector."""

    def __init__(
        self,
        *,
        max_atomic_number: int,
        model_dim: int,
        hidden_dim: int,
        element_embedding: str = "learned-v1",
    ) -> None:
        super().__init__()
        self.max_atomic_number = int(max_atomic_number)
        self.element_embedding = make_element_embedding(
            element_embedding, max_atomic_number + 1, model_dim
        )
        self.chi = nn.Sequential(
            nn.Linear(model_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, model_dim),
        )

    def forward(self, atomic_numbers: Tensor, action_counts: Tensor) -> Tensor:
        """Encode counts with shapes ``[B,...,K]`` and species ``[B,K]``."""

        if atomic_numbers.ndim == 1:
            atomic_numbers = atomic_numbers.unsqueeze(0)
        if atomic_numbers.ndim != 2 or action_counts.ndim < 2:
            raise ValueError("species must be [B,K] and actions [B,...,K]")
        if (
            action_counts.shape[0] != atomic_numbers.shape[0]
            or action_counts.shape[-1] != atomic_numbers.shape[-1]
        ):
            raise ValueError("action vectors must use the supplied composition species")
        if torch.any(action_counts < 0):
            raise ValueError("row-action counts must be nonnegative")
        if torch.any(atomic_numbers < 0) or torch.any(
            atomic_numbers > self.max_atomic_number
        ):
            raise ValueError("atomic number outside configured range")

        numbers = atomic_numbers.to(dtype=torch.long)
        element = self.element_embedding(numbers)
        extra_axes = action_counts.ndim - 2
        element = element.view(
            element.shape[0], *([1] * extra_axes), element.shape[1], element.shape[2]
        ).expand(*action_counts.shape, element.shape[-1])
        mask = (numbers > 0).view(
            numbers.shape[0], *([1] * extra_axes), numbers.shape[1], 1
        )
        counts = action_counts.to(dtype=element.dtype)
        features = torch.cat((element, torch.log1p(counts).unsqueeze(-1)), dim=-1)
        return (self.chi(features) * mask).sum(dim=-2)


class _ActionScorer(nn.Module):
    def __init__(self, model_dim: int) -> None:
        super().__init__()
        self.hidden = nn.Linear(model_dim, model_dim, bias=False)
        self.row = nn.Linear(model_dim, model_dim, bias=False)
        self.action = nn.Linear(model_dim, model_dim, bias=True)
        self.readout = nn.Linear(model_dim, 1, bias=False)

    def forward(
        self, hidden: Tensor, row_embedding: Tensor, actions: Tensor
    ) -> Tensor:
        value = torch.tanh(
            self.hidden(hidden).unsqueeze(-2)
            + self.row(row_embedding).unsqueeze(-2)
            + self.action(actions)
        )
        return self.readout(value).squeeze(-1)


class ArithmeticCoverSelector(nn.Module):
    """Causal selector for ``nu``, pinned group, and Wyckoff row actions.

    ``row_features`` must be deterministic operation-derived descriptors
    (multiplicity, site-symmetry information, and free dimension).  The module
    cannot prove that semantic contract and intentionally has no learned
    arbitrary Wyckoff-letter or row-index embedding.
    """

    def __init__(
        self, config: CoverSelectorConfig, group_descriptors: Tensor | None = None
    ) -> None:
        super().__init__()
        if config.num_nu_choices <= 0 or config.num_space_groups <= 0:
            raise ValueError("selector category counts must be positive")
        if config.model_dim % config.num_heads:
            raise ValueError("model_dim must be divisible by num_heads")
        self.config = config
        self.composition_encoder = CompositionEncoder(
            config.model_dim,
            max_atomic_number=config.max_atomic_number,
            element_dim=config.model_dim // 2,
            hidden_dim=config.model_dim,
            element_embedding=config.element_embedding,
        )
        self.nu_embedding = nn.Embedding(config.num_nu_choices, config.model_dim)
        self.group_embedding = nn.Embedding(
            config.num_space_groups, config.model_dim
        )
        self.nu_head = nn.Sequential(
            nn.LayerNorm(config.model_dim),
            nn.Linear(config.model_dim, config.num_nu_choices),
        )
        self.group_head = nn.Sequential(
            nn.LayerNorm(config.model_dim),
            nn.Linear(config.model_dim, config.model_dim),
            nn.SiLU(),
            nn.Linear(config.model_dim, config.num_space_groups),
        )
        self.row_encoder = nn.Sequential(
            nn.Linear(config.row_feature_dim, config.model_dim),
            nn.SiLU(),
            nn.Linear(config.model_dim, config.model_dim),
        )
        self.action_encoder = CountVectorEncoder(
            max_atomic_number=config.max_atomic_number,
            model_dim=config.model_dim,
            hidden_dim=config.model_dim,
            element_embedding=config.element_embedding,
        )
        layer = nn.TransformerEncoderLayer(
            d_model=config.model_dim,
            nhead=config.num_heads,
            dim_feedforward=config.feedforward_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.row_transformer = nn.TransformerEncoder(
            layer, num_layers=config.num_layers, norm=nn.LayerNorm(config.model_dim)
        )
        self.action_scorer = _ActionScorer(config.model_dim)
        if config.group_descriptor_dim:
            expected = (config.num_space_groups, config.group_descriptor_dim)
            if group_descriptors is None or tuple(group_descriptors.shape) != expected:
                raise ValueError(f"group_descriptors must have shape {expected}")
            # Deterministic from the certified catalogue, hence not checkpointed.
            self.register_buffer(
                "group_descriptors", group_descriptors.detach().clone(), persistent=False
            )
            self.group_descriptor_encoder = nn.Sequential(
                nn.Linear(config.group_descriptor_dim, config.model_dim),
                nn.SiLU(),
                nn.Linear(config.model_dim, config.model_dim),
            )
            self.group_query = nn.Linear(config.model_dim, config.model_dim)
        elif group_descriptors is not None:
            raise ValueError("group_descriptors require group_descriptor_dim > 0")

    def _group_vectors(self) -> Tensor:
        return self.group_descriptor_encoder(
            self.group_descriptors.to(dtype=self.group_embedding.weight.dtype)
        )

    def encode_composition(
        self,
        atomic_numbers: Tensor,
        reduced_counts: Tensor,
        descriptors: Tensor | None = None,
    ) -> Tensor:
        context = self.composition_encoder(
            atomic_numbers, reduced_counts, descriptors=descriptors
        )
        return context.unsqueeze(0) if context.ndim == 1 else context

    def nu_logits(
        self, composition_context: Tensor, *, legal_mask: Tensor
    ) -> Tensor:
        if composition_context.ndim != 2:
            raise ValueError("composition_context must have shape [B,D]")
        return apply_legal_mask(self.nu_head(composition_context), legal_mask)

    def group_logits(
        self,
        composition_context: Tensor,
        nu_indices: Tensor,
        *,
        legal_mask: Tensor,
    ) -> Tensor:
        if composition_context.ndim != 2 or nu_indices.ndim != 1:
            raise ValueError("composition context is [B,D] and nu indices are [B]")
        if composition_context.shape[0] != nu_indices.shape[0]:
            raise ValueError("batch dimensions do not agree")
        hidden = composition_context + self.nu_embedding(nu_indices)
        logits = self.group_head(hidden)
        if self.config.group_descriptor_dim:
            # Groups with similar operation-derived descriptors share evidence.
            logits = logits + self.group_query(hidden) @ self._group_vectors().T / (
                self.config.model_dim**0.5
            )
        return apply_legal_mask(logits, legal_mask)

    def row_logits(
        self,
        composition_context: Tensor,
        nu_indices: Tensor,
        group_indices: Tensor,
        atomic_numbers: Tensor,
        row_features: Tensor,
        previous_actions: Tensor,
        candidate_actions: Tensor,
        *,
        legal_mask: Tensor,
    ) -> Tensor:
        """Score all completion-safe row actions.

        Shapes are ``row_features [B,R,F]``, ``previous_actions [B,R,K]``,
        ``candidate_actions [B,R,A,K]``, and ``legal_mask [B,R,A]``.
        ``previous_actions[:,r]`` is the realized action at row ``r``; it is
        shifted internally, hence the score for row ``r`` sees only rows
        strictly before ``r``.
        """

        if composition_context.ndim != 2:
            raise ValueError("composition_context must have shape [B,D]")
        if row_features.ndim != 3:
            raise ValueError("row_features must have shape [B,R,F]")
        if previous_actions.ndim != 3 or candidate_actions.ndim != 4:
            raise ValueError("row actions have shapes [B,R,K] and [B,R,A,K]")
        batch, rows, _ = row_features.shape
        if (
            composition_context.shape[0] != batch
            or nu_indices.shape != (batch,)
            or group_indices.shape != (batch,)
            or previous_actions.shape[:2] != (batch, rows)
            or candidate_actions.shape[:2] != (batch, rows)
        ):
            raise ValueError("selector batch/row dimensions do not agree")
        if rows == 0:
            return composition_context.new_empty(batch, 0, candidate_actions.shape[2])

        row_embedding = self.row_encoder(row_features)
        action_embedding = self.action_encoder(atomic_numbers, previous_actions)
        shifted = torch.cat(
            (torch.zeros_like(action_embedding[:, :1]), action_embedding[:, :-1]),
            dim=1,
        )
        condition = (
            composition_context
            + self.nu_embedding(nu_indices)
            + self.group_embedding(group_indices)
        )
        if self.config.group_descriptor_dim:
            condition = condition + self._group_vectors()[group_indices]
        transformer_input = row_embedding + shifted + condition.unsqueeze(1)
        causal_mask = torch.triu(
            torch.ones(rows, rows, dtype=torch.bool, device=row_features.device),
            diagonal=1,
        )
        hidden = self.row_transformer(transformer_input, mask=causal_mask)
        candidates = self.action_encoder(atomic_numbers, candidate_actions)
        logits = self.action_scorer(hidden, row_embedding, candidates)
        return apply_legal_mask(logits, legal_mask)

    @staticmethod
    def selected_log_probability(logits: Tensor, selected: Tensor) -> Tensor:
        """Return normalized masked log-probabilities of selected actions."""

        if selected.shape != logits.shape[:-1]:
            raise ValueError("selected indices must match logits batch shape")
        log_probabilities = torch.log_softmax(logits, dim=-1)
        return log_probabilities.gather(-1, selected.unsqueeze(-1)).squeeze(-1)

    def forward(
        self,
        atomic_numbers: Tensor,
        reduced_counts: Tensor,
        nu_target_indices: Tensor,
        group_target_indices: Tensor,
        row_features: Tensor,
        selected_actions: Tensor,
        candidate_actions: Tensor,
        *,
        nu_legal_mask: Tensor,
        group_legal_mask: Tensor,
        candidate_legal_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """DDP-compatible teacher-forced logits for the full exact cover path."""

        context = self.encode_composition(atomic_numbers, reduced_counts)
        nu_logits = self.nu_logits(context, legal_mask=nu_legal_mask)
        group_logits = self.group_logits(
            context,
            nu_target_indices,
            legal_mask=group_legal_mask,
        )
        row_logits = self.row_logits(
            context,
            nu_target_indices,
            group_target_indices,
            atomic_numbers,
            row_features,
            selected_actions,
            candidate_actions,
            legal_mask=candidate_legal_mask,
        )
        return nu_logits, group_logits, row_logits
