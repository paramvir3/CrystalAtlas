"""Composition models used by the discrete and continuous crystal kernels.

This module implements equations (3.3a)--(3.3b) and (3.3).  It deliberately
does *not* implement the canonical-composition grammar: a deterministic live
prefix oracle owns legality, while the neural network only supplies scores.
"""

from __future__ import annotations

from dataclasses import dataclass

from .._optional import require_torch
from ..elements import (
    ELEMENT_SYMBOLS,
    LONG_FORM_COLUMNS,
    NOBLE_GAS_ATOMIC_NUMBERS,
    PERIODIC_BLOCKS,
    periodic_coordinates,
)

torch = require_torch()
Tensor = torch.Tensor
nn = torch.nn
F = torch.nn.functional

ELEMENT_EMBEDDINGS: tuple[str, ...] = ("learned-v1", "periodic-v1")


class PeriodicElementEmbedding(nn.Module):
    """Element embedding tied through the periodic table (``periodic-v1``).

    ``e(Z) = P[period(Z)] + G[column(Z)] + B[block(Z)]`` and ``e(0) = 0`` for
    padding, with the exact long-form coordinates of
    :func:`crystalatlas.elements.periodic_coordinates`.  Period and column
    identify ``Z``, so a downstream network can still tell every element apart;
    what changes is the parameter sharing.  ``learned-v1`` gives each element a
    free vector, trained only by the records that contain it.  Here a column
    vector is trained by every element of that column, a period vector by its
    whole row, and a block vector by the block (the term that ties the
    lanthanides, whose columns hold two elements each).  There is deliberately
    no per-element residual: with one, an adaptive optimizer would fit the
    residual as fast as the shared terms and the prior would be lost.

    Each table starts as ``N(0, 1/3)``, so every entry of ``e(Z)`` has unit
    variance like the default ``nn.Embedding`` it replaces.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if not 2 <= num_embeddings <= len(ELEMENT_SYMBOLS) + 1:
            raise ValueError(
                "num_embeddings must cover padding and atomic numbers 1..Z_max <= "
                f"{len(ELEMENT_SYMBOLS)}"
            )
        coordinates = [periodic_coordinates(z) for z in range(1, num_embeddings)]
        for name, values in (
            ("_period_index", [item.period for item in coordinates]),
            ("_column_index", [item.column for item in coordinates]),
            ("_block_index", [1 + PERIODIC_BLOCKS.index(item.block) for item in coordinates]),
        ):
            # Exact functions of Z, so they are not checkpoint state.  Index 0 is padding.
            self.register_buffer(
                name, torch.tensor([0, *values], dtype=torch.long), persistent=False
            )
        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        self.period = nn.Embedding(
            len(NOBLE_GAS_ATOMIC_NUMBERS) + 1, embedding_dim, padding_idx=0
        )
        self.column = nn.Embedding(LONG_FORM_COLUMNS + 1, embedding_dim, padding_idx=0)
        self.block = nn.Embedding(len(PERIODIC_BLOCKS) + 1, embedding_dim, padding_idx=0)
        with torch.no_grad():
            for table in (self.period, self.column, self.block):
                table.weight.mul_(3.0**-0.5)

    def forward(self, atomic_numbers: Tensor) -> Tensor:
        return (
            self.period(self._period_index[atomic_numbers])
            + self.column(self._column_index[atomic_numbers])
            + self.block(self._block_index[atomic_numbers])
        )


def make_element_embedding(kind: str, num_embeddings: int, embedding_dim: int) -> nn.Module:
    """The element-embedding module of ``kind`` (one of :data:`ELEMENT_EMBEDDINGS`).

    ``learned-v1`` is the historical free table: the same constructor call, so
    its parameter name, initial values and random-stream use are unchanged.
    """

    if kind == "learned-v1":
        return nn.Embedding(num_embeddings, embedding_dim, padding_idx=0)
    if kind == "periodic-v1":
        return PeriodicElementEmbedding(num_embeddings, embedding_dim)
    raise ValueError(f"element embedding must be one of {ELEMENT_EMBEDDINGS}")


def _as_batched_pair(atomic_numbers: Tensor, counts: Tensor) -> tuple[Tensor, Tensor, bool]:
    if atomic_numbers.shape != counts.shape:
        raise ValueError("atomic_numbers and counts must have identical shapes")
    if atomic_numbers.ndim == 1:
        return atomic_numbers.unsqueeze(0), counts.unsqueeze(0), True
    if atomic_numbers.ndim != 2:
        raise ValueError("composition tensors must have shape [K] or [B, K]")
    return atomic_numbers, counts, False


class CompositionEncoder(nn.Module):
    """Permutation-invariant DeepSets encoder for reduced compositions.

    Padding uses ``atomic_number == 0`` and ``count == 0``.  Positive entries
    are encoded as

    ``rho(sum_Z psi(element_embedding[Z], log(1 + c_Z), descriptor_Z))``.

    Selector and flow code should instantiate separate objects; no learned
    weights are shared unless an experiment explicitly chooses that ablation.
    """

    def __init__(
        self,
        output_dim: int,
        *,
        max_atomic_number: int = 118,
        element_dim: int = 64,
        descriptor_dim: int = 0,
        hidden_dim: int = 128,
        element_embedding: str = "learned-v1",
    ) -> None:
        super().__init__()
        if min(output_dim, element_dim, hidden_dim) <= 0:
            raise ValueError("encoder dimensions must be positive")
        if descriptor_dim < 0:
            raise ValueError("descriptor_dim must be nonnegative")
        self.max_atomic_number = int(max_atomic_number)
        self.descriptor_dim = int(descriptor_dim)
        self.element_embedding = make_element_embedding(
            element_embedding, self.max_atomic_number + 1, element_dim
        )
        self.psi = nn.Sequential(
            nn.Linear(element_dim + 1 + descriptor_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.rho = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self,
        atomic_numbers: Tensor,
        counts: Tensor,
        descriptors: Tensor | None = None,
    ) -> Tensor:
        atomic_numbers, counts, squeeze = _as_batched_pair(atomic_numbers, counts)
        if torch.any(atomic_numbers < 0) or torch.any(atomic_numbers > self.max_atomic_number):
            raise ValueError("atomic number outside the configured range")
        if torch.any(counts < 0):
            raise ValueError("composition counts must be nonnegative")

        if descriptors is None:
            descriptors = counts.new_zeros(
                (*counts.shape, self.descriptor_dim), dtype=self.psi[0].weight.dtype
            )
        elif squeeze and descriptors.ndim == 2:
            descriptors = descriptors.unsqueeze(0)
        expected = (*counts.shape, self.descriptor_dim)
        if tuple(descriptors.shape) != expected:
            raise ValueError(f"descriptors must have shape {expected}")

        numbers = atomic_numbers.to(dtype=torch.long)
        count_values = counts.to(dtype=self.psi[0].weight.dtype)
        descriptor_values = descriptors.to(
            device=numbers.device, dtype=self.psi[0].weight.dtype
        )
        mask = (numbers > 0) & (count_values > 0)
        pair_features = torch.cat(
            (
                self.element_embedding(numbers),
                torch.log1p(count_values).unsqueeze(-1),
                descriptor_values,
            ),
            dim=-1,
        )
        encoded = self.psi(pair_features) * mask.unsqueeze(-1)
        pooled = encoded.sum(dim=-2)
        output = self.rho(pooled)
        return output.squeeze(0) if squeeze else output


def apply_legal_mask(logits: Tensor, legal_mask: Tensor | None) -> Tensor:
    """Set illegal categorical logits to ``-inf`` after validating support."""

    if legal_mask is None:
        return logits
    if legal_mask.shape != logits.shape:
        try:
            legal_mask = torch.broadcast_to(legal_mask, logits.shape)
        except RuntimeError as exc:
            raise ValueError("legal_mask is not broadcastable to logits") from exc
    legal_mask = legal_mask.to(device=logits.device, dtype=torch.bool)
    if torch.any(~legal_mask.any(dim=-1)):
        raise ValueError("every categorical decision must retain a legal action")
    return logits.masked_fill(~legal_mask, -torch.inf)


class _CandidateScorer(nn.Module):
    """Bounded additive scorer implementing ``MLP(H, E_x)``."""

    def __init__(self, model_dim: int) -> None:
        super().__init__()
        self.hidden = nn.Linear(model_dim, model_dim, bias=False)
        self.candidate = nn.Linear(model_dim, model_dim, bias=True)
        self.readout = nn.Linear(model_dim, 1, bias=False)

    def forward(self, hidden: Tensor, candidate_embeddings: Tensor) -> Tensor:
        joint = torch.tanh(
            self.hidden(hidden).unsqueeze(-2) + self.candidate(candidate_embeddings)
        )
        return self.readout(joint).squeeze(-1)


@dataclass(frozen=True)
class CompositionTransformerConfig:
    vocab_size: int
    num_token_types: int
    model_dim: int = 192
    num_heads: int = 6
    num_layers: int = 4
    feedforward_dim: int = 512
    max_sequence_length: int = 256
    dropout: float = 0.0


class CanonicalCompositionTransformer(nn.Module):
    """Causal scorer for the unique typed composition token string.

    The external BOS vector initializes a context.  ``next_logits`` receives
    only an already-generated prefix, so the target token is never part of the
    context that predicts itself.  Grammar legality is supplied by
    ``legal_mask`` from the exact live-prefix oracle.
    """

    def __init__(self, config: CompositionTransformerConfig) -> None:
        super().__init__()
        if config.vocab_size <= 0 or config.num_token_types <= 0:
            raise ValueError("vocabulary and token-type sizes must be positive")
        if config.model_dim % config.num_heads:
            raise ValueError("model_dim must be divisible by num_heads")
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.model_dim)
        self.type_embedding = nn.Embedding(config.num_token_types, config.model_dim)
        self.position_embedding = nn.Embedding(config.max_sequence_length + 1, config.model_dim)
        self.bos = nn.Parameter(torch.zeros(config.model_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=config.model_dim,
            nhead=config.num_heads,
            dim_feedforward=config.feedforward_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=config.num_layers, norm=nn.LayerNorm(config.model_dim)
        )
        self.scorer = _CandidateScorer(config.model_dim)
        nn.init.normal_(self.bos, std=config.model_dim**-0.5)

    def _validate_tokens(self, tokens: Tensor, token_types: Tensor) -> None:
        if tokens.shape != token_types.shape or tokens.ndim != 2:
            raise ValueError("tokens and token_types must both have shape [B, T]")
        if tokens.shape[1] > self.config.max_sequence_length:
            raise ValueError("prefix exceeds max_sequence_length")
        if torch.any(tokens < 0) or torch.any(tokens >= self.config.vocab_size):
            raise ValueError("token id outside configured vocabulary")
        if torch.any(token_types < 0) or torch.any(token_types >= self.config.num_token_types):
            raise ValueError("token type outside configured range")

    def _run(self, inputs: Tensor) -> Tensor:
        length = inputs.shape[1]
        positions = torch.arange(length, device=inputs.device)
        inputs = inputs + self.position_embedding(positions).unsqueeze(0)
        causal_mask = torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=inputs.device),
            diagonal=1,
        )
        return self.transformer(inputs, mask=causal_mask)

    def encode_prefix(self, tokens: Tensor, token_types: Tensor) -> Tensor:
        """Return ``H_omega(prefix)`` from BOS and the supplied prefix."""

        self._validate_tokens(tokens, token_types)
        batch = tokens.shape[0]
        bos = self.bos.view(1, 1, -1).expand(batch, 1, -1)
        embedded = self.token_embedding(tokens) + self.type_embedding(token_types)
        states = self._run(torch.cat((bos, embedded), dim=1))
        return states[:, -1]

    def _candidate_embeddings(self, candidate_ids: Tensor, candidate_types: Tensor) -> Tensor:
        if candidate_ids.shape != candidate_types.shape:
            try:
                candidate_ids, candidate_types = torch.broadcast_tensors(
                    candidate_ids, candidate_types
                )
            except RuntimeError as exc:
                raise ValueError("candidate ids/types are not broadcastable") from exc
        if torch.any(candidate_ids < 0) or torch.any(candidate_ids >= self.config.vocab_size):
            raise ValueError("candidate token outside configured vocabulary")
        if torch.any(candidate_types < 0) or torch.any(
            candidate_types >= self.config.num_token_types
        ):
            raise ValueError("candidate type outside configured range")
        return self.token_embedding(candidate_ids) + self.type_embedding(candidate_types)

    def next_logits(
        self,
        prefix_tokens: Tensor,
        prefix_types: Tensor,
        candidate_ids: Tensor,
        candidate_types: Tensor,
        *,
        legal_mask: Tensor | None = None,
    ) -> Tensor:
        """Score candidates after a prefix, with no target-token leakage."""

        hidden = self.encode_prefix(prefix_tokens, prefix_types)
        candidates = self._candidate_embeddings(candidate_ids, candidate_types)
        if candidates.ndim == 2:
            candidates = candidates.unsqueeze(0).expand(hidden.shape[0], -1, -1)
        elif candidates.shape[0] != hidden.shape[0]:
            raise ValueError("batched candidates must match prefix batch size")
        return apply_legal_mask(self.scorer(hidden, candidates), legal_mask)

    def teacher_forced_logits(
        self,
        tokens: Tensor,
        token_types: Tensor,
        *,
        legal_mask: Tensor | None = None,
    ) -> Tensor:
        """Return ``[B,T,V]`` logits using contexts ``BOS,tokens[:t]``."""

        self._validate_tokens(tokens, token_types)
        batch, length = tokens.shape
        if length == 0:
            return self.bos.new_empty(batch, 0, self.config.vocab_size)
        bos = self.bos.view(1, 1, -1).expand(batch, 1, -1)
        shifted = torch.cat(
            (
                bos,
                self.token_embedding(tokens[:, :-1]) + self.type_embedding(token_types[:, :-1]),
            ),
            dim=1,
        )
        hidden = self._run(shifted)
        ids = torch.arange(self.config.vocab_size, device=tokens.device)
        ids = ids.view(1, 1, -1).expand(batch, length, -1)
        types = token_types.unsqueeze(-1).expand_as(ids)
        candidates = self._candidate_embeddings(ids, types)
        return apply_legal_mask(self.scorer(hidden, candidates), legal_mask)

    def negative_log_likelihood(
        self,
        tokens: Tensor,
        token_types: Tensor,
        *,
        legal_mask: Tensor | None = None,
        sequence_mask: Tensor | None = None,
        reduction: str = "mean",
    ) -> Tensor:
        logits = self.teacher_forced_logits(tokens, token_types, legal_mask=legal_mask)
        working_logits = (
            logits.float() if logits.dtype in {torch.float16, torch.bfloat16} else logits
        )
        losses = F.cross_entropy(
            working_logits.reshape(-1, working_logits.shape[-1]),
            tokens.reshape(-1),
            reduction="none",
        ).reshape_as(tokens)
        if sequence_mask is not None:
            if sequence_mask.shape != tokens.shape:
                raise ValueError("sequence_mask must have shape [B, T]")
            selected = losses[sequence_mask.to(dtype=torch.bool, device=losses.device)]
        else:
            selected = losses.reshape(-1)
        if reduction == "none":
            return losses
        if selected.numel() == 0:
            raise ValueError("negative-log-likelihood has no selected tokens")
        if reduction == "sum":
            return selected.sum()
        if reduction == "mean":
            return selected.mean()
        raise ValueError("reduction must be 'none', 'sum', or 'mean'")

    def forward(
        self,
        tokens: Tensor,
        token_types: Tensor,
        *,
        legal_mask: Tensor | None = None,
        sequence_mask: Tensor | None = None,
    ) -> Tensor:
        """DDP-compatible per-token exact grammar NLL.

        Reduction is deliberately left to the empirical-measure layer, which
        knows whether records or unique compositions are the sampling units.
        """

        return self.negative_log_likelihood(
            tokens,
            token_types,
            legal_mask=legal_mask,
            sequence_mask=sequence_mask,
            reduction="none",
        )

    @torch.no_grad()
    def sample_next(
        self,
        prefix_tokens: Tensor,
        prefix_types: Tensor,
        candidate_ids: Tensor,
        candidate_types: Tensor,
        *,
        legal_mask: Tensor,
        temperature: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        logits = self.next_logits(
            prefix_tokens,
            prefix_types,
            candidate_ids,
            candidate_types,
            legal_mask=legal_mask,
        )
        probabilities = torch.softmax(logits / temperature, dim=-1)
        sampled_index = torch.multinomial(probabilities, 1, generator=generator).squeeze(-1)
        candidates = candidate_ids
        if candidates.ndim == 1:
            candidates = candidates.unsqueeze(0).expand(logits.shape[0], -1)
        return candidates.gather(-1, sampled_index.unsqueeze(-1)).squeeze(-1)
