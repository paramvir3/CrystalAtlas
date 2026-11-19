"""Exact-grammar ancestral sampling driven by neural composition logits."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from .._optional import require_torch
from ..composition import Composition, CompositionToken
from ..tokenization import CompositionTokenCodec
from .composition import CanonicalCompositionTransformer

torch = require_torch()


@torch.no_grad()
def sample_composition(
    model: CanonicalCompositionTransformer,
    codec: CompositionTokenCodec,
    *,
    support: Iterable[int] | None = None,
    temperature: float = 1.0,
    generator: torch.Generator | None = None,
    device: torch.device | None = None,
) -> Composition:
    """Draw from the model with the exact live-prefix oracle at every step.

    Supplying ``support`` clamps the support-first prefix and therefore samples
    the model's true stoichiometry conditional for that exact element set.
    """

    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    grammar = codec.grammar
    prefix: tuple[CompositionToken, ...] = (
        () if support is None else grammar.support_prefix(support)
    )
    target_device = device or next(model.parameters()).device
    while not grammar.is_terminal(prefix):
        candidate_ids_np, candidate_types_np = codec.allowed_arrays(prefix)
        prefix_pairs = [codec.encode_token(token) for token in prefix]
        prefix_ids = torch.as_tensor(
            [[pair[0] for pair in prefix_pairs]], dtype=torch.long, device=target_device
        )
        prefix_types = torch.as_tensor(
            [[pair[1] for pair in prefix_pairs]], dtype=torch.long, device=target_device
        )
        candidate_ids = torch.as_tensor(
            candidate_ids_np, dtype=torch.long, device=target_device
        )
        candidate_types = torch.as_tensor(
            candidate_types_np, dtype=torch.long, device=target_device
        )
        selected = model.sample_next(
            prefix_ids,
            prefix_types,
            candidate_ids,
            candidate_types,
            legal_mask=torch.ones(
                (1, candidate_ids.shape[0]), dtype=torch.bool, device=target_device
            ),
            temperature=temperature,
            generator=generator,
        )
        selected_id = int(selected.item())
        match = np.flatnonzero(candidate_ids_np == selected_id)
        if match.size != 1:
            raise RuntimeError("typed grammar candidates did not have unique value IDs")
        selected_type = int(candidate_types_np[int(match[0])])
        prefix += (codec.decode_token(selected_id, selected_type),)
    return grammar.deserialize(prefix)


__all__ = ["sample_composition"]
