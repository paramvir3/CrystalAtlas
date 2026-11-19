from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from crystalatlas.composition import CompositionDomain, CompositionGrammar  # noqa: E402
from crystalatlas.nn.composition import (  # noqa: E402
    CanonicalCompositionTransformer,
    CompositionTransformerConfig,
)
from crystalatlas.nn.sampling import sample_composition  # noqa: E402
from crystalatlas.tokenization import CompositionTokenCodec  # noqa: E402

pytestmark = pytest.mark.ml


def test_neural_sampling_never_leaves_exact_grammar() -> None:
    grammar = CompositionGrammar(CompositionDomain((3, 8, 14), 3, 7))
    codec = CompositionTokenCodec(grammar)
    model = CanonicalCompositionTransformer(
        CompositionTransformerConfig(
            vocab_size=codec.vocab_size,
            num_token_types=codec.num_token_types,
            model_dim=24,
            num_heads=4,
            num_layers=2,
            feedforward_dim=48,
            max_sequence_length=codec.max_sequence_length,
        )
    ).eval()
    generator = torch.Generator().manual_seed(91)
    for _ in range(20):
        composition = sample_composition(model, codec, generator=generator)
        assert grammar.domain.contains(composition)
    for _ in range(10):
        composition = sample_composition(
            model, codec, support=(3, 14), generator=generator
        )
        assert composition.support == (3, 14)
