from __future__ import annotations

import numpy as np

from crystalatlas.composition import Composition, CompositionDomain, CompositionGrammar
from crystalatlas.tokenization import CompositionTokenCodec


def _codec() -> CompositionTokenCodec:
    return CompositionTokenCodec(
        CompositionGrammar(CompositionDomain((3, 8, 14, 26), 3, 8))
    )


def test_codec_is_bijective_and_mask_matches_every_prefix() -> None:
    codec = _codec()
    for composition in codec.grammar.iter_compositions():
        encoded = codec.encode(composition)
        assert codec.decode(encoded.token_ids, encoded.token_types) == composition
        assert np.all(encoded.legal_mask[
            np.arange(encoded.token_ids.size), encoded.token_ids
        ])
    assert len(codec.fingerprint) == 64


def test_collator_padding_is_excluded_but_numerically_legal() -> None:
    codec = _codec()
    short = Composition(((8, 1),))
    long = Composition(((3, 1), (8, 2), (14, 1)))
    batch = codec.collate((short, long))
    assert batch.token_ids.shape[0] == 2
    assert batch.legal_mask.shape[-1] == codec.vocab_size
    assert np.all(batch.legal_mask.any(axis=-1))
    assert batch.sequence_mask[0].sum() < batch.sequence_mask[1].sum()
