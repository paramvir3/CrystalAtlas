"""Bijective neural IDs for the exact typed composition language.

Token *roles* remain disjoint through a type ID even when integer value IDs are
shared.  EOS is value ID zero; every generated positive integer uses its own
value as the ID.  The codec contains no learned state and is fingerprintable.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256

import numpy as np
import numpy.typing as npt

from .composition import (
    EOS_TOKEN,
    Composition,
    CompositionGrammar,
    CompositionToken,
    TokenKind,
)
from .errors import DomainError

IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]

_TYPE_IDS = {
    TokenKind.SUPPORT_SIZE: 0,
    TokenKind.ELEMENT: 1,
    TokenKind.COEFFICIENT: 2,
    TokenKind.EOS: 3,
}


@dataclass(frozen=True, slots=True)
class EncodedComposition:
    token_ids: IntArray
    token_types: IntArray
    legal_mask: BoolArray

    def __post_init__(self) -> None:
        token_ids = np.asarray(self.token_ids, dtype=np.int64).copy()
        token_types = np.asarray(self.token_types, dtype=np.int64).copy()
        legal_mask = np.asarray(self.legal_mask, dtype=bool).copy()
        if token_ids.ndim != 1 or token_types.shape != token_ids.shape:
            raise DomainError("encoded token values/types must be equal-length vectors")
        if legal_mask.ndim != 2 or legal_mask.shape[0] != token_ids.shape[0]:
            raise DomainError("legal_mask must have shape [sequence,vocabulary]")
        token_ids.setflags(write=False)
        token_types.setflags(write=False)
        legal_mask.setflags(write=False)
        object.__setattr__(self, "token_ids", token_ids)
        object.__setattr__(self, "token_types", token_types)
        object.__setattr__(self, "legal_mask", legal_mask)


@dataclass(frozen=True, slots=True)
class CompositionBatch:
    token_ids: IntArray
    token_types: IntArray
    legal_mask: BoolArray
    sequence_mask: BoolArray


class CompositionTokenCodec:
    """Deterministic bridge between :class:`CompositionGrammar` and a network."""

    num_token_types = 4
    eos_id = 0
    eos_type = _TYPE_IDS[TokenKind.EOS]

    def __init__(self, grammar: CompositionGrammar) -> None:
        if not isinstance(grammar, CompositionGrammar):
            raise TypeError("grammar must be a CompositionGrammar")
        self.grammar = grammar
        self.vocab_size = max(119, grammar.domain.max_atoms + 1)
        self.max_sequence_length = 2 * grammar.domain.max_support_size + 2

    @property
    def fingerprint(self) -> str:
        payload = {
            "elements": self.grammar.domain.elements,
            "max_elements": self.grammar.domain.max_elements,
            "max_atoms": self.grammar.domain.max_atoms,
            "vocab_size": self.vocab_size,
            "type_ids": {key.value: value for key, value in _TYPE_IDS.items()},
        }
        return sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def encode_token(self, token: CompositionToken) -> tuple[int, int]:
        if not isinstance(token, CompositionToken):
            raise TypeError("token must be a CompositionToken")
        token_id = self.eos_id if token.kind is TokenKind.EOS else int(token.value)
        if not 0 <= token_id < self.vocab_size:
            raise DomainError("token value lies outside the codec vocabulary")
        return token_id, _TYPE_IDS[token.kind]

    def decode_token(self, token_id: int, token_type: int) -> CompositionToken:
        if isinstance(token_id, bool) or isinstance(token_type, bool):
            raise DomainError("token IDs and types must be integers")
        if not 0 <= int(token_id) < self.vocab_size:
            raise DomainError("token ID lies outside the codec vocabulary")
        inverse = {value: key for key, value in _TYPE_IDS.items()}
        try:
            kind = inverse[int(token_type)]
        except KeyError as exc:
            raise DomainError("unknown composition token type") from exc
        if kind is TokenKind.EOS:
            if int(token_id) != self.eos_id:
                raise DomainError("EOS type requires value ID zero")
            return EOS_TOKEN
        if int(token_id) == 0:
            raise DomainError("generated integer tokens must be positive")
        return CompositionToken(kind, int(token_id))

    def allowed_arrays(
        self, prefix: Sequence[CompositionToken]
    ) -> tuple[IntArray, IntArray]:
        allowed = self.grammar.allowed_next(prefix)
        pairs = [self.encode_token(token) for token in allowed]
        return (
            np.asarray([pair[0] for pair in pairs], dtype=np.int64),
            np.asarray([pair[1] for pair in pairs], dtype=np.int64),
        )

    def encode(self, composition: Composition) -> EncodedComposition:
        serialization = self.grammar.serialize(composition)
        token_pairs = [self.encode_token(token) for token in serialization]
        legal = np.zeros((len(serialization), self.vocab_size), dtype=bool)
        prefix: tuple[CompositionToken, ...] = ()
        for index, target in enumerate(serialization):
            allowed = self.grammar.allowed_next(prefix)
            if target not in allowed:
                raise DomainError("serialized target is absent from its live-prefix support")
            for token in allowed:
                token_id, _ = self.encode_token(token)
                legal[index, token_id] = True
            prefix += (target,)
        return EncodedComposition(
            token_ids=np.asarray([pair[0] for pair in token_pairs], dtype=np.int64),
            token_types=np.asarray([pair[1] for pair in token_pairs], dtype=np.int64),
            legal_mask=legal,
        )

    def decode(self, token_ids: Sequence[int], token_types: Sequence[int]) -> Composition:
        if len(token_ids) != len(token_types):
            raise DomainError("token IDs and types must have equal length")
        tokens = tuple(
            self.decode_token(int(token_id), int(token_type))
            for token_id, token_type in zip(token_ids, token_types, strict=True)
        )
        return self.grammar.deserialize(tokens)

    def collate(self, compositions: Sequence[Composition]) -> CompositionBatch:
        if not compositions:
            raise DomainError("cannot collate an empty composition batch")
        encoded = [self.encode(composition) for composition in compositions]
        length = max(item.token_ids.shape[0] for item in encoded)
        batch = len(encoded)
        token_ids = np.full((batch, length), self.eos_id, dtype=np.int64)
        token_types = np.full((batch, length), self.eos_type, dtype=np.int64)
        legal_mask = np.zeros((batch, length, self.vocab_size), dtype=bool)
        sequence_mask = np.zeros((batch, length), dtype=bool)
        for index, item in enumerate(encoded):
            item_length = item.token_ids.shape[0]
            token_ids[index, :item_length] = item.token_ids
            token_types[index, :item_length] = item.token_types
            legal_mask[index, :item_length] = item.legal_mask
            legal_mask[index, item_length:, self.eos_id] = True
            sequence_mask[index, :item_length] = True
        return CompositionBatch(token_ids, token_types, legal_mask, sequence_mask)


__all__ = [
    "CompositionBatch",
    "CompositionTokenCodec",
    "EncodedComposition",
]
