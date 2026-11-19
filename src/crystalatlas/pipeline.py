"""Task-independent ancestral sampling orchestration.

The engine implements the factorization only.  It does not retry, repair,
filter, relax, rank, or infer physical stability.  Those actions define
separate, auditable proposal or discovery protocols.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Generic, Protocol, TypeVar, runtime_checkable

import numpy as np

from .errors import DomainError, RecoverableGenerationError

CompositionT = TypeVar("CompositionT")
CoverT = TypeVar("CoverT")
StateT = TypeVar("StateT")
CrystalT = TypeVar("CrystalT")


class GenerationMode(str, Enum):  # noqa: UP042 - ML smoke runtime is Python 3.10
    DNG = "dng"
    SPECIES_DNG = "species-dng"
    CSP = "csp"


@dataclass(frozen=True, slots=True)
class GenerationRequest(Generic[CompositionT]):
    """Exactly one of the three first-class input contracts."""

    mode: GenerationMode
    species: tuple[int, ...] | None = None
    composition: CompositionT | None = None
    samples: int = 1
    seed: int = 0
    primitive_atom_count: int | None = None
    """CSP only: condition on the primitive-cell atom count, the standard MP-20
    protocol (review M6).  This is a separately labelled track; ``None`` is the
    reduced-formula track."""

    def __post_init__(self) -> None:
        if self.samples < 1:
            raise DomainError("samples must be positive")
        if self.primitive_atom_count is not None:
            count = self.primitive_atom_count
            if isinstance(count, bool) or not isinstance(count, int) or count < 1:
                raise DomainError("primitive_atom_count must be a positive integer")
            if self.mode is not GenerationMode.CSP:
                raise DomainError("primitive-count conditioning is defined for CSP only")
        if self.seed < 0:
            raise DomainError("seed must be nonnegative")
        if self.mode is GenerationMode.DNG:
            if self.species is not None or self.composition is not None:
                raise DomainError("unconditional DNG receives neither species nor formula")
        elif self.mode is GenerationMode.SPECIES_DNG:
            if not self.species or self.composition is not None:
                raise DomainError("S-DNG receives one nonempty exact species set")
        elif self.mode is GenerationMode.CSP:
            if self.composition is None or self.species is not None:
                raise DomainError("CSP receives one reduced composition and no species set")


@runtime_checkable
class CompositionSampler(Protocol[CompositionT]):
    def sample_unconditional(self, rng: np.random.Generator) -> CompositionT: ...

    def sample_given_species(
        self, species: tuple[int, ...], rng: np.random.Generator
    ) -> CompositionT: ...


@runtime_checkable
class CoverSampler(Protocol[CompositionT, CoverT]):
    def sample_cover(self, composition: CompositionT, rng: np.random.Generator) -> CoverT: ...


@runtime_checkable
class GeometrySampler(Protocol[CompositionT, CoverT, StateT]):
    def sample_geometry(
        self,
        composition: CompositionT,
        cover: CoverT,
        rng: np.random.Generator,
    ) -> StateT: ...


@runtime_checkable
class Decoder(Protocol[CoverT, StateT, CrystalT]):
    def decode(self, cover: CoverT, state: StateT) -> CrystalT: ...


@dataclass(frozen=True, slots=True)
class VerificationResult:
    valid: bool
    reasons: tuple[str, ...] = ()
    metadata: tuple[tuple[str, str], ...] = ()


@runtime_checkable
class Verifier(Protocol[CompositionT, CrystalT]):
    def verify(self, composition: CompositionT, crystal: CrystalT) -> VerificationResult: ...


@dataclass(frozen=True, slots=True)
class GenerationTrace(Generic[CompositionT, CoverT, StateT, CrystalT]):
    """One raw attempt; failures remain first-class outputs."""

    sample_index: int
    seed: int
    composition: CompositionT | None = None
    cover: CoverT | None = None
    state: StateT | None = None
    crystal: CrystalT | None = None
    verification: VerificationResult | None = None
    failed_stage: str | None = None
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return (
            self.failed_stage is None
            and self.verification is not None
            and self.verification.valid
        )


class GenerationEngine(Generic[CompositionT, CoverT, StateT, CrystalT]):
    """Shared DNG/S-DNG/CSP ancestral sampler from equations (3.1–3.1b)."""

    def __init__(
        self,
        *,
        composition_sampler: CompositionSampler[CompositionT],
        cover_sampler: CoverSampler[CompositionT, CoverT],
        geometry_sampler: GeometrySampler[CompositionT, CoverT, StateT],
        decoder: Decoder[CoverT, StateT, CrystalT],
        verifier: Verifier[CompositionT, CrystalT],
    ) -> None:
        self._composition_sampler = composition_sampler
        self._cover_sampler = cover_sampler
        self._geometry_sampler = geometry_sampler
        self._decoder = decoder
        self._verifier = verifier

    def sample(
        self, request: GenerationRequest[CompositionT]
    ) -> tuple[GenerationTrace[CompositionT, CoverT, StateT, CrystalT], ...]:
        if request.primitive_atom_count is not None:
            raise DomainError("this generation pipeline has no primitive-count conditioning")
        seed_sequence = np.random.SeedSequence(request.seed)
        child_sequences = seed_sequence.spawn(request.samples)
        traces = [
            self._sample_one(request, index, child_sequence)
            for index, child_sequence in enumerate(child_sequences)
        ]
        return tuple(traces)

    def _sample_one(
        self,
        request: GenerationRequest[CompositionT],
        sample_index: int,
        seed_sequence: np.random.SeedSequence,
    ) -> GenerationTrace[CompositionT, CoverT, StateT, CrystalT]:
        child_seed = int(seed_sequence.generate_state(1, dtype=np.uint64)[0])
        # Separate stage streams keep results invariant to internal draw counts.
        composition_seed, cover_seed, geometry_seed = seed_sequence.spawn(3)
        composition: CompositionT | None = None
        cover: CoverT | None = None
        state: StateT | None = None
        crystal: CrystalT | None = None
        try:
            if request.mode is GenerationMode.DNG:
                composition = self._composition_sampler.sample_unconditional(
                    np.random.default_rng(composition_seed)
                )
            elif request.mode is GenerationMode.SPECIES_DNG:
                assert request.species is not None
                composition = self._composition_sampler.sample_given_species(
                    request.species, np.random.default_rng(composition_seed)
                )
            else:
                assert request.composition is not None
                composition = request.composition
        except RecoverableGenerationError as exc:
            return GenerationTrace(
                sample_index=sample_index,
                seed=child_seed,
                failed_stage="composition",
                error=f"{type(exc).__name__}: {exc}",
            )
        try:
            cover = self._cover_sampler.sample_cover(
                composition, np.random.default_rng(cover_seed)
            )
        except RecoverableGenerationError as exc:
            return GenerationTrace(
                sample_index=sample_index,
                seed=child_seed,
                composition=composition,
                failed_stage="cover",
                error=f"{type(exc).__name__}: {exc}",
            )
        try:
            state = self._geometry_sampler.sample_geometry(
                composition, cover, np.random.default_rng(geometry_seed)
            )
        except RecoverableGenerationError as exc:
            return GenerationTrace(
                sample_index=sample_index,
                seed=child_seed,
                composition=composition,
                cover=cover,
                failed_stage="geometry",
                error=f"{type(exc).__name__}: {exc}",
            )
        try:
            crystal = self._decoder.decode(cover, state)
        except RecoverableGenerationError as exc:
            return GenerationTrace(
                sample_index=sample_index,
                seed=child_seed,
                composition=composition,
                cover=cover,
                state=state,
                failed_stage="decode",
                error=f"{type(exc).__name__}: {exc}",
            )
        try:
            verification = self._verifier.verify(composition, crystal)
        except RecoverableGenerationError as exc:
            return GenerationTrace(
                sample_index=sample_index,
                seed=child_seed,
                composition=composition,
                cover=cover,
                state=state,
                crystal=crystal,
                failed_stage="verify",
                error=f"{type(exc).__name__}: {exc}",
            )
        return GenerationTrace(
            sample_index=sample_index,
            seed=child_seed,
            composition=composition,
            cover=cover,
            state=state,
            crystal=crystal,
            verification=verification,
        )
