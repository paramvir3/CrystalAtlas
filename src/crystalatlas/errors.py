"""Domain-specific errors.

The package fails closed: an unavailable catalogue entry, unproved support
condition, or invalid numerical state is an error rather than an invitation to
silently repair a generated crystal.
"""

from __future__ import annotations


class CrystalAtlasError(Exception):
    """Base class for package errors."""


class RecoverableGenerationError(CrystalAtlasError):
    """A declared attempt can fail without invalidating the implementation.

    Generation boundaries may turn only this exception family into a failed
    attempt.  Catalogue corruption, invariant failures, programming errors,
    accelerator failures, and resource exhaustion must remain visible to the
    caller rather than being misreported as model invalidity.
    """


class DomainError(RecoverableGenerationError, ValueError):
    """An input lies outside the declared finite model domain."""


class CatalogueError(CrystalAtlasError):
    """The exact crystallographic catalogue is incomplete or inconsistent."""


class InvariantViolation(CrystalAtlasError):  # noqa: N818 - public scientific term
    """An equation-level invariant failed during execution."""


class OptionalDependencyError(CrystalAtlasError, ImportError):
    """A requested optional feature is unavailable."""


class NumericalError(RecoverableGenerationError, ArithmeticError):
    """A numerical calculation failed its declared tolerance contract."""


class ResourceLimitError(CrystalAtlasError):
    """An explicit work budget was exceeded, not a physically invalid sample."""


class GeneratedSampleError(RecoverableGenerationError):
    """One generated state lies on a declared invalid/defect set."""


class RecordPreprocessingError(CrystalAtlasError):
    """One source row failed the frozen crystallographic preprocessing contract.

    Only this exception family may be converted into a dataset-row rejection.
    Catalogue errors, invariant violations, programming errors, and model/output
    quality decisions must abort compilation instead of disappearing from data.
    """


class AmbiguousPreprocessingError(RecordPreprocessingError):
    """A source structure has no unique certified crystallographic interpretation."""
