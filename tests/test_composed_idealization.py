"""Composition fidelity is not implied by per-frame componentwise tolerances."""

from dataclasses import replace

import numpy as np
import pytest

from crystalatlas.errors import AmbiguousPreprocessingError, InvariantViolation
from crystalatlas.standardization import FloatingStructure, SpglibPolicy, SpglibStandardizer

pytestmark = pytest.mark.filterwarnings("ignore:Set OLD_ERROR_HANDLING.*:DeprecationWarning")


def _hexagonal(angle_error: float) -> FloatingStructure:
    gamma = np.deg2rad(120.0 - angle_error)
    a = 4.8
    lattice = np.asarray(
        [[a, 0.0, 0.0], [a * np.cos(gamma), a * np.sin(gamma), 0.0], [0.0, 0.0, 7.7]]
    )
    positions = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.5],
            [1 / 3, 2 / 3, 0.75],
            [2 / 3, 1 / 3, 0.25],
            [1 / 3, 2 / 3, 0.25],
            [2 / 3, 1 / 3, 0.75],
        ]
    )
    return FloatingStructure(lattice, positions, (60, 60, 82, 82, 79, 79))


def _standardizer() -> SpglibStandardizer:
    spglib = pytest.importorskip("spglib")
    return SpglibStandardizer(
        {194: 488}, SpglibPolicy(required_spglib_version=spglib.__version__)
    )


@pytest.mark.parametrize("angle_error,accepted", [(0.0, True), (1e-7, True), (6e-7, False)])
def test_final_fidelity_is_checked_after_composing_proper_frames(
    angle_error: float, accepted: bool
) -> None:
    standardizer = _standardizer()
    raw = _hexagonal(angle_error)
    initial = standardizer._canonicalize_pinned(raw)
    tolerance = standardizer.policy.roundtrip_tolerance
    assert np.allclose(
        initial.raw_frame_primitive_lattice @ initial.raw_frame_to_standard_rotation,
        initial.primitive_lattice,
        atol=tolerance,
        rtol=tolerance,
    )
    if accepted:
        result = standardizer.canonicalize(raw)
        assert result.international_number == 194
        assert np.allclose(
            np.asarray(result.exact_input_to_raw_frame_primitive_basis, dtype=float)
            @ raw.lattice
            @ result.raw_frame_to_standard_rotation,
            result.primitive_lattice,
            atol=tolerance,
            rtol=tolerance,
        )
    else:
        with pytest.raises(AmbiguousPreprocessingError, match="composed idealization"):
            standardizer.canonicalize(raw)


@pytest.mark.parametrize("bad", ["lattice", "rotation", "basis"])
def test_internal_map_corruption_is_fatal_before_input_fidelity(
    monkeypatch: pytest.MonkeyPatch, bad: str
) -> None:
    from crystalatlas import primitive_canonicalization

    original = primitive_canonicalization.canonicalize_primitive_presentation

    def altered(*args, **kwargs):
        result = original(*args, **kwargs)
        if bad == "lattice":
            return replace(result, lattice=result.lattice * 1.001)
        if bad == "rotation":
            return replace(result, cartesian_rotation=-result.cartesian_rotation)
        return replace(result, basis_change=((1, 1, 0), (0, 1, 0), (0, 0, 1)))

    monkeypatch.setattr(
        primitive_canonicalization, "canonicalize_primitive_presentation", altered
    )
    with pytest.raises(InvariantViolation, match="internal primitive presentation map"):
        _standardizer().canonicalize(_hexagonal(6e-7))


def test_staged_raw_frame_check_detects_corrupted_composed_witness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    standardizer = _standardizer()
    original = standardizer._canonicalize_pinned
    calls = 0

    def altered(raw):
        nonlocal calls
        result = original(raw)
        calls += 1
        if calls == 1:
            # Intentionally bypass the immutable certificate constructor to
            # simulate a broken internal primitive-basis witness.
            object.__setattr__(
                result,
                "exact_input_to_raw_frame_primitive_basis",
                ((1, 1, 0), (0, 1, 0), (0, 0, 1)),
            )
        return result

    monkeypatch.setattr(standardizer, "_canonicalize_pinned", altered)
    with pytest.raises(InvariantViolation, match="composed primitive witness"):
        standardizer.canonicalize(_hexagonal(6e-7))
