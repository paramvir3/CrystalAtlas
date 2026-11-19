"""Reported symmetry idealization and exact lattice topology are distinct."""

from dataclasses import replace

import numpy as np
import pytest

from crystalatlas.errors import AmbiguousPreprocessingError, InvariantViolation
from crystalatlas.standardization import (
    STANDARDIZATION_ALGORITHM_ID,
    FloatingStructure,
    SpglibPolicy,
    SpglibStandardizer,
)

pytestmark = pytest.mark.filterwarnings("ignore:Set OLD_ERROR_HANDLING.*:DeprecationWarning")


def _near_cubic(strain: float, rotated: bool = False) -> FloatingStructure:
    angle = 0.371 if rotated else 0.0
    rotation = np.asarray(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return FloatingStructure(
        np.diag([4.0 * (1.0 + strain), 4.0, 4.0]) @ rotation,
        np.zeros((1, 3)),
        (14,),
    )


@pytest.mark.parametrize("rotated", [False, True])
@pytest.mark.parametrize("strain,accepted", [(0.0, True), (1e-8, True), (1e-7, False)])
def test_fidelity_uses_existing_mixed_absolute_relative_predicate(
    strain: float, accepted: bool, rotated: bool
) -> None:
    spglib = pytest.importorskip("spglib")
    policy = SpglibPolicy(required_spglib_version=spglib.__version__)
    raw = _near_cubic(strain, rotated)
    cell = (raw.lattice, raw.fractional_positions, raw.species)
    dataset = spglib.get_symmetry_dataset(cell, hall_number=517, symprec=policy.symprec)
    assert dataset is not None and dataset.number == 221
    # For this primitive cubic setting, P^-T is itself the exact primitive
    # basis change. Do not infer it from the idealized metric.
    basis = np.linalg.inv(dataset.transformation_matrix).T
    np.testing.assert_allclose(basis, np.rint(basis), atol=1e-12, rtol=0.0)
    raw_primitive = np.rint(basis) @ raw.lattice
    reported_raw_frame = dataset.std_lattice @ dataset.std_rotation_matrix
    existing_check = np.allclose(
        raw_primitive,
        reported_raw_frame,
        atol=policy.roundtrip_tolerance,
        rtol=policy.roundtrip_tolerance,
    ) and np.allclose(
        raw_primitive @ dataset.std_rotation_matrix.T,
        dataset.std_lattice,
        atol=policy.roundtrip_tolerance,
        rtol=policy.roundtrip_tolerance,
    )
    assert bool(existing_check) == accepted
    standardizer = SpglibStandardizer({221: 517}, policy)
    if not accepted:
        with pytest.raises(AmbiguousPreprocessingError, match="reported idealized lattice"):
            standardizer._canonicalize_pinned(raw)
    else:
        result = standardizer._canonicalize_pinned(raw)
        assert result.provenance.algorithm == STANDARDIZATION_ALGORITHM_ID
        assert result.provenance.algorithm.endswith("-v6")
        np.testing.assert_allclose(
            result.conventional_lattice, dataset.std_lattice, atol=1e-12, rtol=0.0
        )
        assert result.raw_translation_index == 1


@pytest.mark.parametrize(
    "bad_field", ["rotation", "raw_primitive", "transformation", "input_map", "standard_map"]
)
def test_bad_backend_certificate_remains_fatal_before_fidelity_rejection(
    monkeypatch: pytest.MonkeyPatch, bad_field: str
) -> None:
    spglib = pytest.importorskip("spglib")
    original = spglib.get_symmetry_dataset

    def altered(*args, **kwargs):
        result = original(*args, **kwargs)
        assert result is not None
        if bad_field == "rotation":
            return replace(result, std_rotation_matrix=np.diag([-1.0, 1.0, 1.0]))
        if bad_field == "raw_primitive":
            return replace(result, primitive_lattice=result.primitive_lattice * 1.1)
        if bad_field == "input_map":
            return replace(result, mapping_to_primitive=np.asarray([], dtype=np.int32))
        if bad_field == "standard_map":
            return replace(result, std_mapping_to_primitive=np.asarray([7], dtype=np.int32))
        return replace(result, transformation_matrix=np.zeros((3, 3)))

    monkeypatch.setattr(spglib, "get_symmetry_dataset", altered)
    standardizer = SpglibStandardizer(
        {221: 517}, SpglibPolicy(required_spglib_version=spglib.__version__)
    )
    with pytest.raises(InvariantViolation):
        standardizer._canonicalize_pinned(_near_cubic(1e-7))
