"""Dimensioned raw-to-ideal idealization gate (review finding B1).

The legacy v6 policy compares raw and spglib-idealized lattices with a
dimensionless componentwise rtol=atol=2e-8 while classifying symmetry at
symprec=1e-5 A.  A cell distorted by 1e-7 relative (about 3e-7 A, thirty times
below symprec) is therefore classified as hexagonal and then rejected.  The
dimensioned policy bounds the Cartesian displacement of every point of the
primitive cell under lattice idealization by a length in angstrom instead.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

pytest.importorskip("spglib")

from crystalatlas.config import DomainConfig  # noqa: E402
from crystalatlas.dataset_identity import IdentityPolicy, build_canonical_identity  # noqa: E402
from crystalatlas.dataset_storage import (  # noqa: E402
    _canonicalization_payload,
    _parse_canonicalization,
)
from crystalatlas.errors import AmbiguousPreprocessingError, DomainError  # noqa: E402
from crystalatlas.full_catalogue import (  # noqa: E402
    build_full_catalogue,
    pinned_hall_numbers,
)
from crystalatlas.ingestion import (  # noqa: E402
    CanonicalizationReceipt,
    CatalogueTargetAssigner,
    ComposedCanonicalizer,
    RawPeriodicStructure,
    structure_to_catalogue_training_target,
)
from crystalatlas.standardization import (  # noqa: E402
    FloatingStructure,
    SpglibPolicy,
    SpglibStandardizer,
    StandardizationProvenance,
    lattice_idealization_displacement,
)

_A, _C = 3.2, 5.2
_HCP = np.array([[1 / 3, 2 / 3, 0.25], [2 / 3, 1 / 3, 0.75]])


def _hcp(relative_distortion: float) -> FloatingStructure:
    scale = 1.0 + relative_distortion
    lattice = np.array(
        [[_A, 0.0, 0.0], [-_A / 2 * scale, _A * np.sqrt(3) / 2 * scale, 0.0], [0.0, 0.0, _C]]
    )
    return FloatingStructure(lattice, _HCP, (12, 12))


def _standardizer(policy: SpglibPolicy) -> SpglibStandardizer:
    return SpglibStandardizer(pinned_hall_numbers(), policy)


def test_default_policy_payload_equals_the_recorded_v6_policy() -> None:
    # artifacts/mp20_pilot_2026-09-25/preprocessing512_v6.json "standardizer_policy"
    assert SpglibPolicy().fingerprint_payload() == {
        "angle_tolerance": -1.0,
        "canonical_decimals": 12,
        "required_spglib_version": "2.7.0",
        "roundtrip_tolerance": 2e-08,
        "stability_symprec_factors": (0.5, 2.0),
        "symprec": 1e-05,
    }


def test_legacy_provenance_fingerprint_is_unchanged() -> None:
    provenance = StandardizationProvenance(
        backend="spglib",
        backend_version="2.7.0",
        hall_map_fingerprint="a" * 64,
        hall_number=488,
        symprec=1e-5,
        angle_tolerance=-1.0,
        canonical_decimals=12,
        input_atom_count=4,
        primitive_atom_count=2,
        conventional_atom_count=2,
        stability_symprec_factors=(0.5, 2.0),
        symmetry_stability_verified=True,
    )
    assert provenance.idealization_tolerance_angstrom is None
    # Value computed with the pre-change implementation.
    assert provenance.fingerprint == (
        "b0d6ef34d9dc9dfb2d9111f6da66babf0ef3d2fc53f7880cf7147eff85d0158a"
    )


def test_idealization_tolerance_is_validated_and_fingerprinted() -> None:
    for invalid in (0.0, -1e-3, float("nan"), float("inf")):
        with pytest.raises(DomainError):
            SpglibPolicy(idealization_tolerance_angstrom=invalid)
    dimensioned = SpglibPolicy(symprec=1e-3, idealization_tolerance_angstrom=1e-3)
    assert dimensioned.fingerprint_payload()["idealization_tolerance_angstrom"] == 1e-3
    common = {
        "backend": "spglib",
        "backend_version": "2.7.0",
        "hall_map_fingerprint": "a" * 64,
        "hall_number": 488,
        "symprec": 1e-3,
        "angle_tolerance": -1.0,
        "canonical_decimals": 12,
        "input_atom_count": 4,
        "primitive_atom_count": 2,
        "conventional_atom_count": 2,
        "stability_symprec_factors": (0.5, 2.0),
        "symmetry_stability_verified": True,
    }
    legacy = StandardizationProvenance(**common)  # type: ignore[arg-type]
    gated = StandardizationProvenance(
        **common, idealization_tolerance_angstrom=1e-3  # type: ignore[arg-type]
    )
    assert legacy.fingerprint != gated.fingerprint
    with pytest.raises(DomainError):
        StandardizationProvenance(
            **common, idealization_tolerance_angstrom=0.0  # type: ignore[arg-type]
        )


def test_vertex_displacement_is_the_exact_maximum_over_the_cell() -> None:
    rng = np.random.default_rng(3)
    for _ in range(20):
        raw = np.eye(3) * 4.0 + rng.standard_normal((3, 3))
        delta = 1e-3 * rng.standard_normal((3, 3))
        value = lattice_idealization_displacement(raw, raw + delta)
        # The norm of an affine image is convex, so the cube maximum is a vertex.
        grid = np.array(list(itertools.product(np.linspace(0.0, 1.0, 5), repeat=3)))
        dense = float(np.max(np.linalg.norm(grid @ delta, axis=1)))
        assert value >= dense - 1e-15
        vertices = np.array(list(itertools.product((0.0, 1.0), repeat=3)))
        assert value == pytest.approx(float(np.max(np.linalg.norm(vertices @ delta, axis=1))))
    assert lattice_idealization_displacement(np.eye(3), np.diag([1.0 + 2e-3, 1.0, 1.0])) == (
        pytest.approx(2e-3)
    )


def test_exact_input_needs_no_idealization_under_either_policy() -> None:
    for policy in (SpglibPolicy(), SpglibPolicy(idealization_tolerance_angstrom=1e-5)):
        result = _standardizer(policy).canonicalize(_hcp(0.0))
        assert result.international_number == 194
        assert result.idealization_lattice_displacement_angstrom is not None
        assert result.idealization_lattice_displacement_angstrom < 1e-12
        assert result.idealization_site_displacement_angstrom is not None
        assert result.idealization_site_displacement_angstrom < 1e-12


def test_sub_symprec_lattice_idealization_is_accepted_and_recorded() -> None:
    structure = _hcp(1.0e-7)
    with pytest.raises(AmbiguousPreprocessingError, match="input-fidelity"):
        _standardizer(SpglibPolicy()).canonicalize(structure)

    result = _standardizer(SpglibPolicy(idealization_tolerance_angstrom=1e-5)).canonicalize(
        structure
    )
    assert result.international_number == 194
    assert result.provenance.idealization_tolerance_angstrom == 1e-5
    displacement = result.idealization_lattice_displacement_angstrom
    assert displacement is not None
    # |b| changes by 3.2e-7 A; spglib's idealization moves the cell by that order.
    assert 1e-8 < displacement < 1e-6
    assert result.idealization_site_displacement_angstrom is not None
    assert 0.0 <= result.idealization_site_displacement_angstrom <= 1e-5


def test_idealization_beyond_the_declared_length_is_rejected() -> None:
    policy = SpglibPolicy(idealization_tolerance_angstrom=1e-9)
    with pytest.raises(AmbiguousPreprocessingError, match="idealization"):
        _standardizer(policy).canonicalize(_hcp(1.0e-7))


def test_receipt_idealization_metadata_round_trips_and_legacy_records_still_parse() -> None:
    common = {
        "source_fingerprint": "1" * 64,
        "standardization_fingerprint": "2" * 64,
        "catalogue_fingerprint": "3" * 64,
        "hall_number": 488,
        "hall_symbol": "-P 6c 2c",
        "source_to_primitive": ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        "origin_shift": (0, 0, 0),
        "free_origin_shift": (0, 0, 0),
        "translation_index": 1,
        "symmetry_tolerance": 1e-5,
        "roundtrip_max_cartesian_error": 0.0,
        "symmetry_stability_verified": True,
        "catalogue_closure_verified": True,
        "wyckoff_multiplicity_verified": True,
        "symmetry_verified": True,
        "canonical_origin_verified": True,
        "scaffold_roundtrip_verified": True,
    }
    from fractions import Fraction

    common["source_to_primitive"] = tuple(
        tuple(Fraction(v) for v in row) for row in common["source_to_primitive"]  # type: ignore[attr-defined]
    )
    common["origin_shift"] = (Fraction(0),) * 3
    common["free_origin_shift"] = (Fraction(0),) * 3
    legacy = CanonicalizationReceipt(**common)  # type: ignore[arg-type]
    legacy_payload = _canonicalization_payload(legacy)
    assert "idealization_lattice_displacement_angstrom" not in legacy_payload
    assert _parse_canonicalization(legacy_payload) == legacy

    recorded = CanonicalizationReceipt(
        **common,  # type: ignore[arg-type]
        idealization_lattice_displacement_angstrom=3.1e-7,
        idealization_site_displacement_angstrom=2.0e-9,
    )
    assert _parse_canonicalization(_canonicalization_payload(recorded)) == recorded
    with pytest.raises(DomainError):
        CanonicalizationReceipt(
            **common,  # type: ignore[arg-type]
            idealization_lattice_displacement_angstrom=3.1e-7,
        )
    with pytest.raises(DomainError):
        CanonicalizationReceipt(
            **common,  # type: ignore[arg-type]
            idealization_lattice_displacement_angstrom=-1.0,
            idealization_site_displacement_angstrom=0.0,
        )


def _hcp_raw(relative_distortion: float) -> RawPeriodicStructure:
    structure = _hcp(relative_distortion)
    return RawPeriodicStructure(
        structure.lattice, structure.fractional_positions, (12, 12), np.ones(2)
    )


@pytest.mark.parametrize(
    ("distortion", "tau", "legacy_group"),
    [
        # 3e-7 A below symprec: legacy classifies P6_3/mmc, then rejects fidelity.
        (1.0e-7, 1.0e-5, None),
        # 1.6e-4 A above legacy symprec: legacy lowers the label to P2_1/m (B1).
        (5.0e-5, 1.0e-3, "P2_1/m"),
    ],
)
def test_dimensioned_policy_builds_a_complete_identified_training_target(
    distortion: float, tau: float, legacy_group: str | None
) -> None:
    """Every downstream raw-to-ideal check must honour the declared length.

    The primitive-quotient and volume/index relations compare the raw cell with
    the idealized primitive; under the dimensioned policy they are bounded by
    the recorded vertex displacement d through |V'/V - 1| <= (1 + sqrt(3) d /
    sigma_min)^3 - 1, not by a fixed rtol of 1e-8.
    """

    catalogue = build_full_catalogue()  # the assigner requires all 230 types
    raw = _hcp_raw(distortion)
    legacy = SpglibStandardizer(pinned_hall_numbers(), SpglibPolicy())
    if legacy_group is None:
        with pytest.raises(AmbiguousPreprocessingError, match="input-fidelity"):
            structure_to_catalogue_training_target(
                raw,
                standardizer=legacy,
                target_assigner=CatalogueTargetAssigner(catalogue),
                domain=DomainConfig(),
            )
    else:
        lowered = structure_to_catalogue_training_target(
            raw,
            standardizer=legacy,
            target_assigner=CatalogueTargetAssigner(catalogue),
            domain=DomainConfig(),
        )
        assert lowered.canonicalized.scaffold.space_group == legacy_group
    gated = SpglibStandardizer(
        pinned_hall_numbers(), SpglibPolicy(symprec=tau, idealization_tolerance_angstrom=tau)
    )
    record = structure_to_catalogue_training_target(
        raw,
        standardizer=gated,
        target_assigner=CatalogueTargetAssigner(catalogue),
        domain=DomainConfig(),
    )
    assert record.canonicalized.scaffold.space_group == "P6_3/mmc"
    receipt = record.canonicalized.receipt
    assert receipt.idealization_lattice_displacement_angstrom is not None
    assert 0.0 < receipt.idealization_lattice_displacement_angstrom <= tau
    build_canonical_identity(record, catalogue=catalogue, policy=IdentityPolicy())


# MP-20 pilot record 31 (mp-2832, R-3), raw primitive input as parsed.  Its
# conventional coordinates carry thirds such as 0.026255 / 3, so 12-decimal
# quantization of different orbit points gives exact 18f orbits that differ by
# at most 1.4e-11 A, far below the assigner's 2e-8 A round-trip tolerance.
_MP_2832_LATTICE = np.array(
    [
        [6.94430285, 0.0, 0.0],
        [4.032597578827893, 5.65345017133339, 0.0],
        [4.032597058036954, 2.076914760819408, 5.258129003407344],
    ]
)
_MP_2832_SITES = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.5],
        [0.676095, 0.676095, 0.676095],
        [0.323905, 0.323905, 0.323905],
        [0.737924, 0.095833, 0.403751],
        [0.403751, 0.737924, 0.095833],
        [0.095833, 0.403751, 0.737924],
        [0.262076, 0.904167, 0.596249],
        [0.596249, 0.262076, 0.904167],
        [0.904167, 0.596249, 0.262076],
    ]
)


def test_quantization_equivalent_exact_orbits_are_one_orbit() -> None:
    catalogue = build_full_catalogue()
    raw = RawPeriodicStructure(
        _MP_2832_LATTICE, _MP_2832_SITES, (24,) * 4 + (34,) * 6, np.ones(10)
    )
    standardizer = SpglibStandardizer(
        pinned_hall_numbers(),
        SpglibPolicy(
            symprec=1e-3, idealization_tolerance_angstrom=1e-3, relative_metric_quantum=1e-9
        ),
    )
    strict = ComposedCanonicalizer(standardizer, CatalogueTargetAssigner(catalogue))
    with pytest.raises(AmbiguousPreprocessingError, match="ambiguous in Wyckoff chart 18f"):
        strict.canonicalize(raw)
    merged = ComposedCanonicalizer(
        standardizer,
        CatalogueTargetAssigner(catalogue, merge_quantization_equivalent_orbits=True),
    )
    canonicalized = merged.canonicalize(raw)
    assert canonicalized.scaffold.space_group == "R-3"
    assert sorted(orbit.wyckoff_label for orbit in canonicalized.scaffold.orbits) == [
        "18f",
        "3a",
        "3b",
        "6c",
    ]
