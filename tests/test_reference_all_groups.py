"""The untrained reference law beyond P1/P-1 (review m3).

Validity.  Every sampled scaffold must be collision-free with exactly the
declared multiplicities, without retry.  A pool keeps only rational parameters
whose orbit has the row's declared multiplicity, so each point of a pooled orbit
has exactly the row's site-symmetry group S_A: its stabilizer contains S_A,
which fixes the row's subspace, and has the same order.  Orbits of one row are
equal or disjoint, and the sampler draws distinct orbits without replacement.
Orbits of two rows are disjoint, since their stabilizers are non-conjugate.
The sampler checks this exactly for every pooled orbit of every requested group.

Sufficiency.  Row A needs up to capacity // m_A distinct orbits.  Its grid
denominator is the smallest odd q >= the declared base that provides them, so
every feasible cover leaf can be realized and no infeasible leaf is sampled.
"""

from __future__ import annotations

from fractions import Fraction
from functools import cache

import pytest

from crystalatlas.composition import Composition
from crystalatlas.config import CrystalAtlasConfig, DomainConfig
from crystalatlas.errors import InvariantViolation
from crystalatlas.full_catalogue import build_space_group_setting
from crystalatlas.pipeline import GenerationMode, GenerationRequest
from crystalatlas.reference import (
    ReferenceGenerator,
    ReferenceGeometrySampler,
    ReferenceLawConfig,
    certify_disjoint_orbit_pools,
)

GROUPS = ("P2_1/c", "Pmmm", "Cmcm", "P4_2/mnm", "R-3m", "P6_3/mmc", "Fm-3m", "Ia-3d")


_NUMBERS = (14, 47, 63, 136, 166, 194, 225, 230)


@cache
def _catalogue():  # type: ignore[no-untyped-def]
    """The eight test groups only; all 230 are certified by
    `artifacts/peer_review_2026-10-19/verify_reference_law_all_groups.py`."""

    from crystalatlas.catalogue import (
        FIXTURE_EXPECTED_ROWS,
        MINIMAL_CATALOGUE,
        CertifiedCatalogue,
    )

    settings = {key: MINIMAL_CATALOGUE.get(key) for key in ("P1", "P-1")}
    rows = dict(FIXTURE_EXPECTED_ROWS)
    for number in _NUMBERS:
        setting = build_space_group_setting(number)
        settings[setting.key] = setting
        rows[setting.key] = frozenset(setting.wyckoff_positions)
    assert set(GROUPS) <= set(settings)
    return CertifiedCatalogue(settings, rows, certification_scope="reference-law test subset")


def _config() -> CrystalAtlasConfig:
    return CrystalAtlasConfig(
        domain=DomainConfig(
            element_vocabulary=(6, 8, 11, 17),
            dng_element_vocabulary=(6, 8, 11, 17),
            max_distinct_elements=2,
        )
    )


@cache
def _sampler() -> ReferenceGeometrySampler:
    return ReferenceGeometrySampler(
        _config(), ReferenceLawConfig(space_groups=GROUPS), _catalogue()
    )


def test_reference_law_accepts_any_catalogue_group() -> None:
    law = ReferenceLawConfig(space_groups=GROUPS)
    assert law.space_groups == GROUPS
    assert ReferenceLawConfig().space_groups == ("P1", "P-1")  # legacy default


def test_pools_are_sufficient_and_denominators_minimal() -> None:
    sampler = _sampler()
    capacity = _config().domain.atom_bound
    for group in GROUPS:
        setting = _catalogue().get(group)
        for row in setting.wyckoff_positions.values():
            if row.dimension == 0:
                continue
            needed = capacity.capacity(setting.centering_index) // row.multiplicity
            if needed == 0:
                # No feasible cover can occupy a row larger than the capacity.
                assert (group, row.label) not in sampler._parameter_pools
                continue
            denominator = sampler.parameter_grid_denominators[(group, row.label)]
            assert denominator % 2 == 1 and denominator >= 5
            assert len(sampler._parameter_pools[(group, row.label)]) >= needed
            if denominator > 5:
                smaller = sampler._build_parameter_pool(setting, row.label, denominator - 2)
                assert len(smaller) < needed
    # Pmmm's 2-fold special lines need 20 orbits of 40 atoms: the worst row.
    assert sampler.parameter_grid_denominators[("Pmmm", "2t")] == 41


def test_pooled_orbits_have_exact_multiplicity_and_are_pairwise_disjoint() -> None:
    sampler = _sampler()
    for group in GROUPS:
        setting = _catalogue().get(group)
        owner: dict[tuple[Fraction, ...], tuple[str, int]] = {}
        for row in setting.wyckoff_positions.values():
            if row.dimension > 0 and (group, row.label) not in sampler._parameter_pools:
                continue  # unoccupiable at this capacity
            pool = sampler._parameter_pools.get((group, row.label), ((),))
            for index, parameter in enumerate(pool):
                points = row.expand_exact(parameter, require_declared_multiplicity=True)
                assert len(points) == row.multiplicity
                for point in points:
                    assert point not in owner, (group, row.label, owner.get(point))
                    owner[point] = (row.label, index)


def test_disjointness_certificate_refuses_a_shared_point() -> None:
    sampler = _sampler()
    setting = _catalogue().get("Pmmm")
    pools = {
        label: sampler._parameter_pools[("Pmmm", label)]
        for label in ("2t", "1a")
        if ("Pmmm", label) in sampler._parameter_pools
    }
    certify_disjoint_orbit_pools(setting, pools)
    first = pools["2t"][0]
    with pytest.raises(InvariantViolation, match="share a point"):
        certify_disjoint_orbit_pools(setting, {"2t": (first, first)})


def test_p1_and_p_minus_1_law_is_unchanged_at_its_declared_grid() -> None:
    """The tests/test_reference.py configuration: q = 3 already suffices."""

    config = CrystalAtlasConfig(
        domain=DomainConfig(
            element_vocabulary=(1, 6, 8, 14),
            dng_element_vocabulary=(1, 6, 8),
            max_distinct_elements=3,
            max_raw_atoms=8,
            max_formula_units=4,
        )
    )
    sampler = ReferenceGeometrySampler(
        config, ReferenceLawConfig(parameter_grid_denominator=3), _catalogue()
    )
    assert set(sampler.parameter_grid_denominators.values()) == {3}


@pytest.mark.parametrize(
    ("group", "composition"),
    [
        ("Fm-3m", {11: 1, 17: 1}),
        ("P6_3/mmc", {6: 1}),
        ("R-3m", {6: 1, 8: 1}),
        ("Pmmm", {6: 1, 8: 2}),
        ("Ia-3d", {6: 1}),
    ],
)
def test_samples_in_general_groups_decode_and_pass_the_exact_audit(group, composition) -> None:  # type: ignore[no-untyped-def]
    generator = ReferenceGenerator(
        _config(), ReferenceLawConfig(space_groups=(group,)), _catalogue()
    )
    request = GenerationRequest(
        mode=GenerationMode.CSP,
        composition=Composition.from_counts(composition),
        samples=6,
        seed=101,
    )
    for sample in generator.require_samples(request):
        decoded = sample.decoded
        assert decoded.setting.key == group
        assert decoded.audit.passed
        assert len(set(decoded.fractional_positions)) == len(decoded.fractional_positions)
        assert dict(decoded.conventional_counts) == {
            species: decoded.scaffold.nu * count for species, count in composition.items()
        }
