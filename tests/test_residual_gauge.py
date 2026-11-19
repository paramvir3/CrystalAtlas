"""Certified residual gauge of a cover (option (a), the learned-F10 finding).

The intrinsic coordinate eta of a cover is not unique.  These parameter maps
leave the crystal unchanged, or map it onto a properly congruent copy:

* per-orbit in-group gauges: an operation h of G that maps a Wyckoff row's
  parametrized subspace onto itself, x(p) -> x(p A + a).  The orbit at p and
  at p A + a is the same point set.  P-1's sign flip p -> -p is one example
  (it was the only one certified before).  Pm-3m 6e (x, 0, 0) -> (-x, 0, 0) is
  another;
* global automorphisms from the proper Euclidean normalizer, mapping every
  occupied (row, species) onto an occupied one.  P2_13 4a gives
  x -> x + 1/2, 1/4 - x and 3/4 - x;
* identical-orbit relabellings (unchanged, existing).

Every map is derived exactly and certified on exact decoded point sets.
"""

from __future__ import annotations

from fractions import Fraction
from functools import cache

import numpy as np
import pytest

from crystalatlas.catalogue import FIXTURE_EXPECTED_ROWS, MINIMAL_CATALOGUE, CertifiedCatalogue
from crystalatlas.composition import Composition
from crystalatlas.full_catalogue import build_space_group_setting
from crystalatlas.geometry import ContinuousCover, IntrinsicCoverLayout, OrbitType
from crystalatlas.residual_gauge import cover_residual_gauge, row_parameter_gauges

F = Fraction
HALF = F(1, 2)


@cache
def _catalogue():  # type: ignore[no-untyped-def]
    settings = {key: MINIMAL_CATALOGUE.get(key) for key in ("P1", "P-1")}
    rows = dict(FIXTURE_EXPECTED_ROWS)
    for number in (4, 47, 198, 221):
        setting = build_space_group_setting(number)
        settings[setting.key] = setting
        rows[setting.key] = frozenset(setting.wyckoff_positions)
    return CertifiedCatalogue(settings, rows, certification_scope="residual-gauge tests")


def _layout(group: str, orbits, nu: int, counts) -> IntrinsicCoverLayout:  # type: ignore[no-untyped-def]
    cover = ContinuousCover(
        nu=nu,
        space_group=group,
        composition=Composition(tuple(sorted(counts.items()))),
        orbits=tuple(OrbitType(label, z) for label, z in orbits),
    )
    return IntrinsicCoverLayout(cover, catalogue=_catalogue())


def _affine_set(maps) -> set[tuple[tuple[Fraction, ...], tuple[Fraction, ...]]]:  # type: ignore[no-untyped-def]
    return {(tuple(np.asarray(a).ravel()), tuple(b)) for a, b in maps}


def test_p_minus_1_row_gauge_is_the_sign_flip() -> None:
    row = _catalogue().get("P-1").wyckoff("2i")
    gauges = row_parameter_gauges(_catalogue().get("P-1"), row)
    identity = ((F(1), F(0), F(0)), (F(0), F(1), F(0)), (F(0), F(0), F(1)))
    minus = tuple(tuple(-v for v in r) for r in identity)
    assert _affine_set(gauges) == _affine_set([(identity, (F(0),) * 3), (minus, (F(0),) * 3)])


def test_pm_3m_6e_row_gauge_flips_x() -> None:
    setting = _catalogue().get("Pm-3m")
    gauges = row_parameter_gauges(setting, setting.wyckoff("6e"))
    assert _affine_set(gauges) == _affine_set([(((F(1),),), (F(0),)), (((F(-1),),), (F(0),))])


def test_p2_13_4a_cover_gauge_is_the_order_4_normalizer_action() -> None:
    layout = _layout("P2_13", [("4a", 6)], 4, {6: 1})
    gauge = cover_residual_gauge(layout)
    eta = np.asarray([0.12])
    images = sorted(
        float(np.mod(eta @ matrix + shift, 1.0)[0]) for matrix, shift in gauge.all_eta_actions()
    )
    np.testing.assert_allclose(
        images, sorted(np.mod([0.12, 0.62, 0.13, 0.63], 1.0)), atol=1e-12
    )


def test_p1_with_two_orbits_keeps_only_the_relabelling() -> None:
    # P1 has no in-group or discrete normalizer gauge; the relabelling of two
    # identical orbits (with the origin re-anchored) is the whole residual group.
    layout = _layout("P1", [("1a", 6), ("1a", 6)], 2, {6: 1})
    gauge = cover_residual_gauge(layout)
    assert len(gauge.orbit_choices[0]) == 1 and len(gauge.orbit_choices[1]) == 1
    assert len(gauge.normalizer_maps) == 1
    assert len(gauge.all_eta_actions()) == 2  # identity and the exchange


def test_p_minus_1_cover_gauge_contains_signs_and_half_translations() -> None:
    layout = _layout("P-1", [("2i", 8)], 2, {8: 1})
    gauge = cover_residual_gauge(layout)
    eta = np.asarray([0.1, 0.2, 0.3])
    images = {
        tuple(np.round(np.mod(eta @ matrix + shift, 1.0), 12))
        for matrix, shift in gauge.all_eta_actions()
    }
    # +/- p and the 8 inversion-centre choices: 16 distinct images.
    assert len(images) == 16
    assert tuple(np.round(np.mod(-eta, 1.0), 12)) in images
    assert tuple(np.round(np.mod(eta + [0.5, 0.0, 0.0], 1.0), 12)) in images


def test_every_action_is_certified_on_exact_decoded_point_sets() -> None:
    for layout in (
        _layout("P2_13", [("4a", 6), ("4a", 8)], 4, {6: 1, 8: 1}),
        _layout("Pm-3m", [("6e", 6), ("1a", 8)], 1, {6: 6, 8: 1}),
        _layout("Pmmm", [("2i", 6), ("1a", 8)], 1, {6: 2, 8: 1}),
        _layout("P2_1", [("2a", 6), ("2a", 8)], 2, {6: 1, 8: 1}),
    ):
        gauge = cover_residual_gauge(layout)
        assert gauge.certified
        assert len(gauge.all_eta_actions()) >= 1


def test_uniform_product_sampling_covers_the_group() -> None:
    layout = _layout("P2_13", [("4a", 6)], 4, {6: 1})
    gauge = cover_residual_gauge(layout)
    rng = np.random.default_rng(0)
    seen = set()
    eta = np.asarray([0.12])
    for _ in range(200):
        matrix, shift = gauge.sample_eta_action(rng, permutation=(0,))
        seen.add(round(float(np.mod(eta @ matrix + shift, 1.0)[0]), 12))
    assert len(seen) == 4


@pytest.mark.parametrize("label", ["4a"])
def test_gauge_images_decode_to_congruent_crystals(label: str) -> None:
    layout = _layout("P2_13", [(label, 6)], 4, {6: 1})
    gauge = cover_residual_gauge(layout)
    for matrix, shift in gauge.all_eta_actions():
        assert (
            np.allclose(matrix, np.round(matrix))
            and abs(abs(np.linalg.det(matrix)) - 1) < 1e-12
        )
        assert np.all((shift >= -1e-12) & (shift < 1.0))


def _exact_crystal(layout: IntrinsicCoverLayout, eta) -> frozenset:  # type: ignore[no-untyped-def]
    from crystalatlas.residual_gauge import _orbit_points

    parameters = layout.reconstruct_parameters(np.asarray([float(v) for v in eta]))
    points = set()
    for value, row, orbit in zip(
        parameters, layout.positions, layout.cover.orbits, strict=True
    ):
        exact = tuple(F(float(v)).limit_denominator(100_000) for v in value)
        for point in _orbit_points(row, exact):
            points.add((point, orbit.atomic_number))
    return frozenset(points)


@pytest.mark.parametrize(
    ("group", "orbits", "nu", "counts"),
    [
        ("P2_1", [("2a", 6), ("2a", 8)], 2, {6: 1, 8: 1}),  # polar: anchored origin
        ("P2_13", [("4a", 6), ("4a", 8)], 4, {6: 1, 8: 1}),
        ("Pmmm", [("2i", 6), ("1a", 8)], 1, {6: 2, 8: 1}),
        ("P-1", [("2i", 6), ("2i", 6)], 4, {6: 1}),  # identical orbits too
    ],
)
@pytest.mark.parametrize("improper", [False, True])
def test_eta_actions_decode_to_congruent_crystals_exactly(  # type: ignore[no-untyped-def]
    group, orbits, nu, counts, improper
) -> None:
    from crystalatlas.normalizer import (
        euclidean_normalizer,
        free_origin_axes,
        proper_euclidean_normalizer,
    )

    layout = _layout(group, orbits, nu, counts)
    gauge = cover_residual_gauge(layout, improper=improper)
    setting = layout.setting
    free = free_origin_axes(setting)
    eta = tuple(F(11 * i + 5, 97) for i in range(layout.intrinsic_dimension))
    reference = _exact_crystal(layout, eta)
    cosets = (euclidean_normalizer if improper else proper_euclidean_normalizer)(setting)
    for matrix, shift in gauge.all_eta_actions():
        moved = np.mod(np.asarray([float(v) for v in eta]) @ matrix + shift, 1.0)
        target = _exact_crystal(layout, [F(float(v)).limit_denominator(100_000) for v in moved])
        matched = False
        for g in cosets:
            image = frozenset((tuple(v % 1 for v in g.apply_exact(p)), z) for p, z in reference)
            anchor_point, anchor_z = next(iter(image))
            for point, z in target:
                if z != anchor_z:
                    continue
                offset = tuple(
                    (point[a] - anchor_point[a]) if a in free else F(0) for a in range(3)
                )
                if (
                    frozenset(
                        (tuple((p[a] + offset[a]) % 1 for a in range(3)), w) for p, w in image
                    )
                    == target
                ):
                    matched = True
                    break
            if matched:
                break
        assert matched, (group, matrix.tolist(), shift.tolist())


def test_p2_13_4a_racemic_gauge_is_plus_minus_x_plus_quarters() -> None:
    """normalizer-v3 adds the improper elements x -> -x + k/2 (Ia-3d over P2_13).

    The order-8 group {+-x + k/4} fixes x = k/8: x = 0, 1/4, 1/2, 3/4 (FCC, fixed
    by improper elements) and x = 1/8 + k/4 (fixed by proper ones).
    """

    layout = _layout("P2_13", [("4a", 6)], 4, {6: 1})
    racemic = cover_residual_gauge(layout, improper=True)
    assert racemic.certified and racemic is not cover_residual_gauge(layout)
    eta = np.asarray([0.12])
    images = sorted(
        float(np.mod(eta @ matrix + shift, 1.0)[0])
        for matrix, shift in racemic.all_eta_actions()
    )
    expected = sorted(np.mod([sign * 0.12 + k / 4 for sign in (1, -1) for k in range(4)], 1.0))
    np.testing.assert_allclose(images, expected, atol=1e-12)
    assert len(cover_residual_gauge(layout).all_eta_actions()) == 4  # normalizer-v2 unchanged


@pytest.mark.parametrize(
    ("group", "orbits", "nu", "counts"),
    [
        ("Pm-3m", [("6e", 6), ("1a", 8)], 1, {6: 6, 8: 1}),
        ("P-1", [("2i", 8)], 2, {8: 1}),
        ("Pmmm", [("2i", 6), ("1a", 8)], 1, {6: 2, 8: 1}),
    ],
)
def test_racemic_gauge_equals_the_proper_gauge_for_centrosymmetric_covers(  # type: ignore[no-untyped-def]
    group, orbits, nu, counts
) -> None:
    layout = _layout(group, orbits, nu, counts)
    keys = [
        {
            tuple(np.round(np.concatenate([m.ravel(), np.mod(t, 1.0)]), 12))
            for m, t in cover_residual_gauge(layout, improper=improper).all_eta_actions()
        }
        for improper in (False, True)
    ]
    assert keys[0] == keys[1]
