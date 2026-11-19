"""Proper Euclidean normalizers for general metrics (residual gauge, option (a)).

In row convention an operation (R, t) acts as x -> x R + t.  Conjugation by
g = (W, w) gives R' = W^-1 R W and t' = t W + w - w R' (mod 1).  g normalizes G
when every conjugate lies in G.  For a general metric, W G W^T = G must hold for
every metric G that the point group preserves.  Proper means det W = +1: under
the SO(3) chirality quotient an improper element maps a crystal to its
enantiomorph.  Translations along continuous-origin axes are quotiented (the
layout's origin gauge).

The expected cases follow from the definition:
* P1: the continuous origin is quotiented, so only the identity remains.
* P-1: conjugating the inversion by a translation w gives 2w = 0 mod 1, hence
  the 8 half-translations.
* P2_13: the proper part has index 4.  It holds the centring translation
  (1/2, 1/2, 1/2) and the 2-fold rotation about [0 1 -1] followed by
  (1/4, 1/4, 1/4), i.e. I4_132 over P2_13 (full Euclidean normalizer Ia-3d).
  On the 4a orbit (x, x, x) they act as x -> x + 1/2, 1/4 - x, 3/4 - x; this is
  checked on exact orbit point sets, independently of the search.
* Pm-3m: the translation (1/2, 1/2, 1/2) maps the inversion centre at the
  origin to the one at the body centre (index 2).
* Fm-3m: the translation (1/2, 0, 0) is not in the F lattice but normalizes the
  group (index 2 in the proper part).
"""

from __future__ import annotations

from fractions import Fraction
from functools import cache

import pytest

from crystalatlas.algebra import AffineOperation
from crystalatlas.full_catalogue import build_space_group_setting
from crystalatlas.normalizer import (
    compose,
    euclidean_normalizer,
    invariant_metric_basis,
    normalizes,
    proper_euclidean_normalizer,
)

HALF = Fraction(1, 2)
ZERO = Fraction(0)
IDENTITY = ((1, 0, 0), (0, 1, 0), (0, 0, 1))


@cache
def _setting(number: int):  # type: ignore[no-untyped-def]
    return build_space_group_setting(number)


def _translations(cosets) -> set[tuple[Fraction, ...]]:  # type: ignore[no-untyped-def]
    assert all(tuple(map(tuple, item.rotation)) == IDENTITY for item in cosets)
    return {tuple(item.translation) for item in cosets}


def test_p1_has_only_the_identity_after_the_continuous_origin_quotient() -> None:
    assert _translations(proper_euclidean_normalizer(_setting(1))) == {(ZERO,) * 3}


def test_p_minus_1_has_exactly_the_eight_half_translations() -> None:
    expected = {(a, b, c) for a in (ZERO, HALF) for b in (ZERO, HALF) for c in (ZERO, HALF)}
    assert _translations(proper_euclidean_normalizer(_setting(2))) == expected


def test_pm_3m_gains_the_body_centring_translation() -> None:
    cosets = proper_euclidean_normalizer(_setting(221))
    assert _translations(cosets) == {(ZERO,) * 3, (HALF, HALF, HALF)}


def test_p2_13_proper_normalizer_acts_on_4a_as_the_order_4_parameter_gauge() -> None:
    setting = _setting(198)
    cosets = proper_euclidean_normalizer(setting)
    assert len(cosets) == 4
    row = setting.wyckoff("4a")
    for x in (Fraction(3, 25), Fraction(3, 10), Fraction(41, 100)):
        images = set()
        for item in cosets:
            points = {item.apply_exact(point) for point in row.expand_exact((x,))}
            matches = [
                y
                for y in (x, x + HALF, Fraction(1, 4) - x, Fraction(3, 4) - x)
                if points
                == {tuple(c % 1 for c in point) for point in row.expand_exact((y % 1,))}
            ]
            assert len(matches) == 1
            images.add(matches[0] % 1)
        assert len(images) == 4


def test_fm_3m_gains_one_half_translation_coset() -> None:
    cosets = proper_euclidean_normalizer(_setting(225))
    assert len(cosets) == 2
    nontrivial = next(item for item in cosets if any(item.translation))
    # Any representative is (1/2, 0, 0), (0, 1/2, 0), (0, 0, 1/2) or (1/2, 1/2, 1/2) mod F.
    assert sum(1 for value in nontrivial.translation if value == HALF) in (1, 3)


def test_invariant_metric_basis_has_the_crystal_family_dimension() -> None:
    expected = {1: 6, 3: 4, 16: 3, 75: 2, 143: 2, 168: 2, 195: 1}
    for number, dimension in expected.items():
        assert len(invariant_metric_basis(_setting(number).point_group_rotations)) == dimension


@pytest.mark.parametrize("number", [2, 14, 47, 63, 136, 166, 194, 198, 225, 227, 230])
def test_cosets_normalize_the_group_form_a_group_and_permute_wyckoff_positions(number) -> None:  # type: ignore[no-untyped-def]
    setting = _setting(number)
    cosets = proper_euclidean_normalizer(setting)
    assert any(
        tuple(map(tuple, item.rotation)) == IDENTITY and not any(item.translation)
        for item in cosets
    )
    for item in cosets:
        assert normalizes(setting, item)
        # A normalizer element maps every Wyckoff position onto one of equal multiplicity.
        for row in setting.wyckoff_positions.values():
            points = {
                item.apply_exact(point) for point in row.expand_exact(row.generic_parameter)
            }
            # The image is again a G-orbit (closed under G) of the same multiplicity.
            assert len(points) == row.multiplicity
            assert {op.apply_exact(point) for op in setting.operations for point in points} == (
                points
            ), (setting.key, row.label)
    for left in cosets:
        for right in cosets:
            product = compose(left, right)
            assert normalizes(setting, product)


def test_completeness_on_a_finer_translation_grid() -> None:
    """Denominator 48 finds no coset beyond denominator 24 on screw/glide-rich groups."""

    for number in (88, 98, 109, 141, 142, 203, 210, 214, 227, 228, 230):
        setting = _setting(number)
        coarse = proper_euclidean_normalizer(setting)
        fine = proper_euclidean_normalizer(setting, translation_denominator=48)
        assert len(fine) == len(coarse), setting.key


def test_identity_operation_is_neutral_under_composition() -> None:
    setting = _setting(198)
    identity = AffineOperation.identity()
    for item in proper_euclidean_normalizer(setting):
        assert compose(identity, item) == item and compose(item, identity) == item


# --- full (proper and improper) Euclidean normalizer: normalizer-v3 -------------

# Space-group types with proper operations only (65), and the 11 enantiomorphic
# pairs among them (ITA Table 3.1.3.1 lists the 22 enantiomorphic types).
SOHNCKE = (
    {1, 3, 4, 5}
    | set(range(16, 25))
    | set(range(75, 81))
    | set(range(89, 99))
    | set(range(143, 147))
    | set(range(149, 156))
    | set(range(168, 174))
    | set(range(177, 183))
    | set(range(195, 200))
    | set(range(207, 215))
)
ENANTIOMORPHIC_PAIRS = (
    (76, 78),
    (91, 95),
    (92, 96),
    (144, 145),
    (151, 153),
    (152, 154),
    (169, 170),
    (171, 172),
    (178, 179),
    (180, 181),
    (212, 213),
)
MINUS_ONE = ((-1, 0, 0), (0, -1, 0), (0, 0, -1))


def _det3(m) -> int:  # type: ignore[no-untyped-def]
    return (
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    )


def test_the_sohncke_and_enantiomorphic_tables_have_their_known_sizes() -> None:
    members = {number for pair in ENANTIOMORPHIC_PAIRS for number in pair}
    assert len(SOHNCKE) == 65 and len(members) == 22 and members <= SOHNCKE


def test_full_normalizer_of_p2_13_adds_the_inversion_cosets() -> None:
    """Ia-3d over P2_13: index 8, twice the proper index 4.

    The inversion at the origin maps the 4a orbit at x onto the orbit at -x
    (tests/test_residual_gauge.py), so it is a residual gauge only up to the
    enantiomorph: normalizer-v3, not normalizer-v2.
    """

    from crystalatlas.normalizer import _coset, _representative, free_origin_axes

    setting = _setting(198)
    proper, full = proper_euclidean_normalizer(setting), euclidean_normalizer(setting)
    assert len(proper) == 4 and len(full) == 8
    assert set(proper) <= set(full)
    assert sorted(_det3(item.rotation) for item in full) == [-1] * 4 + [1] * 4
    inversion = AffineOperation(MINUS_ONE, (ZERO,) * 3)
    assert normalizes(setting, inversion)
    assert _representative(_coset(inversion, setting, free_origin_axes(setting))) in full
    for item in full:
        assert normalizes(setting, item)


@pytest.mark.parametrize("number", [2, 14, 47, 221, 225, 230])
def test_full_normalizer_equals_the_proper_one_for_centrosymmetric_groups(number: int) -> None:
    """An improper element times the group's inversion is proper: same cosets."""

    setting = _setting(number)
    assert euclidean_normalizer(setting) == proper_euclidean_normalizer(setting)


@pytest.mark.parametrize(
    "number",
    # P1, P2_1, P2_12_12_1, P4_22_12, P2_13, I4_132: non-enantiomorphic Sohncke;
    # P4_1, P3_1, P4_332: enantiomorphic; P4mm, Pmn2_1, P-42m, P-43m: acentric
    # with improper operations.
    [1, 4, 19, 94, 198, 214, 76, 144, 212, 99, 31, 111, 215],
)
def test_full_normalizer_doubles_the_proper_one_exactly_for_non_enantiomorphic_sohncke(
    number: int,
) -> None:
    """[N : N+] = 2 iff G has no improper operation and is not enantiomorphic.

    If G holds an improper operation g0, an improper normalizer element n is
    (n g0) g0^-1 with n g0 proper, so the cosets coincide.  A Sohncke type is
    non-enantiomorphic iff an improper isometry maps it onto itself, i.e. iff it
    has an improper normalizer element.  The all-230 check is
    artifacts/peer_review_2026-10-19/verify_full_normalizer_all_groups.py.
    """

    enantiomorphic = {n for pair in ENANTIOMORPHIC_PAIRS for n in pair}
    setting = _setting(number)
    expected = 2 if number in SOHNCKE and number not in enantiomorphic else 1
    proper, full = proper_euclidean_normalizer(setting), euclidean_normalizer(setting)
    assert len(full) == expected * len(proper)
    assert set(proper) <= set(full)
    for item in full:
        assert abs(_det3(item.rotation)) == 1 and normalizes(setting, item)
