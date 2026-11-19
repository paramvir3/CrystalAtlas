"""Certified tolerance-based periodic structure matcher (review finding B3).

Rejection tests carry an independent certificate.  With one atom per species
the bijection is forced, and with a lattice whose only proper correspondence
is the identity the minimax displacement of any translation is at least half
the largest pairwise torus distance between per-site displacements (triangle
inequality).  Acceptance tests either plant a witness or audit the returned
witness with numpy and the certified periodic solver.
"""

from __future__ import annotations

import math
from fractions import Fraction
from itertools import permutations

import numpy as np
import pytest

from crystalatlas.diagnostics import (
    AcceptableTarget,
    BootstrapProtocol,
    DiagnosticEvaluationContract,
    GeometryDecompositionCase,
    RankedAttempt,
    geometry_corpus_fingerprints,
    matcher_callable_sha256,
)
from crystalatlas.errors import DomainError
from crystalatlas.periodic import closest_periodic_image
from crystalatlas.physical_matching import (
    SITE_TOLERANCE,
    _assignment,
    _closest_image_squared,
    _correspondences_from,
    _cosine_window,
    _cube_root,
    _integer_cube_root,
    _lll_reduce,
    _normalized_metric,
    _reference_bases,
    _short_vectors,
    _transform_gram,
    encode_physical_structure,
    evaluate_physical_geometry_decomposition,
    physical_structure_match,
    physical_structure_matcher_contract,
    physical_structure_rms,
    physical_structure_witness,
    validate_physical_payload,
)


def _random_structure(seed: int, species: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    lattice = np.diag([4.0, 4.6, 5.2]) + 0.6 * rng.standard_normal((3, 3))
    if np.linalg.det(lattice) < 0:
        lattice[0] *= -1.0
    return lattice, rng.random((len(species), 3))


def _proper_rotation(seed: int) -> np.ndarray:
    q, r = np.linalg.qr(np.random.default_rng(seed).standard_normal((3, 3)))
    q = q @ np.diag(np.sign(np.diag(r)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1.0
    return q


def _from_parameters(a, b, c, alpha, beta, gamma):  # type: ignore[no-untyped-def]
    al, be, ga = np.radians([alpha, beta, gamma])
    cx = c * np.cos(be)
    cy = c * (np.cos(al) - np.cos(be) * np.cos(ga)) / np.sin(ga)
    cz = np.sqrt(c * c - cx * cx - cy * cy)
    return np.array([[a, 0.0, 0.0], [b * np.cos(ga), b * np.sin(ga), 0.0], [cx, cy, cz]])


_UNIMODULAR = np.array([[1, 1, 0], [0, 1, 0], [0, -1, 1]], dtype=np.int64) @ np.array(
    [[1, 0, 0], [2, 1, 0], [0, 0, 1]], dtype=np.int64
)
_CHIRAL = (8, 14, 26, 29)
# A triclinic cell whose only proper correspondence under the frozen
# 30%/10-degree lattice filter is the identity (asserted below).
_RIGID = _from_parameters(3.1, 4.6, 7.2, 64.0, 76.0, 98.0)


def _payload(lattice, fractional, species):  # type: ignore[no-untyped-def]
    return encode_physical_structure(lattice, fractional, species)


def _unit(lattice: np.ndarray, count: int) -> float:
    return float(np.linalg.det(lattice) / count) ** (1.0 / 3.0)


def _parse(payload: bytes) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Test-side reader, independent of the matcher's parser."""

    lines = payload.decode("ascii").splitlines()
    lattice = np.array([float(v) for v in lines[1].split()[1:]]).reshape(3, 3)
    sites = [line.split() for line in lines[2:]]
    return (
        lattice,
        np.array([[float(v) for v in fields[2:]] for fields in sites]),
        [int(fields[1]) for fields in sites],
    )


def _audit_witness(candidate: bytes, target: bytes) -> float:
    """Independently verify the returned witness; return its normalized RMS."""

    witness = physical_structure_witness(candidate, target)
    assert witness is not None
    total, swapped, transform, basis, translation, pairs = witness
    first, second = (target, candidate) if swapped else (candidate, target)
    first_lattice, first_points, first_species = _parse(first)
    second_lattice, second_points, second_species = _parse(second)
    count = len(first_species)
    transform_m = np.array(transform, dtype=float)
    basis_m = np.array(basis, dtype=float)
    assert round(np.linalg.det(transform_m)) == 1 and round(np.linalg.det(basis_m)) == 1
    rows_a = transform_m @ (first_lattice / _unit(first_lattice, count))
    rows_b = basis_m @ (second_lattice / _unit(second_lattice, count))
    # Lattice filter: lengths within a factor 1.3 and angles within 10 degrees.
    lengths_a = np.linalg.norm(rows_a, axis=1)
    lengths_b = np.linalg.norm(rows_b, axis=1)
    assert np.all(lengths_a <= 1.3 * lengths_b * (1 + 1e-12))
    assert np.all(lengths_b <= 1.3 * lengths_a * (1 + 1e-12))
    for i, j in ((0, 1), (0, 2), (1, 2)):
        angle_a = math.degrees(math.acos(rows_a[i] @ rows_a[j] / (lengths_a[i] * lengths_a[j])))
        angle_b = math.degrees(math.acos(rows_b[i] @ rows_b[j] / (lengths_b[i] * lengths_b[j])))
        assert abs(angle_a - angle_b) <= 10.0 + 1e-7
    metric = 0.5 * (rows_a @ rows_a.T + rows_b @ rows_b.T)
    common = np.linalg.cholesky(metric)
    mapped_a = first_points @ np.linalg.inv(transform_m)
    mapped_b = second_points @ np.linalg.inv(basis_m)
    assert sorted(i for i, _ in pairs) == list(range(count))
    assert sorted(j for _, j in pairs) == list(range(count))
    squared = 0.0
    for i, j in pairs:
        assert first_species[i] == second_species[j]
        delta = mapped_b[j] - mapped_a[i] - np.asarray(translation)
        image = closest_periodic_image(delta, common)
        assert image.squared_distance <= SITE_TOLERANCE**2 * (1 + 1e-12)
        squared += image.squared_distance
    assert squared == pytest.approx(total, rel=1e-9, abs=1e-12)
    return math.sqrt(total / count)


def test_identical_structures_match_with_zero_rms() -> None:
    lattice, fractional = _random_structure(1, _CHIRAL)
    payload = _payload(lattice, fractional, _CHIRAL)
    assert physical_structure_match(payload, payload)
    assert physical_structure_rms(payload, payload) == pytest.approx(0.0, abs=1e-12)


def test_proper_basis_change_origin_rotation_and_permutation_are_invisible() -> None:
    lattice, fractional = _random_structure(2, _CHIRAL)
    assert round(np.linalg.det(_UNIMODULAR)) == 1
    rotated = _UNIMODULAR @ lattice @ _proper_rotation(3)
    moved = np.mod((fractional + np.array([0.31, -0.12, 0.47])) @ np.linalg.inv(_UNIMODULAR), 1)
    order = np.array([2, 0, 3, 1])
    left = _payload(lattice, fractional, _CHIRAL)
    right = _payload(rotated, moved[order], tuple(np.asarray(_CHIRAL)[order]))
    assert physical_structure_match(left, right)
    assert physical_structure_rms(left, right) == pytest.approx(0.0, abs=1e-9)
    assert _audit_witness(left, right) == pytest.approx(0.0, abs=1e-9)


def test_supercell_is_reduced_to_its_primitive_cell_before_matching() -> None:
    lattice, fractional = _random_structure(4, _CHIRAL)
    doubled_lattice = np.diag([2.0, 1.0, 1.0]) @ lattice
    halved = fractional @ np.diag([0.5, 1.0, 1.0])
    doubled = np.vstack((halved, halved + np.array([0.5, 0.0, 0.0])))
    assert physical_structure_match(
        _payload(lattice, fractional, _CHIRAL),
        _payload(doubled_lattice, doubled, _CHIRAL + _CHIRAL),
    )


def test_small_coordinate_noise_matches() -> None:
    lattice, fractional = _random_structure(5, _CHIRAL)
    reference = _payload(lattice, fractional, _CHIRAL)
    noisy = fractional + 0.002 * np.random.default_rng(6).standard_normal(fractional.shape)
    assert physical_structure_match(reference, _payload(lattice, noisy, _CHIRAL))


def test_rigid_lattice_has_only_the_identity_correspondence() -> None:
    """Premise of the forced-bijection certificates, checked on the search path."""

    for count in (2, 4):
        _, reduced = _lll_reduce(_normalized_metric(_RIGID.tolist(), count))
        references = _reference_bases(reduced)
        assert len(references) == 1
        reference_metric = _transform_gram(reduced, references[0])
        vectors = _short_vectors(reduced, 1.69 * max(np.diag(reference_metric)) * 1.01)
        assert _correspondences_from(vectors, reduced, reference_metric) == references


def test_face_centred_cubic_reference_set_contains_inequivalent_reduced_forms() -> None:
    fcc = 4.0 * np.array([[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]])
    _, reduced = _lll_reduce(_normalized_metric(fcc.tolist(), 1))
    angles = set()
    for rows in _reference_bases(reduced):
        gram = np.array(_transform_gram(reduced, rows))
        lengths = np.sqrt(np.diag(gram))
        cosines = [gram[i, j] / (lengths[i] * lengths[j]) for i, j in ((0, 1), (0, 2), (1, 2))]
        angles.add(tuple(sorted(round(math.degrees(math.acos(c))) for c in cosines)))
    assert (60, 60, 60) in angles
    assert (60, 60, 90) in angles


def test_borderline_decisions_do_not_depend_on_representation() -> None:
    """Re-representing both crystals never changes a decision (B3 acceptance)."""

    rng = np.random.default_rng(51)
    targets = (
        np.array([[3.0, 0.0, 0.0], [-1.5, 1.5 * np.sqrt(3.0), 0.0], [0.0, 0.0, 5.0]]),
        4.0 * np.array([[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]]),
    )
    unimodular = (
        _UNIMODULAR,
        np.array([[0, 1, 0], [-1, -1, 0], [1, 2, 1]], dtype=np.int64),
    )
    species = (8, 8, 14)
    outcomes = set()
    for trial in range(24):
        base = targets[trial % 2]
        fractional = rng.random((3, 3))
        strain = np.eye(3) + rng.uniform(0.08, 0.2) * rng.standard_normal((3, 3))
        if np.linalg.det(strain) <= 0:
            continue
        noisy = np.mod(fractional + 0.05 * rng.standard_normal(fractional.shape), 1.0)
        first = _payload(base, fractional, species)
        second = _payload(base @ strain, noisy, species)
        u, v = unimodular[trial % 2], unimodular[(trial + 1) % 2]
        assert round(np.linalg.det(u)) == 1 and round(np.linalg.det(v)) == 1
        order = np.array([2, 0, 1])
        first_again = _payload(
            u @ base @ _proper_rotation(600 + trial),
            np.mod((fractional + rng.random(3)) @ np.linalg.inv(u), 1.0),
            species,
        )
        second_again = _payload(
            v @ base @ strain @ _proper_rotation(700 + trial),
            np.mod((noisy[order] + rng.random(3)) @ np.linalg.inv(v), 1.0),
            tuple(np.asarray(species)[order]),
        )
        decision = physical_structure_match(first, second)
        assert physical_structure_match(second_again, first_again) is decision
        outcomes.add(decision)
    assert outcomes == {True, False}


def test_forced_bijection_threshold_and_rms_are_exact() -> None:
    """Two atoms of distinct species: match iff the torus displacement is <= 2 stol.

    The least-squares step moves both sites by half the relative displacement,
    which is the exact minimax translation for two points on a flat torus.
    """

    base = np.array([[0.11, 0.23, 0.31], [0.57, 0.62, 0.74]])
    species = (8, 14)
    unit = _unit(_RIGID, 2)
    reference = _payload(_RIGID, base, species)
    rng = np.random.default_rng(21)
    outcomes = set()
    for _ in range(120):
        direction = rng.standard_normal(3)
        direction /= np.linalg.norm(direction)
        cartesian = rng.uniform(0.2, 1.4) * unit * direction
        moved = base.copy()
        moved[1] += cartesian @ np.linalg.inv(_RIGID)
        relative = math.sqrt(
            closest_periodic_image(moved[1] - base[1], _RIGID / unit).squared_distance
        )
        if abs(relative - 2 * SITE_TOLERANCE) < 1e-9:
            continue
        expected = relative <= 2 * SITE_TOLERANCE
        other = _payload(_RIGID, moved, species)
        assert physical_structure_match(reference, other) is expected
        if expected:
            assert physical_structure_rms(reference, other) == pytest.approx(
                relative / 2, rel=1e-9
            )
            assert _audit_witness(reference, other) == pytest.approx(relative / 2, rel=1e-9)
        else:
            assert physical_structure_rms(reference, other) is None
        outcomes.add(expected)
    assert outcomes == {True, False}


def test_mirror_image_of_a_chiral_motif_is_rejected_with_a_certificate() -> None:
    motif = np.array(
        [[0.0, 0.0, 0.0], [0.21, 0.05, 0.12], [0.09, 0.31, 0.07], [0.13, 0.08, 0.36]]
    )
    inverted = np.mod(-motif, 1.0)
    unit = _unit(_RIGID, 4)
    displacements = (inverted - motif) @ _RIGID
    certificate = max(
        math.sqrt(
            closest_periodic_image(
                (displacements[i] - displacements[j]) @ np.linalg.inv(_RIGID), _RIGID / unit
            ).squared_distance
        )
        / 2.0
        for i in range(4)
        for j in range(i)
    )
    assert certificate > 1.2 * SITE_TOLERANCE
    reference = _payload(_RIGID, motif, _CHIRAL)
    mirror = _payload(_RIGID, inverted, _CHIRAL)
    assert not physical_structure_match(reference, mirror)
    assert physical_structure_rms(reference, mirror) is None


def test_centrosymmetric_motif_matches_its_inversion_image_exactly() -> None:
    half = np.array([[0.21, 0.05, 0.12], [0.09, 0.31, 0.07]])
    motif = np.vstack((half, np.mod(-half, 1.0)))
    species = (8, 14, 8, 14)
    left = _payload(_RIGID, motif, species)
    right = _payload(_RIGID, np.mod(-motif, 1.0), species)
    assert physical_structure_match(left, right)
    assert physical_structure_rms(left, right) == pytest.approx(0.0, abs=1e-9)


def test_anchor_choice_does_not_depend_on_coordinate_order() -> None:
    """A heuristic-band case whose decision flips if the anchor follows site order.

    Found by search with three refinement steps (seed 7, trial 135); the
    coordinate-order anchor mutation in
    artifacts/peer_review_2026-10-19/verify_b3_mutations.py is killed only by
    this test.  Anchoring on the first species-8 site in sorted order accepts
    one origin and rejects another, although both payloads describe the same
    crystal.
    """

    first = np.array(
        [
            [0.7302720316855383, 0.2133770743741984, 0.8447322944889888],
            [0.6675852240937911, 0.2901616508294579, 0.2686041391040265],
            [0.17703702056259008, 0.4047122060897036, 0.36008293511298217],
            [0.28534036038033417, 0.873761451641813, 0.7609010616288382],
            [0.49277699801313257, 0.9627202260670041, 0.6736685795543035],
        ]
    )
    second = np.array(
        [
            [0.9820400764454369, 0.4170513681397843, 0.87909698637442],
            [0.09429055175185441, 0.20453112189520634, 0.24294414763561184],
            [0.4240962718654664, 0.5798099556971984, 0.3997785075373397],
            [0.05181537550319079, 0.7477324368856171, 0.7325478808393928],
            [0.9618758100587554, 1.042918414419152, 0.6645866410332504],
        ]
    )
    species = (8, 8, 14, 14, 14)
    reference = _payload(_RIGID, first, species)
    found = np.array([0.8869760373575541, 0.8449236469082152, 0.34599416756119383])
    shifts = [np.zeros(3), found]
    shifts += list(np.random.default_rng(61).random((6, 3)))
    decisions = set()
    for shift in shifts:
        moved = _payload(_RIGID, np.mod(second + shift, 1.0), species)
        decisions.add(physical_structure_match(reference, moved))
    assert decisions == {True}


def test_iterated_refinement_recovers_a_heuristic_band_witness() -> None:
    """MP-20 mp-1185903 near miss (verify_b3_pymatgen_agreement.py, seed 20261019).

    StructureMatcher accepts this pair with a proper map at max distance 0.450;
    the same witness scores 0.451 in this module's metric, inside the
    heuristic band (stol/2, stol].  One least-squares step missed it.
    """

    target_lattice = np.array(
        [
            [5.794034993178307, 0.0, 1.6505080579413587],
            [4.748775187246569, 3.3196348779897544, 1.6505080579413587],
            [0.0, 0.0, 7.93524964],
        ]
    )
    target_sites = np.array(
        [
            [0.939849, 0.939849, 0.614253],
            [0.616051, 0.616051, 0.274573],
            [0.001774, 0.001774, 0.004927],
            [0.280889, 0.280889, 0.940035],
            [0.657986, 0.657986, 0.659922],
            [0.336784, 0.336784, 0.339624],
        ]
    )
    candidate_lattice = np.array(
        [
            [5.146618471646596, 1.2822863034214436, 1.937212825441868],
            [4.932522075370963, 4.792122992277181, 2.0084944668108196],
            [0.45417018861459146, 0.3133969549244095, 7.7193136675084935],
        ]
    )
    candidate_sites = np.array(
        [
            [0.9747010674029623, 0.8971194899404211, 0.6276425070299487],
            [0.5760645825585101, 0.6660671232396048, 0.3053032006614432],
            [0.02836655724994462, 0.9625539766842791, 0.032787315599104604],
            [0.267386178732103, 0.29387715805332054, 0.9088504449031852],
            [0.6692144579388599, 0.5960512667756854, 0.6804060720824477],
            [0.2496043145683532, 0.39631292617337865, 0.3281245593346884],
        ]
    )
    species = (12, 12, 49, 49, 49, 49)
    target = _payload(target_lattice, target_sites, species)
    candidate = _payload(candidate_lattice, candidate_sites, species)
    assert physical_structure_match(candidate, target)
    assert physical_structure_match(target, candidate)
    assert _audit_witness(candidate, target) <= SITE_TOLERANCE


def test_planted_witness_within_half_tolerance_is_always_found() -> None:
    """Completeness guarantee: a witness at <= stol/2 is never missed."""

    species = (8, 8, 8, 14, 14, 26)
    rng = np.random.default_rng(31)
    for seed in range(12):
        lattice, fractional = _random_structure(200 + seed, species)
        unit = _unit(lattice, len(species))
        displacement = rng.standard_normal((len(species), 3))
        displacement /= np.linalg.norm(displacement, axis=1, keepdims=True)
        displacement *= (0.49 * SITE_TOLERANCE * unit) * rng.random((len(species), 1))
        noisy = fractional + displacement @ np.linalg.inv(lattice)
        rotated = _UNIMODULAR @ lattice @ _proper_rotation(300 + seed)
        moved = np.mod((noisy + rng.random(3)) @ np.linalg.inv(_UNIMODULAR), 1.0)
        order = rng.permutation(len(species))
        left = _payload(lattice, fractional, species)
        right = _payload(rotated, moved[order], tuple(np.asarray(species)[order]))
        assert physical_structure_match(left, right)
        assert _audit_witness(left, right) <= SITE_TOLERANCE


def test_every_accepted_witness_is_independently_valid() -> None:
    species = (8, 8, 14, 14, 26)
    rng = np.random.default_rng(41)
    accepted = 0
    rejected = 0
    for seed in range(16):
        lattice, fractional = _random_structure(400 + seed, species)
        strain = np.eye(3) + rng.uniform(0.0, 0.12) * rng.standard_normal((3, 3))
        other = fractional + rng.uniform(0.0, 0.2) * rng.standard_normal(fractional.shape)
        left = _payload(lattice, fractional, species)
        right = _payload(lattice @ strain @ _proper_rotation(500 + seed), other, species)
        if physical_structure_match(left, right):
            assert _audit_witness(left, right) <= SITE_TOLERANCE
            accepted += 1
        else:
            assert physical_structure_witness(left, right) is None
            rejected += 1
    assert accepted and rejected


def test_lattice_tolerance_and_volume_scaling() -> None:
    lattice, fractional = _random_structure(7, _CHIRAL)
    reference = _payload(lattice, fractional, _CHIRAL)
    # Uniform rescaling is quotiented (volume-scaled comparison).
    assert physical_structure_match(reference, _payload(1.1 * lattice, fractional, _CHIRAL))
    # A 60% anisotropic stretch exceeds the frozen 30% length tolerance.
    stretched = np.diag([1.6, 1.0, 1.0]) @ lattice
    assert not physical_structure_match(reference, _payload(stretched, fractional, _CHIRAL))


def test_composition_and_atom_count_mismatches_never_match() -> None:
    lattice, fractional = _random_structure(9, _CHIRAL)
    reference = _payload(lattice, fractional, _CHIRAL)
    other_species = _payload(lattice, fractional, (8, 14, 26, 30))
    fewer_atoms = _payload(lattice, fractional[:3], _CHIRAL[:3])
    assert not physical_structure_match(reference, other_species)
    assert not physical_structure_match(reference, fewer_atoms)


def test_decision_is_exactly_symmetric() -> None:
    lattice, fractional = _random_structure(10, _CHIRAL)
    rng = np.random.default_rng(11)
    for scale in (0.0, 0.01, 0.05, 0.08, 0.12):
        other = fractional + scale * rng.standard_normal(fractional.shape)
        left = _payload(lattice, fractional, _CHIRAL)
        right = _payload(lattice @ _proper_rotation(12), other, _CHIRAL)
        assert physical_structure_match(left, right) == physical_structure_match(right, left)
        assert physical_structure_rms(left, right) == physical_structure_rms(right, left)


def test_assignment_equals_bruteforce_optimum() -> None:
    rng = np.random.default_rng(13)
    for size in range(1, 7):
        cost = rng.random((size, size)).tolist()
        assigned = _assignment(cost)
        best = min(
            sum(cost[i][p[i]] for i in range(size)) for p in permutations(range(size))
        )
        assert sorted(assigned) == list(range(size))
        assert sum(cost[i][assigned[i]] for i in range(size)) == pytest.approx(best)


def test_closest_image_equals_certified_periodic_solver_in_skew_cells() -> None:
    rng = np.random.default_rng(14)
    lattices = [np.array([[1.0, 0.0, 0.0], [0.97, 0.25, 0.0], [0.4, 0.3, 0.2]])]
    lattices += [np.eye(3) + 0.5 * rng.standard_normal((3, 3)) for _ in range(4)]
    for lattice in lattices:
        if np.linalg.det(lattice) < 0:
            lattice = lattice[[1, 0, 2]]
        metric = (lattice @ lattice.T).tolist()
        for delta in rng.uniform(-3.0, 3.0, size=(100, 3)):
            squared, vector = _closest_image_squared(delta.tolist(), metric)
            expected = closest_periodic_image(delta, lattice).squared_distance
            assert squared == pytest.approx(expected, rel=1e-12, abs=1e-14)
            shift = np.asarray(vector) - delta
            assert np.allclose(shift, np.rint(shift), atol=1e-12)


def test_integer_cube_root_is_exact_and_float_cube_root_is_within_one_ulp() -> None:
    for root in (1, 2, 3, 10, 12345, 2**40 + 7, 3**50):
        cube = root**3
        assert _integer_cube_root(cube) == root
        assert _integer_cube_root(cube - 1) == root - 1
        assert _integer_cube_root(cube + 1) == root
    assert _integer_cube_root(0) == 0
    rng = np.random.default_rng(15)
    for value in [1e-300, 0.125, 1.0, 2.0, 27.0, 1e300, *rng.uniform(0.1, 200.0, 50)]:
        result = _cube_root(float(value))
        exact = Fraction(float(value))
        assert Fraction(math.nextafter(result, 0.0)) ** 3 < exact
        assert Fraction(math.nextafter(result, math.inf)) ** 3 > exact
    assert _cube_root(27.0) == 3.0


def test_cosine_window_is_equivalent_to_the_angle_tolerance() -> None:
    rng = np.random.default_rng(16)
    checked = 0
    for reference, candidate in rng.uniform(0.0, 180.0, size=(4000, 2)):
        if abs(abs(reference - candidate) - 10.0) < 1e-6:
            continue
        lower, upper = _cosine_window(math.cos(math.radians(reference)))
        inside = lower <= math.cos(math.radians(candidate)) <= upper
        assert inside == (abs(reference - candidate) <= 10.0)
        checked += 1
    for reference in (0.0, 3.0, 90.0, 177.0, 180.0):
        lower, upper = _cosine_window(math.cos(math.radians(reference)))
        assert lower <= math.cos(math.radians(reference)) <= upper
    assert checked > 3900


def test_matcher_satisfies_the_certified_diagnostic_purity_contract() -> None:
    contract = physical_structure_matcher_contract()
    assert contract.implementation_sha256 == matcher_callable_sha256(physical_structure_match)
    manifest = contract.to_manifest()
    assert manifest["parameters"]["length_tolerance"] == 0.3
    assert manifest["parameters"]["site_tolerance"] == 0.5
    assert manifest["parameters"]["angle_tolerance_degrees"] == 10.0
    assert manifest["parameters"]["rotation_group"] == "SO(3)"
    assert manifest["parameters"]["angle_tolerance_cosine"] == pytest.approx(
        math.cos(math.radians(10.0)), abs=2e-16
    )


def test_encoder_reduces_arbitrarily_skewed_input_bases() -> None:
    """A skewed basis (sigma_min ~ 0.01 A) must not exhaust the periodic budget."""

    fcc = 4.0 * np.array([[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]])
    skew = np.linalg.matrix_power(_UNIMODULAR, 4)
    assert round(np.linalg.det(skew)) == 1
    assert np.linalg.svd(skew @ fcc, compute_uv=False)[-1] < 0.05
    # Two atoms of the least-populated species exercise the translation search.
    fractional = np.array(
        [[0.0, 0.0, 0.0], [0.27, 0.31, 0.12], [0.61, 0.08, 0.44], [0.15, 0.72, 0.58]]
    )
    species = (8, 8, 14, 14)
    plain = _payload(fcc, fractional, species)
    skewed = _payload(skew @ fcc, np.mod(fractional @ np.linalg.inv(skew), 1.0), species)
    assert physical_structure_match(plain, skewed)
    assert physical_structure_rms(plain, skewed) == pytest.approx(0.0, abs=1e-9)


def test_encoder_rejects_left_handed_and_malformed_input() -> None:
    lattice, fractional = _random_structure(15, _CHIRAL)
    with pytest.raises(DomainError):
        encode_physical_structure(lattice @ np.diag([1.0, 1.0, -1.0]), fractional, _CHIRAL)
    with pytest.raises(DomainError):
        encode_physical_structure(lattice, fractional, _CHIRAL[:2])


def test_noncanonical_payloads_fail_closed() -> None:
    lattice, fractional = _random_structure(17, _CHIRAL)
    payload = _payload(lattice, fractional, _CHIRAL)
    validate_physical_payload(payload)
    lines = payload.decode("ascii").split("\n")
    tampered = [
        payload.replace(b"site 8 ", b"site 08 "),
        payload + b"site 1 0.5 0.5 0.5\n",
        "\n".join([lines[0], lines[1], lines[3], lines[2], *lines[4:]]).encode("ascii"),
        payload.replace(b"/1 ", b"/2 "),
        payload.rstrip(b"\n"),
    ]
    first_float = lines[1].split(" ")[1]
    tampered.append(payload.replace(first_float.encode(), (first_float + "0").encode(), 1))
    for bad in tampered:
        with pytest.raises(DomainError):
            validate_physical_payload(bad)
        with pytest.raises(ValueError):
            physical_structure_match(bad, payload)
    with pytest.raises(TypeError):
        physical_structure_match(payload.decode("ascii"), payload)  # type: ignore[arg-type]


def test_physical_geometry_decomposition_end_to_end() -> None:
    lattice, fractional = _random_structure(18, _CHIRAL)
    target = _payload(lattice, fractional, _CHIRAL)
    shifted = np.mod(fractional + 0.3, 1.0)
    equivalent = _payload(lattice @ _proper_rotation(19), shifted, _CHIRAL)
    different = _payload(np.diag([1.6, 1.0, 1.0]) @ lattice, fractional, _CHIRAL)
    cases = (
        GeometryDecompositionCase(
            "q-1",
            (AcceptableTarget("target", target),),
            (RankedAttempt("oracle", 1.0, True, equivalent),),
            (RankedAttempt("selected", 1.0, True, different),),
        ),
    )
    contract = geometry_corpus_fingerprints(cases).bind(
        DiagnosticEvaluationContract(
            dataset_manifest_sha256="1" * 64,
            split_manifest_sha256="2" * 64,
            checkpoint_sha256="3" * 64,
            target_corpus_sha256="4" * 64,
            generated_attempt_corpus_sha256="b" * 64,
            generation_protocol_sha256="5" * 64,
            generation_seed=17,
            attempt_budget=1,
            score_policy="test-score-descending",
            validity_policy="test-invalid-attempts-retained",
        )
    )
    result = evaluate_physical_geometry_decomposition(
        cases,
        ks=(1,),
        evaluation_contract=contract,
        bootstrap=BootstrapProtocol(samples=200, confidence=0.90, seed=1),
    )
    assert result.at_k[0].oracle_scaffold_match.value == 1.0
    assert result.at_k[0].predicted_scaffold_match.value == 0.0
    malformed = (
        GeometryDecompositionCase(
            "q-1",
            (AcceptableTarget("target", target),),
            (RankedAttempt("oracle", 1.0, True, b"not a structure"),),
            (RankedAttempt("selected", 1.0, True, different),),
        ),
    )
    with pytest.raises(DomainError, match="invalid physical-structure payload"):
        evaluate_physical_geometry_decomposition(
            malformed, ks=(1,), evaluation_contract=contract
        )


def test_optional_agreement_with_pymatgen_structure_matcher() -> None:
    """Isolated adapter check (theory S5 firewall): pymatgen is never a core import."""

    pymatgen_core = pytest.importorskip("pymatgen.core")
    analysis = pytest.importorskip("pymatgen.analysis.structure_matcher")
    matcher = analysis.StructureMatcher(ltol=0.3, stol=0.5, angle_tol=10.0)
    rng = np.random.default_rng(16)
    agree = 0
    total = 0
    for seed in range(40):
        lattice, fractional = _random_structure(100 + seed, _CHIRAL)
        other = fractional + rng.uniform(0.0, 0.15) * rng.standard_normal(fractional.shape)
        ours = physical_structure_match(
            _payload(lattice, fractional, _CHIRAL), _payload(lattice, other, _CHIRAL)
        )
        theirs = matcher.fit(
            pymatgen_core.Structure(lattice, list(_CHIRAL), fractional),
            pymatgen_core.Structure(lattice, list(_CHIRAL), np.mod(other, 1.0)),
        )
        agree += int(ours == theirs)
        total += 1
    assert agree / total >= 0.95
