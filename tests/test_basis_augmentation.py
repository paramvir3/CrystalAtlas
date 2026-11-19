"""Bounded unimodular basis-change augmentation (review 2026-10-29, C2).

For P1, P-1 and the monoclinic groups the canonical training presentation is a
fundamental domain of the GL(3, Z) basis gauge with sharp boundaries.  The
``shear-v1`` augmentation moves a record to a neighbouring presentation by an
admissible elementary shear: the same crystal, described in the basis
``x -> x W`` with ``G' = W^-1 G W^-T``.  Admissibility is exact: ``(W, 0)`` must
normalize the setting and map every row of the cover onto itself.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from crystalatlas.algebra import AffineOperation
from crystalatlas.basis_augmentation import (
    CoverBasisAugmentation,
    cover_basis_augmentation,
    elementary_shears,
)
from crystalatlas.decoder import CrystalScaffold, OrbitAssignment
from crystalatlas.full_catalogue import build_space_group_setting
from crystalatlas.geometry import ContinuousCover, ContinuousGeometry, IntrinsicCoverLayout
from crystalatlas.normalizer import normalizes


def _scaffold(group, composition, *orbits):  # type: ignore[no-untyped-def]
    return CrystalScaffold.from_mapping(
        nu=1,
        space_group=group,
        reduced_composition=composition,
        orbits=tuple(OrbitAssignment(*orbit) for orbit in orbits),
    )


def _positions_by_species(decoded):  # type: ignore[no-untyped-def]
    table: dict[int, np.ndarray] = {}
    for number in sorted(set(int(z) for z in decoded.atomic_numbers)):
        rows = decoded.fractional_positions[decoded.atomic_numbers == number]
        table[number] = rows
    return table


def _same_point_set(left: np.ndarray, right: np.ndarray, tol: float = 1e-9) -> bool:
    if left.shape != right.shape:
        return False
    unmatched = list(range(len(right)))
    for row in left:
        for candidate in unmatched:
            gap = np.remainder(row - right[candidate] + 0.5, 1.0) - 0.5
            if np.abs(gap).max() <= tol:
                unmatched.remove(candidate)
                break
        else:
            return False
    return not unmatched


def test_elementary_shears_are_the_twelve_unimodular_transvections() -> None:
    shears = elementary_shears()
    assert len(shears) == 12 and len(set(shears)) == 12
    for w in shears:
        matrix = np.asarray(w)
        assert round(np.linalg.det(matrix)) == 1
        assert np.count_nonzero(matrix - np.eye(3, dtype=int)) == 1


def test_general_position_cover_admits_every_shear_and_describes_the_same_crystal() -> None:
    scaffold = _scaffold(
        "P1",
        {6: 1, 8: 1},
        ("1a", 6, (Fraction(1, 7), Fraction(2, 7), Fraction(3, 7))),
        ("1a", 8, (Fraction(5, 11), Fraction(3, 11), Fraction(9, 11))),
    )
    layout = IntrinsicCoverLayout(ContinuousCover.from_exact_scaffold(scaffold))
    augmentation = cover_basis_augmentation(layout)
    assert isinstance(augmentation, CoverBasisAugmentation) and len(augmentation) == 12
    assert cover_basis_augmentation(layout) is augmentation  # cached per cover
    z = np.array([0.3, -0.2, 0.1, 0.25, -0.15, 0.05])
    eta = layout.encode_exact_parameters(scaffold)
    original = layout.decode(ContinuousGeometry(z, eta))
    for index, change in enumerate(augmentation.changes):
        moved = augmentation.transform(index, z, eta)
        assert moved is not None, index
        decoded = layout.decode(ContinuousGeometry(*moved))
        np.testing.assert_allclose(
            decoded.metric, change.move_metric(original.metric), rtol=0, atol=1e-9
        )
        # Same crystal: fractional coordinates transform by x -> x W, species kept.
        expected = _positions_by_species(original)
        actual = _positions_by_species(decoded)
        for number, rows in expected.items():
            image = np.remainder(rows @ change.matrix, 1.0)
            assert _same_point_set(image, actual[number]), (index, number)
        # Same geometry: the Cartesian pair vectors agree up to the lattice.
        cart_original = original.fractional_positions @ original.lattice
        cart_moved = decoded.fractional_positions @ decoded.lattice
        assert cart_original.shape == cart_moved.shape
        assert abs(np.linalg.det(original.lattice) - np.linalg.det(decoded.lattice)) < 1e-9


def test_fixed_sites_restrict_the_admissible_shears() -> None:
    scaffold = _scaffold("P-1", {6: 1, 8: 1}, ("1a", 6, ()), ("1b", 8, ()))
    layout = IntrinsicCoverLayout(ContinuousCover.from_exact_scaffold(scaffold))
    augmentation = cover_basis_augmentation(layout)
    # 1b = (0, 0, 1/2) is fixed by x -> x W iff row 2 of W is (0, 0, 1): the
    # four shears I +- E_20, I +- E_21 move it onto another fixed site.
    point = np.asarray([float(v) for v in layout.setting.wyckoff("1b").representative_offset])
    assert point.tolist() == [0.0, 0.0, 0.5]
    assert len(augmentation) == 8
    for change in augmentation.changes:
        assert change.matrix[2].tolist() == [0, 0, 1]
        np.testing.assert_array_equal(np.remainder(point @ change.matrix, 1.0), point)
    admitted = {tuple(map(tuple, change.matrix.tolist())) for change in augmentation.changes}
    for w in elementary_shears():
        moved = np.remainder(point @ np.asarray(w), 1.0)
        assert (tuple(map(tuple, w)) in admitted) == bool(np.array_equal(moved, point))
    setting = layout.setting
    assert all(
        normalizes(setting, AffineOperation(w, (Fraction(0),) * 3)) for w in elementary_shears()
    )


def test_cubic_settings_admit_no_shear() -> None:
    setting = build_space_group_setting(221)
    assert not any(
        normalizes(setting, AffineOperation(w, (Fraction(0),) * 3)) for w in elementary_shears()
    )


def test_objective_option_is_opt_in_and_moves_the_endpoint() -> None:
    torch = pytest.importorskip("torch")
    from test_learned import _tiny_config

    from crystalatlas.catalogue import MINIMAL_CATALOGUE
    from crystalatlas.errors import DomainError
    from crystalatlas.learned import CrystalAtlasModel, LearnedCrystalGenerator
    from crystalatlas.training_objectives import CFMDequantization

    torch.manual_seed(2)
    model = CrystalAtlasModel(_tiny_config(), catalogue=MINIMAL_CATALOGUE).double()
    generator = LearnedCrystalGenerator(model, allow_untrained=True)
    scaffold = _scaffold(
        "P1",
        {6: 1, 8: 1},
        ("1a", 6, (Fraction(1, 7), Fraction(2, 7), Fraction(3, 7))),
        ("1a", 8, (Fraction(5, 11), Fraction(3, 11), Fraction(9, 11))),
    )
    layout = model.layout(ContinuousCover.from_exact_scaffold(scaffold))
    metric = layout.chart.decode_latent(np.full(layout.lattice_dimension, 0.1))

    def run(**kwargs):  # type: ignore[no-untyped-def]
        return generator.geometry_cfm_objective(
            scaffold,
            metric,
            time=torch.tensor(0.4, dtype=torch.float64),
            sigma_z=0.05,
            sigma_eta=0.03,
            generator=torch.Generator().manual_seed(7),
            **kwargs,
        )

    legacy = run()
    none = run(basis_augmentation="none")
    assert torch.equal(none.loss.total, legacy.loss.total)
    assert torch.equal(none.path.state.eta, legacy.path.state.eta)
    moved = run(basis_augmentation="shear-v1", basis_augmentation_probability=1.0)
    assert torch.isfinite(moved.loss.total)
    assert not torch.equal(moved.path.target_velocity.z, legacy.path.target_velocity.z)
    for kwargs in (
        {"basis_augmentation": "rotate-v9"},
        {"basis_augmentation": "shear-v1", "basis_augmentation_probability": 1.5},
    ):
        with pytest.raises(DomainError):
            run(**kwargs)
    assert "basis_augmentation" not in CFMDequantization().to_dict()
    payload = CFMDequantization(
        basis_augmentation="shear-v1", basis_augmentation_probability=0.25
    ).to_dict()
    assert payload["basis_augmentation"] == "shear-v1"
    assert payload["basis_augmentation_probability_hex"] == (0.25).hex()
    with pytest.raises(DomainError):
        CFMDequantization(basis_augmentation="shear-v1", basis_augmentation_probability=-0.1)
