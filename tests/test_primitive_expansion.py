"""Opt-in primitive-cell field expansion (review 2026-10-29, M3).

The dense field is quadratic in expanded atoms, and the conventional cell of an
F-centred crystal holds four copies of every primitive atom.  Under
``geometry_expansion_cell="primitive-v2"`` the bridge keeps one coset
representative per class modulo the centring translations and hands the field
the primitive lattice ``P L`` with coordinates ``x P^-1``: the same geometry
with ``N / I_c`` atoms.  Cartesian positions and Jacobians agree atom by atom,
the conventional translation-quotient metric is ``I_c`` times the primitive
one (so the per-atom loss is identical), and the Cartesian kernel keeps the
conventional metric.  The chart and the decoder are unchanged.
"""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from test_learned import _tiny_config  # noqa: E402

from crystalatlas.catalogue import (  # noqa: E402
    FIXTURE_EXPECTED_ROWS,
    MINIMAL_CATALOGUE,
    CertifiedCatalogue,
)
from crystalatlas.decoder import CrystalScaffold, OrbitAssignment  # noqa: E402
from crystalatlas.errors import DomainError  # noqa: E402
from crystalatlas.full_catalogue import build_space_group_setting  # noqa: E402
from crystalatlas.geometry import ContinuousCover, IntrinsicCoverLayout  # noqa: E402
from crystalatlas.nn.bridge import (  # noqa: E402
    TorchCoverGeometry,
    expand_padded,
    intrinsic_noise_covariance,
)
from crystalatlas.nn.field import translation_quotient_normal_matrix  # noqa: E402

F64 = torch.float64


def _catalogue() -> CertifiedCatalogue:
    settings = {key: MINIMAL_CATALOGUE.get(key) for key in ("P1", "P-1")}
    rows = dict(FIXTURE_EXPECTED_ROWS)
    for number in (12, 166, 225, 229):  # C2/m, R-3m, Fm-3m, Im-3m
        setting = build_space_group_setting(number)
        settings[setting.key] = setting
        rows[setting.key] = frozenset(setting.wyckoff_positions)
    return CertifiedCatalogue(settings, rows, certification_scope="M3 test subset")


def _scaffold(nu, group, composition, *orbits):  # type: ignore[no-untyped-def]
    return CrystalScaffold.from_mapping(
        nu=nu,
        space_group=group,
        reduced_composition=composition,
        orbits=tuple(OrbitAssignment(*orbit) for orbit in orbits),
    )


def _scaffolds():  # type: ignore[no-untyped-def]
    return (
        # rock salt: 8 conventional atoms, 2 primitive
        _scaffold(4, "Fm-3m", {6: 1, 8: 1}, ("4a", 6, ()), ("4b", 8, ())),
        # body-centred: 2 conventional atoms, 1 primitive
        _scaffold(2, "Im-3m", {14: 1}, ("2a", 14, ())),
        # rhombohedral (hexagonal setting): 6 conventional atoms, 2 primitive
        _scaffold(6, "R-3m", {14: 1}, ("6c", 14, (Fraction(2, 9),))),
        # base-centred monoclinic: 8 conventional atoms, 4 primitive
        _scaffold(
            4,
            "C2/m",
            {6: 1, 8: 1},
            ("4i", 6, (Fraction(1, 7), Fraction(3, 7))),
            ("4i", 8, (Fraction(1, 3), Fraction(5, 9))),
        ),
    )


def _match(conventional, primitive, inverse: np.ndarray, atol: float = 1e-10) -> list[int]:
    """Primitive atom carrying each conventional atom, modulo the primitive lattice."""

    matches = []
    moved = np.remainder(conventional @ inverse, 1.0)
    for row in moved:
        gaps = np.remainder(primitive - row + 0.5, 1.0) - 0.5
        hits = np.flatnonzero(np.abs(gaps).max(axis=1) <= atol)
        assert len(hits) == 1, (row, hits)
        matches.append(int(hits[0]))
    return matches


def _torus_gap(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float((torch.remainder(actual - expected + 0.5, 1.0) - 0.5).abs().max())


@pytest.mark.parametrize("index", range(4))
def test_primitive_expansion_is_the_same_geometry_with_fewer_atoms(index: int) -> None:
    scaffold = _scaffolds()[index]
    cover = ContinuousCover.from_exact_scaffold(scaffold)
    layout = IntrinsicCoverLayout(cover, catalogue=_catalogue())
    conventional = TorchCoverGeometry(layout).double()
    primitive = TorchCoverGeometry(layout, expansion_cell="primitive-v2").double()
    copies = layout.setting.centering_index
    assert copies > 1
    assert primitive.metric_copies == copies and conventional.metric_copies == 1
    assert primitive.atom_count * copies == conventional.atom_count
    assert conventional.atom_count == layout.expanded_atom_count
    generator = torch.Generator().manual_seed(index)
    z = 0.2 * torch.randn(layout.lattice_dimension, generator=generator, dtype=F64)
    eta = torch.rand(layout.intrinsic_dimension, generator=generator, dtype=F64)
    full = conventional(z, eta)
    reduced = primitive(z, eta)
    basis = primitive.primitive_basis(dtype=F64, device=z.device)
    torch.testing.assert_close(reduced.lattice, basis @ full.lattice, rtol=0, atol=1e-12)
    torch.testing.assert_close(reduced.metric, full.metric, rtol=0, atol=0)
    volume_full = torch.det(full.lattice).item()
    assert abs(torch.det(reduced.lattice).item() * copies - volume_full) < 1e-9
    inverse = np.linalg.inv(basis.numpy())
    matches = _match(
        full.fractional_coordinates.numpy(), reduced.fractional_coordinates.numpy(), inverse
    )
    assert sorted(set(matches)) == list(range(primitive.atom_count))
    assert all(matches.count(i) == copies for i in range(primitive.atom_count))
    for conventional_index, primitive_index in enumerate(matches):
        assert (
            full.atomic_numbers[conventional_index] == reduced.atomic_numbers[primitive_index]
        )
        torch.testing.assert_close(
            full.jacobian_cartesian[conventional_index],
            reduced.jacobian_cartesian[primitive_index],
            rtol=0,
            atol=1e-12,
        )
        torch.testing.assert_close(
            full.atom_features[conventional_index], reduced.atom_features[primitive_index]
        )
    if layout.intrinsic_dimension:
        full_metric = translation_quotient_normal_matrix(full.jacobian_cartesian)
        reduced_metric = translation_quotient_normal_matrix(reduced.jacobian_cartesian)
        torch.testing.assert_close(full_metric, copies * reduced_metric, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(
            intrinsic_noise_covariance(conventional, z, eta, 0.1, reference_length=1.0),
            intrinsic_noise_covariance(primitive, z, eta, 0.1, reference_length=1.0),
            rtol=1e-11,
            atol=1e-14,
        )
    batched = primitive.forward_batch(z.expand(2, -1), eta.expand(2, -1))
    gap = _torus_gap(batched.fractional_coordinates[1], reduced.fractional_coordinates)
    assert gap <= 1e-12
    torch.testing.assert_close(batched.lattice[0], reduced.lattice)
    padded = expand_padded([primitive, conventional], [z, z], [eta, eta])
    n = primitive.atom_count
    torch.testing.assert_close(padded.lattice[0], reduced.lattice, rtol=0, atol=1e-12)
    torch.testing.assert_close(padded.lattice[1], full.lattice, rtol=0, atol=1e-12)
    gap = _torus_gap(padded.fractional_coordinates[0, :n], reduced.fractional_coordinates)
    assert gap <= 1e-12
    torch.testing.assert_close(
        padded.jacobian_cartesian[0, :n], reduced.jacobian_cartesian, rtol=0, atol=1e-12
    )


def test_option_is_versioned_and_restricted_to_the_dense_backbone() -> None:
    base = _tiny_config()
    assert base.model.geometry_expansion_cell == "conventional-v1"
    assert "geometry_expansion_cell" not in base.to_dict()["model"]
    primitive = replace(base, model=replace(base.model, geometry_expansion_cell="primitive-v2"))
    assert primitive.fingerprint != base.fingerprint
    assert primitive.to_dict()["model"]["geometry_expansion_cell"] == "primitive-v2"
    with pytest.raises(DomainError, match="conventional-v1 or primitive-v2"):
        replace(base.model, geometry_expansion_cell="reduced-v9")
    with pytest.raises(DomainError, match="invariant_global"):
        replace(
            base.model,
            geometry_expansion_cell="primitive-v2",
            geometry_backbone="equivariant_local",
        )
    cover = ContinuousCover.from_exact_scaffold(_scaffolds()[0])
    layout = IntrinsicCoverLayout(cover, catalogue=_catalogue())
    with pytest.raises(DomainError, match="expansion_cell"):
        TorchCoverGeometry(layout, expansion_cell="reduced-v9")


def test_model_trains_and_samples_on_the_primitive_cell() -> None:
    from crystalatlas.learned import CrystalAtlasModel, LearnedCrystalGenerator

    base = _tiny_config()
    config = replace(
        base,
        domain=replace(base.domain, max_raw_atoms=8, max_formula_units=4),
        model=replace(base.model, geometry_expansion_cell="primitive-v2"),
    )
    torch.manual_seed(5)
    model = CrystalAtlasModel(config, catalogue=_catalogue()).double()
    generator = LearnedCrystalGenerator(model, allow_untrained=True)
    scaffold = _scaffolds()[0]  # rock salt: 8 conventional atoms, 2 primitive
    cover = ContinuousCover.from_exact_scaffold(scaffold)
    layout = model.layout(cover)
    bridge = generator._bridge(layout, torch.device("cpu"), F64)
    assert bridge.expansion_cell == "primitive-v2" and bridge.atom_count == 2
    metric = layout.chart.decode_latent(np.zeros(layout.lattice_dimension))
    result = generator.geometry_cfm_objective(
        scaffold,
        metric,
        time=torch.tensor(0.3, dtype=F64),
        sigma_z=0.05,
        sigma_eta=0.03,
        generator=torch.Generator().manual_seed(3),
    )
    assert torch.isfinite(result.loss.total)
    result.loss.total.backward()
    model.eval()
    _geometry, decoded = generator.sample_geometry_given_cover(cover, seed=11)
    assert decoded.fractional_positions.shape == (8, 3)  # decode keeps the conventional cell
    assert decoded.audit.passed
