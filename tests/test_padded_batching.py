"""Mixed-topology padded batching is exact (review finding M5).

A padded batch must reproduce independent per-crystal evaluations: padded
atoms, lattice slots, and intrinsic coordinates may contribute only exact
zeros.  Agreement is to floating-point rounding rather than bitwise, because
inserting zeros can change the association of a reduction.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from crystalatlas.nn.invariant_transformer import (  # noqa: E402
    InvariantPeriodicTransformer,
    InvariantPeriodicTransformerConfig,
)

pytestmark = pytest.mark.ml

F64 = torch.float64
ROUNDING = 1.0e-12


def _transformer(**overrides) -> InvariantPeriodicTransformer:  # type: ignore[no-untyped-def]
    torch.manual_seed(5)
    arguments = dict(
        input_scalar_dim=5,
        model_dim=12,
        pair_dim=6,
        num_heads=2,
        num_layers=2,
        output_vector_dim=3,
    )
    arguments.update(overrides)
    return InvariantPeriodicTransformer(
        InvariantPeriodicTransformerConfig(**arguments)
    ).double()


def _crystals():  # type: ignore[no-untyped-def]
    generator = torch.Generator().manual_seed(11)
    crystals = []
    for atoms, lattice in (
        (2, [[3.1, 0.0, 0.0], [0.2, 2.9, 0.0], [0.1, -0.3, 3.4]]),
        (5, [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 6.5]]),
        (3, [[2.5, 0.0, 0.0], [-1.25, 2.165, 0.0], [0.0, 0.0, 4.1]]),
    ):
        crystals.append(
            (
                torch.randn(atoms, 5, generator=generator, dtype=F64),
                torch.rand(atoms, 3, generator=generator, dtype=F64),
                torch.tensor(lattice, dtype=F64),
            )
        )
    return crystals


def _pad(crystals):  # type: ignore[no-untyped-def]
    width = max(len(scalars) for scalars, _, _ in crystals)
    batch = len(crystals)
    scalars = torch.zeros(batch, width, 5, dtype=F64)
    # Padded atoms sit on a real atom: a worst case for collision features.
    fractional = torch.zeros(batch, width, 3, dtype=F64)
    mask = torch.zeros(batch, width, dtype=torch.bool)
    for index, (values, coordinates, _) in enumerate(crystals):
        count = len(values)
        scalars[index, :count] = values
        fractional[index, :count] = coordinates
        fractional[index, count:] = coordinates[0]
        scalars[index, count:] = 7.0  # arbitrary finite padding must not leak
        mask[index, :count] = True
    lattice = torch.stack([lattice for _, _, lattice in crystals])
    return scalars, fractional, lattice, mask


@pytest.mark.parametrize("radial", [(), (0.6, 1.2)])
def test_masked_dense_backbone_equals_independent_calls(radial) -> None:  # type: ignore[no-untyped-def]
    transformer = _transformer(radial_gaussian_widths=radial).eval()
    crystals = _crystals()
    scalars, fractional, lattice, mask = _pad(crystals)
    padded = transformer(scalars, fractional, lattice, atom_mask=mask)
    for index, (values, coordinates, cell) in enumerate(crystals):
        alone = transformer(values, coordinates, cell)
        count = len(values)
        for name in ("scalar_channels", "vector_channels"):
            expected = getattr(alone, name)
            actual = getattr(padded, name)[index, :count]
            assert torch.allclose(actual, expected, rtol=ROUNDING, atol=ROUNDING), name
            assert bool((getattr(padded, name)[index, count:] == 0.0).all()), name
        assert torch.allclose(
            padded.pooled_scalars[index], alone.pooled_scalars, rtol=ROUNDING, atol=ROUNDING
        )


def test_a_full_mask_is_bit_identical_to_the_unmasked_backbone() -> None:
    transformer = _transformer().eval()
    values, coordinates, cell = _crystals()[1]
    plain = transformer(values[None], coordinates[None], cell[None])
    masked = transformer(
        values[None],
        coordinates[None],
        cell[None],
        atom_mask=torch.ones(1, len(values), dtype=torch.bool),
    )
    for name in ("scalar_channels", "vector_channels", "pooled_scalars"):
        assert torch.equal(getattr(plain, name), getattr(masked, name)), name


def test_atom_mask_is_validated() -> None:
    transformer = _transformer()
    scalars, fractional, lattice, mask = _pad(_crystals())
    with pytest.raises(ValueError, match="atom_mask"):
        transformer(scalars, fractional, lattice, atom_mask=mask.to(F64))
    with pytest.raises(ValueError, match="atom_mask"):
        transformer(scalars, fractional, lattice, atom_mask=mask[:, :-1])
    empty = mask.clone()
    empty[0] = False
    with pytest.raises(ValueError, match="at least one"):
        transformer(scalars, fractional, lattice, atom_mask=empty)


# --- field level -----------------------------------------------------------

from crystalatlas.nn.field import (  # noqa: E402
    IntrinsicFieldConfig,
    IntrinsicGeometryField,
    TangentRankError,
)

_SHAPES = (  # atoms, lattice dimension, intrinsic dimension, species
    (2, 1, 1, 1),
    (5, 4, 3, 2),
    (3, 2, 0, 2),
)


def _field(head: str) -> IntrinsicGeometryField:
    torch.manual_seed(7)
    config = IntrinsicFieldConfig(
        atom_feature_dim=3,
        composition_dim=6,
        species_embedding_dim=4,
        max_lattice_dim=6,
        num_time_frequencies=2,
        scalar_dim=12,
        vector_dim=6,
        num_heads=2,
        num_layers=2,
        lattice_head=head,
    )
    return IntrinsicGeometryField(config).double().eval()


def _field_crystals(rank_deficient: int | None = None):  # type: ignore[no-untyped-def]
    generator = torch.Generator().manual_seed(13)
    crystals = []
    for index, (atoms, lattice_dim, intrinsic_dim, species) in enumerate(_SHAPES):
        z = torch.randn(lattice_dim, generator=generator, dtype=F64)
        jacobian = torch.randn(atoms, 3, intrinsic_dim, generator=generator, dtype=F64)
        if index == rank_deficient and intrinsic_dim > 1:
            jacobian[..., 1] = 2.0 * jacobian[..., 0]
        crystals.append(
            dict(
                time=torch.tensor(0.1 + 0.3 * index, dtype=F64),
                z=z,
                eta=torch.rand(intrinsic_dim, generator=generator, dtype=F64),
                composition_atomic_numbers=torch.tensor([8, 26][:species], dtype=torch.long),
                composition_counts=torch.tensor([3.0, 2.0][:species], dtype=F64),
                atomic_numbers=torch.tensor([8, 26, 8, 26, 8][:atoms], dtype=torch.long),
                atom_features=torch.randn(atoms, 3, generator=generator, dtype=F64),
                fractional_coordinates=torch.rand(atoms, 3, generator=generator, dtype=F64),
                lattice=torch.diag(torch.tensor([3.0, 3.5, 4.0], dtype=F64))
                + 0.2 * torch.rand(3, 3, generator=generator, dtype=F64).triu(1),
                bounded_lattice_coordinates=torch.tanh(z),
                jacobian_cartesian=jacobian,
                lattice_commutant_basis=torch.linalg.qr(
                    torch.randn(6, lattice_dim, generator=generator, dtype=F64)
                ).Q,
            )
        )
    return crystals


def _pad_field_inputs(crystals):  # type: ignore[no-untyped-def]
    batch = len(crystals)
    atoms = max(len(c["atomic_numbers"]) for c in crystals)
    lattice_dim = max(len(c["z"]) for c in crystals)
    intrinsic_dim = max(len(c["eta"]) for c in crystals)
    species = max(len(c["composition_counts"]) for c in crystals)
    padded = dict(
        time=torch.stack([c["time"] for c in crystals]),
        z=torch.zeros(batch, lattice_dim, dtype=F64),
        eta=torch.zeros(batch, intrinsic_dim, dtype=F64),
        composition_atomic_numbers=torch.zeros(batch, species, dtype=torch.long),
        composition_counts=torch.zeros(batch, species, dtype=F64),
        atomic_numbers=torch.zeros(batch, atoms, dtype=torch.long),
        atom_mask=torch.zeros(batch, atoms, dtype=torch.bool),
        atom_features=torch.zeros(batch, atoms, 3, dtype=F64),
        fractional_coordinates=torch.zeros(batch, atoms, 3, dtype=F64),
        lattice=torch.stack([c["lattice"] for c in crystals]),
        bounded_lattice_coordinates=torch.zeros(batch, lattice_dim, dtype=F64),
        jacobian_cartesian=torch.zeros(batch, atoms, 3, intrinsic_dim, dtype=F64),
        lattice_commutant_basis=torch.zeros(batch, 6, lattice_dim, dtype=F64),
        lattice_dimensions=torch.tensor([len(c["z"]) for c in crystals]),
        intrinsic_dimensions=torch.tensor([len(c["eta"]) for c in crystals]),
    )
    for b, c in enumerate(crystals):
        n, d, e, k = (
            len(c["atomic_numbers"]),
            len(c["z"]),
            len(c["eta"]),
            len(c["composition_counts"]),
        )
        padded["z"][b, :d] = c["z"]
        padded["eta"][b, :e] = c["eta"]
        padded["composition_atomic_numbers"][b, :k] = c["composition_atomic_numbers"]
        padded["composition_counts"][b, :k] = c["composition_counts"]
        padded["atomic_numbers"][b, :n] = c["atomic_numbers"]
        padded["atom_mask"][b, :n] = True
        padded["atom_features"][b, :n] = c["atom_features"]
        padded["fractional_coordinates"][b, :n] = c["fractional_coordinates"]
        padded["fractional_coordinates"][b, n:] = c["fractional_coordinates"][0]
        padded["bounded_lattice_coordinates"][b, :d] = c["bounded_lattice_coordinates"]
        padded["jacobian_cartesian"][b, :n, :, :e] = c["jacobian_cartesian"]
        padded["lattice_commutant_basis"][b, :, :d] = c["lattice_commutant_basis"]
    return padded


def _call_alone(field, crystal):  # type: ignore[no-untyped-def]
    arguments = dict(crystal)
    time, z, eta = arguments.pop("time"), arguments.pop("z"), arguments.pop("eta")
    if field.config.lattice_head != "commutant-v2":
        arguments.pop("lattice_commutant_basis")
    return field(time, z, eta, None, **arguments)


def _call_padded(field, padded):  # type: ignore[no-untyped-def]
    arguments = dict(padded)
    time, z, eta = arguments.pop("time"), arguments.pop("z"), arguments.pop("eta")
    if field.config.lattice_head != "commutant-v2":
        arguments.pop("lattice_commutant_basis")
    return field.forward_padded(time, z, eta, None, **arguments)


def _close(actual, expected):  # type: ignore[no-untyped-def]
    return torch.allclose(actual, expected, rtol=ROUNDING, atol=ROUNDING)


@pytest.mark.parametrize("head", ["truncated-v1", "commutant-v2"])
def test_padded_field_equals_independent_unbatched_calls(head: str) -> None:
    field = _field(head)
    crystals = _field_crystals()
    output = _call_padded(field, _pad_field_inputs(crystals))
    for b, crystal in enumerate(crystals):
        alone = _call_alone(field, crystal)
        n, d, e = len(crystal["atomic_numbers"]), len(crystal["z"]), len(crystal["eta"])
        assert _close(output.lattice_velocity[b, :d], alone.lattice_velocity)
        assert bool((output.lattice_velocity[b, d:] == 0.0).all())
        assert _close(output.intrinsic_velocity[b, :e], alone.intrinsic_velocity)
        assert bool((output.intrinsic_velocity[b, e:] == 0.0).all())
        assert _close(output.cartesian_vectors[b, :n], alone.cartesian_vectors)
        assert bool((output.cartesian_vectors[b, n:] == 0.0).all())
        assert _close(output.coordinate_metric[b, :e, :e], alone.coordinate_metric)
        identity = torch.eye(output.coordinate_metric.shape[-1], dtype=F64)
        assert torch.equal(output.coordinate_metric[b, e:, e:], identity[e:, e:])
        assert bool((output.coordinate_metric[b, :e, e:] == 0.0).all())


def test_padded_field_keeps_the_exact_per_crystal_rank_certificate() -> None:
    """A dependent Jacobian column is refused, padded or not.

    Numerically its smallest singular value is ~1e-16 rather than zero, so
    both audits refuse it through the condition-number bound.
    """

    field = _field("truncated-v1")
    crystals = _field_crystals(rank_deficient=1)
    certificate = "rank deficient|condition-number support"
    with pytest.raises(TangentRankError, match=certificate):
        _call_alone(field, crystals[1])
    with pytest.raises(TangentRankError, match=certificate):
        _call_padded(field, _pad_field_inputs(crystals))
    for index in (0, 2):  # the other crystals are certified on their own
        _call_alone(field, crystals[index])


# --- objective level -------------------------------------------------------


@pytest.fixture(scope="module")
def mixed_model():  # type: ignore[no-untyped-def]
    from dataclasses import replace

    from test_learned import _tiny_config

    from crystalatlas.catalogue import (
        FIXTURE_EXPECTED_ROWS,
        MINIMAL_CATALOGUE,
        CertifiedCatalogue,
    )
    from crystalatlas.full_catalogue import build_space_group_setting
    from crystalatlas.learned import CrystalAtlasModel

    settings = {key: MINIMAL_CATALOGUE.get(key) for key in ("P1", "P-1")}
    rows = dict(FIXTURE_EXPECTED_ROWS)
    for number in (123, 221):  # P4/mmm, Pm-3m
        setting = build_space_group_setting(number)
        settings[setting.key] = setting
        rows[setting.key] = frozenset(setting.wyckoff_positions)
    catalogue = CertifiedCatalogue(
        settings, rows, certification_scope="P1, P-1, P4/mmm, Pm-3m test subset"
    )
    base = _tiny_config()
    config = replace(base, domain=replace(base.domain, max_raw_atoms=8))
    torch.manual_seed(31)
    return CrystalAtlasModel(config, catalogue=catalogue).double()


def _mixed_scaffolds():  # type: ignore[no-untyped-def]
    from fractions import Fraction

    from crystalatlas.decoder import CrystalScaffold, OrbitAssignment

    def scaffold(nu, group, composition, *orbits):  # type: ignore[no-untyped-def]
        return CrystalScaffold.from_mapping(
            nu=nu,
            space_group=group,
            reduced_composition=composition,
            orbits=tuple(OrbitAssignment(*orbit) for orbit in orbits),
        )

    return (
        # cubic: d_z = 1, d_eta = 1, 7 atoms
        scaffold(1, "Pm-3m", {6: 1, 8: 6}, ("1a", 6, ()), ("6e", 8, (Fraction(2, 7),))),
        # triclinic: d_z = 6, d_eta = 3, 2 atoms
        scaffold(2, "P-1", {8: 1}, ("2i", 8, (Fraction(1, 7), Fraction(2, 7), Fraction(3, 7)))),
        # tetragonal: d_z = 2, d_eta = 1, 3 atoms
        scaffold(1, "P4/mmm", {6: 1, 8: 2}, ("1a", 6, ()), ("2g", 8, (Fraction(3, 10),))),
        # cubic without free parameters: d_eta = 0, 4 atoms
        scaffold(1, "Pm-3m", {14: 1, 8: 3}, ("1a", 14, ()), ("3c", 8, ())),
    )


def _reference_metrics(model, scaffolds):  # type: ignore[no-untyped-def]
    import numpy as np

    from crystalatlas.geometry import ContinuousCover

    metrics = []
    for index, scaffold in enumerate(scaffolds):
        layout = model.layout(ContinuousCover.from_exact_scaffold(scaffold))
        latent = np.full(layout.lattice_dimension, 0.03 * (index + 1))
        metrics.append(layout.chart.decode_latent(latent))
    return tuple(metrics)


def test_padded_objective_equals_sequential_single_crystal_objectives(mixed_model) -> None:  # type: ignore[no-untyped-def]
    from crystalatlas.learned import LearnedCrystalGenerator

    generator = LearnedCrystalGenerator(mixed_model, allow_untrained=True)
    scaffolds = _mixed_scaffolds()
    metrics = _reference_metrics(mixed_model, scaffolds)
    times = torch.tensor([0.2, 0.45, 0.7, 0.9], dtype=F64)
    mixed_model.eval()
    padded = generator.geometry_cfm_objective_padded(
        scaffolds,
        metrics,
        time=times,
        sigma_z=0.05,
        sigma_eta=0.03,
        generator=torch.Generator().manual_seed(17),
    )
    sequential = torch.Generator().manual_seed(17)
    for index, (scaffold, metric) in enumerate(zip(scaffolds, metrics, strict=True)):
        single = generator.geometry_cfm_objective(
            scaffold,
            metric,
            time=times[index],
            sigma_z=0.05,
            sigma_eta=0.03,
            generator=sequential,
        )
        assert _close(padded.loss.per_sample[index], single.loss.total), index
        dimension = single.path.target_velocity.eta.numel()
        assert _close(padded.prediction.eta[index, :dimension], single.prediction.eta)
        assert dimension == int(padded.intrinsic_dimensions[index])
    assert padded.intrinsic_dimensions.tolist() == [1, 3, 1, 0]
    assert padded.lattice_dimensions.tolist() == [1, 6, 2, 1]
    padded.loss.total.backward()
    gradients = [p.grad for p in mixed_model.geometry_field.parameters() if p.requires_grad]
    assert any(g is not None and bool(g.abs().sum() > 0) for g in gradients)
