"""Batched mixed-cover bridge expansion for the padded objective (review 2026-10-29, H2).

``geometry_cfm_objective_padded`` expanded every record with its own
``TorchCoverGeometry`` call: a Python loop over the record's orbits and coset
operations, hundreds of small tensor operations per record.  ``expand_padded``
stacks each bridge's exact per-atom expansion constants ``A[n] = M_o C_o R_j``
and ``c[n] = o_o R_j + t_j`` (so ``x_n = eta A[n] + c[n]`` mod 1 and
``J_n = L^T A[n]^T``) into padded tensors and expands the whole batch in two
``einsum`` calls.  The values equal the per-record expansion to rounding.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from test_padded_batching import (  # noqa: E402
    _mixed_scaffolds,
    _reference_metrics,
    mixed_model,  # noqa: F401 - pytest fixture
)

from crystalatlas.geometry import ContinuousCover  # noqa: E402
from crystalatlas.learned import LearnedCrystalGenerator  # noqa: E402
from crystalatlas.nn.bridge import TorchCoverGeometry, expand_padded  # noqa: E402

F64 = torch.float64


def _torus_close(actual: torch.Tensor, expected: torch.Tensor, tol: float = 1e-12) -> bool:
    gap = torch.remainder(actual - expected + 0.5, 1.0) - 0.5
    return bool(gap.abs().max() <= tol)


def test_padded_expansion_equals_per_record_expansion(mixed_model) -> None:  # type: ignore[no-untyped-def]  # noqa: F811,E501
    layouts = [
        mixed_model.layout(ContinuousCover.from_exact_scaffold(s)) for s in _mixed_scaffolds()
    ]
    bridges = [TorchCoverGeometry(layout).double() for layout in layouts]
    generator = torch.Generator().manual_seed(9)
    z_values, eta_values = [], []
    for layout in layouts:
        z_values.append(
            (0.3 * torch.randn(layout.lattice_dimension, generator=generator, dtype=F64))
            .requires_grad_()
        )
        eta_values.append(
            (2.0 * torch.randn(layout.intrinsic_dimension, generator=generator, dtype=F64))
            .requires_grad_()
        )
    padded = expand_padded(bridges, z_values, eta_values)
    atoms = [layout.expanded_atom_count for layout in layouts]
    assert padded.fractional_coordinates.shape == (4, max(atoms), 3)
    assert padded.jacobian_cartesian.shape[:2] == (4, max(atoms))
    assert padded.atom_mask.sum(dim=1).tolist() == atoms
    for b, (bridge, z, eta) in enumerate(zip(bridges, z_values, eta_values, strict=True)):
        single = bridge(z, eta)
        n, k = atoms[b], layouts[b].intrinsic_dimension
        assert _torus_close(padded.fractional_coordinates[b, :n], single.fractional_coordinates)
        # Padded slots repeat atom 0, as the loop did, so masked attention sees finite data.
        assert torch.equal(
            padded.fractional_coordinates[b, n:],
            padded.fractional_coordinates[b, :1].expand(max(atoms) - n, 3),
        )
        jacobian = padded.jacobian_cartesian[b]
        torch.testing.assert_close(
            jacobian[:n, :, :k], single.jacobian_cartesian, rtol=0, atol=1e-12
        )
        assert torch.equal(jacobian[n:], torch.zeros_like(jacobian[n:]))
        assert torch.equal(jacobian[:, :, k:], torch.zeros_like(jacobian[:, :, k:]))
        torch.testing.assert_close(padded.lattice[b], single.lattice, rtol=0, atol=1e-12)
        torch.testing.assert_close(
            padded.bounded_lattice_coordinates[b],
            single.bounded_lattice_coordinates,
            rtol=0,
            atol=0,
        )
        assert torch.equal(padded.atomic_numbers[b, :n], single.atomic_numbers)
        assert torch.equal(padded.atom_features[b, :n], single.atom_features)
        torch.testing.assert_close(
            padded.lattice_commutant_basis[b], single.lattice_commutant_basis, rtol=0, atol=0
        )
    # Gradients flow through both blocks of every record.
    scalar = padded.jacobian_cartesian.square().sum()
    (scalar + torch.sin(padded.fractional_coordinates).sum()).backward()
    assert all(z.grad is not None and bool(torch.isfinite(z.grad).all()) for z in z_values)
    assert all(
        eta.grad is not None and bool(torch.isfinite(eta.grad).all())
        for eta in eta_values
        if eta.numel()
    )


def test_padded_objective_still_equals_the_sequential_objective(mixed_model) -> None:  # type: ignore[no-untyped-def]  # noqa: F811,E501
    generator = LearnedCrystalGenerator(mixed_model, allow_untrained=True)
    scaffolds = _mixed_scaffolds() * 2
    metrics = _reference_metrics(mixed_model, scaffolds)
    times = torch.linspace(0.1, 0.9, len(scaffolds), dtype=F64)
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
        torch.testing.assert_close(
            padded.loss.per_sample[index], single.loss.total, rtol=1e-10, atol=1e-10
        )
