"""Analytic mass-balance checks, independent of learned energies or datasets."""

import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from crystalatlas.errors import DomainError
from crystalatlas.evaluation import PhaseReference, energy_above_hull, hull_energy


@pytest.mark.parametrize(
    "weights,energy,message",
    [
        ([0.5], -1.0, "phase weights"),
        ([float("nan"), 0.5], -1.0, "phase weights"),
        ([-0.1, 1.1], -1.0, "phase weights"),
        ([0.4, 0.6], -1.2, "mass balance"),
        ([0.5, 0.50000001], -1.0, "mass balance"),
        ([0.5, 0.5], float("nan"), "objective"),
        ([0.5, 0.5], float("inf"), "objective"),
        ([0.5, 0.5], -2.0, "objective"),
    ],
)
def test_solver_success_does_not_bypass_solution_checks(weights, energy, message):
    references = (PhaseReference("A", (1.0, 0.0), 0.0), PhaseReference("B", (0.0, 1.0), -2.0))
    solver = SimpleNamespace(
        linprog=lambda *a, **kw: SimpleNamespace(success=True, x=weights, fun=energy)
    )
    with patch.dict(sys.modules, {"scipy.optimize": solver}):
        with pytest.raises(DomainError, match=message):
            hull_energy((0.5, 0.5), references)


def test_consistent_solver_weights_reconstruct_energy():
    references = (PhaseReference("A", (1.0, 0.0), 0.0), PhaseReference("B", (0.0, 1.0), -2.0))
    solver = SimpleNamespace(
        linprog=lambda *a, **kw: SimpleNamespace(success=True, x=[0.5, 0.5], fun=-1.0)
    )
    with patch.dict(sys.modules, {"scipy.optimize": solver}):
        assert hull_energy((0.5, 0.5), references) == -1.0


@pytest.mark.parametrize(
    "query",
    [
        (0.5, 0.50000005),
        (0.5, 0.49999995),
        (float("nan"), 1.0),
        (float("inf"), 0.0),
        (-0.1, 1.1),
    ],
)
def test_invalid_simplex_rejected_before_solver(query):
    references = (PhaseReference("A", (1.0, 0.0), 0.0),)

    # Poison the optional solver module: input validation must precede imports.
    class ForbiddenSolver:
        def __getattr__(self, name):
            raise AssertionError("invalid composition reached the solver")

    with patch.dict(sys.modules, {"scipy.optimize": ForbiddenSolver()}):
        with pytest.raises(DomainError, match="query fractions"):
            hull_energy(query, references)


@pytest.mark.parametrize("fallback", [False, True])
def test_ternary_mass_balance_and_energy_reference_shift(fallback):
    if not fallback:
        pytest.importorskip("scipy.optimize")
    references = (
        PhaseReference("A", (1.0, 0.0, 0.0), 0.0),
        PhaseReference("B", (0.0, 1.0, 0.0), 0.0),
        PhaseReference("C", (0.0, 0.0, 1.0), 0.0),
        PhaseReference("AB", (0.5, 0.5, 0.0), -1.0),
    )
    query = (0.25, 0.25, 0.5)
    # Half AB + half C gives -0.5; a lower value would require more AB
    # than the A/B mass balance allows.
    shifts = (1.25, -0.75, 2.0)
    offset = sum(x * y for x, y in zip(query, shifts, strict=True))
    shifted = tuple(
        PhaseReference(
            r.identifier,
            r.fractions,
            r.enthalpy_per_atom + sum(x * y for x, y in zip(r.fractions, shifts, strict=True)),
        )
        for r in references
    )
    with patch.dict(sys.modules, {"scipy.optimize": None} if fallback else {}):
        assert hull_energy(query, references) == pytest.approx(-0.5, abs=1e-12)
        assert hull_energy(query, shifted) == pytest.approx(-0.5 + offset, abs=1e-12)
        # A discovery below the reference hull must remain negative, not clipped.
        assert energy_above_hull(query, -0.6, references) == pytest.approx(-0.1)
        assert energy_above_hull(query, -0.6 + offset, shifted) == pytest.approx(-0.1)
        with pytest.raises(DomainError):
            hull_energy(query, references[:2])


@pytest.mark.parametrize("value", [True, 0, -1, 1.5, float("inf")])
@pytest.mark.parametrize("parameter", ["exhaustive_limit", "exhaustive_subset_limit"])
def test_fallback_limits_require_positive_integers(value, parameter):
    reference = PhaseReference("A", (1.0,), 0.0)
    with pytest.raises(DomainError, match="positive integers"):
        hull_energy((1.0,), (reference,), **{parameter: value})


def test_multielement_fallback_refuses_before_enumeration():
    references = tuple(PhaseReference(str(i), (0.125,) * 8, -1.0) for i in range(32))
    with patch.dict(sys.modules, {"scipy.optimize": None}):
        with patch("crystalatlas.evaluation._small_hull_energy") as enumerate_subsets:
            with pytest.raises(DomainError, match="subset budget"):
                hull_energy((0.125,) * 8, references)
            enumerate_subsets.assert_not_called()


def test_subset_budget_exact_boundary():
    references = (PhaseReference("A", (1.0, 0.0), 0.0), PhaseReference("B", (0.0, 1.0), 0.0))
    # Two singleton subsets plus the pair: exactly three candidates.
    with patch.dict(sys.modules, {"scipy.optimize": None}):
        with pytest.raises(DomainError, match="subset budget"):
            hull_energy((0.5, 0.5), references, exhaustive_subset_limit=2)
        assert hull_energy((0.5, 0.5), references, exhaustive_subset_limit=3) == pytest.approx(
            0.0
        )


def test_fallback_does_not_hide_nonfinite_subset_solution():
    references = (PhaseReference("A", (1.0,), -1.0),)
    with patch.dict(sys.modules, {"scipy.optimize": None}):
        with patch("numpy.linalg.lstsq", return_value=(np.array([np.nan]),)):
            with pytest.raises(DomainError, match="nonfinite phase weights"):
                hull_energy((1.0,), references)


@pytest.mark.parametrize("elements", [2, 3, 4])
def test_seeded_small_hulls_agree_between_solvers(elements):
    pytest.importorskip("scipy.optimize")
    rng = np.random.default_rng(402 + elements)
    compositions = np.vstack([np.eye(elements), rng.dirichlet(np.ones(elements), size=4)])
    references = tuple(
        PhaseReference(str(i), tuple(c), float(rng.uniform(-2, 1)))
        for i, c in enumerate(compositions)
    )
    for query in rng.dirichlet(np.ones(elements), size=4):
        expected = hull_energy(query, references)
        with patch.dict(sys.modules, {"scipy.optimize": None}):
            actual = hull_energy(query, references)
        assert actual == pytest.approx(expected, abs=1e-10, rel=0)
