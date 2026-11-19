"""Learned half of the F10 tiny-distribution oracle (theory F10; review M1).

The production geometry field is trained with the production CFM objective on
one fixed cover whose law the exact oracle :class:`~crystalatlas.flow_oracle.
TinyFlowOracle` evaluates in closed form.  Two threshold-free measurements are
then reported:

* the relative weighted velocity error
  ``rho_v = E||v_theta - v*||_H^2 / E||v*||_H^2`` over production path samples
  ``(t, q_t)``.  ``v*`` is the exact marginal velocity, and ``H`` is the
  production loss metric at ``q_t``: the lattice pullback metric plus the field's
  intrinsic coordinate metric divided by the atom count, exactly as in
  :func:`~crystalatlas.flow.conditional_flow_matching_loss`.  Under that
  weighting the CFM minimizer is ``v*`` (theory 2.9).  Over physically invariant
  fields it is the velocity of the law symmetrized over the cover's residual
  normalizer gauge (:class:`ResidualTranslation`).  The production objective
  does not symmetrize that gauge beyond identical-orbit relabellings and P-1
  signs.  The comparison target is therefore the gauge-symmetrized oracle, and
  ``rho_v -> 0`` exactly when the objective is learned;
* the endpoint density ``L1`` error and the relative moment errors of the learned
  ODE against the oracle's exact endpoint law, by deterministic quadrature.  A
  node ``y`` gets density ``p_0(x) |det dx/dy|``, where ``x`` is the production
  RK4 back-integral of ``y`` through the learned field.  The Jacobian comes from
  reverse-mode autograd through that integration, one backward pass per state
  dimension, because the Loewner chart's adjoint is reverse-mode only.

Pass thresholds for the learned field are a versioned theory amendment
(theory 4.10).  This module computes the values only and asserts none.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from ._optional import require_torch
from .decoder import CrystalScaffold, OrbitAssignment
from .errors import DomainError
from .flow import FlowState, FlowVelocity, integrate_rk4
from .flow_oracle import TinyFlowOracle
from .geometry import ContinuousCover, IntrinsicCoverLayout
from .learned import CrystalAtlasModel, LearnedCrystalGenerator
from .nn.bridge import TorchCoverGeometry, intrinsic_noise_covariance
from .nn.field import PackedIntrinsicFieldOutput

torch = require_torch()
Tensor = torch.Tensor
VelocityField = Callable[[Tensor, Tensor, Tensor], FlowVelocity]


@dataclass(frozen=True)
class F10Toy:
    """Records of one cover with exactly known flow law (weights uniform)."""

    scaffolds: tuple[CrystalScaffold, ...]
    metrics: tuple[np.ndarray, ...]
    sigma_z: float
    sigma_eta: float

    def __post_init__(self) -> None:
        if not self.scaffolds or len(self.scaffolds) != len(self.metrics):
            raise DomainError("an F10 toy needs aligned nonempty scaffolds and metrics")
        covers = {ContinuousCover.from_exact_scaffold(item) for item in self.scaffolds}
        if len(covers) != 1:
            raise DomainError("an F10 toy uses exactly one cover")
        if not (self.sigma_z > 0.0 and self.sigma_eta > 0.0):
            raise DomainError("dequantization scales must be positive")

    @property
    def cover(self) -> ContinuousCover:
        return ContinuousCover.from_exact_scaffold(self.scaffolds[0])


@dataclass(frozen=True)
class ResidualTranslation:
    """A proper residual Euclidean-normalizer translation of one cover.

    ``translation`` is a fractional (conventional-cell) translation and
    ``parameter_shifts`` gives, per orbit in scaffold order, the parameter shift
    it induces.  :func:`toy_oracle` verifies the claim exactly on every record.
    """

    translation: tuple[Fraction, Fraction, Fraction]
    parameter_shifts: tuple[tuple[Fraction, ...], ...]


def _translated_scaffold(
    scaffold: CrystalScaffold, gauge: ResidualTranslation, setting: object
) -> CrystalScaffold:
    if len(gauge.parameter_shifts) != len(scaffold.orbits):
        raise DomainError("a residual translation needs one parameter shift per orbit")
    orbits = []
    for orbit, shift in zip(scaffold.orbits, gauge.parameter_shifts, strict=True):
        if len(shift) != len(orbit.parameter):
            raise DomainError("parameter shift has the wrong dimension")
        moved = tuple(
            value + delta for value, delta in zip(orbit.parameter, shift, strict=True)
        )
        row = setting.wyckoff(orbit.wyckoff_label)  # type: ignore[attr-defined]
        translated = {
            tuple((c + t) % 1 for c, t in zip(point, gauge.translation, strict=True))
            for point in row.expand_exact(orbit.parameter)
        }
        image = {tuple(c % 1 for c in point) for point in row.expand_exact(moved)}
        if translated != image:
            raise DomainError(
                f"{gauge.translation} is not a residual gauge of orbit "
                f"{orbit.wyckoff_label} at parameter {orbit.parameter}"
            )
        orbits.append(OrbitAssignment(orbit.wyckoff_label, orbit.species, moved))
    return CrystalScaffold.from_mapping(
        nu=scaffold.nu,
        space_group=scaffold.space_group,
        reduced_composition=dict(scaffold.reduced_composition),
        orbits=orbits,
    )


def toy_oracle(
    model: CrystalAtlasModel,
    toy: F10Toy,
    *,
    gauge: Sequence[ResidualTranslation] = (),
    layout_gauge: bool = False,
    intrinsic_kernel: str = "fractional-v1",
) -> TinyFlowOracle:
    """The exact law that the production objective samples on ``toy``.

    ``intrinsic_kernel`` is the torus dequantization kernel of the objective being
    checked: ``fractional-v1`` (``toy.sigma_eta`` in fractional units) or
    ``cartesian-v2`` (``toy.sigma_eta`` in angstrom, one covariance per record).

    The production objective draws a residual action uniformly from
    identical-orbit relabellings (and P-1 sign flips).  This harness supports
    toys whose such group is trivial, so the oracle's action set is {I}.

    ``gauge`` lists further proper residual normalizer translations, which
    together with the identity must form a group.  Each is verified exactly on
    every record.  The returned law is then the gauge-symmetrized one: every
    record is duplicated at each gauge image, with weight divided by the group
    order.  That is exact because the Haar base and the heat kernel are
    translation-invariant.  It is the CFM target that a physically invariant
    field can represent, namely the projection of the production target onto
    gauge-invariant velocities.

    ``layout_gauge=True`` carries the cover's whole certified residual group
    instead: the normalizer-v3 (racemic) group when the model is configured
    with it, and otherwise the normalizer-v2 group.
    """

    layout = model.layout(toy.cover)
    reference_length = model.scientific_config.domain.reference_length_angstrom
    if layout_gauge:
        if gauge:
            raise DomainError("pass either a manual gauge or layout_gauge, not both")
        improper = model.scientific_config.model.geometry_residual_gauge == "normalizer-v3"
        return _layout_gauge_oracle(
            layout,
            toy,
            improper=improper,
            intrinsic_kernel=intrinsic_kernel,
            reference_length=reference_length,
        )
    if layout.residual_orbit_groups or layout.setting.key == "P-1":
        raise DomainError("the learned F10 harness supports trivial residual groups only")
    translations = {tuple(Fraction(v) % 1 for v in item.translation) for item in gauge}
    for left in translations:
        for right in translations:
            total = tuple((a + b) % 1 for a, b in zip(left, right, strict=True))
            if any(total) and total not in translations:
                raise DomainError("residual gauge translations must be closed under addition")
    bridge = TorchCoverGeometry(layout).double()
    records: list[tuple[np.ndarray, CrystalScaffold]] = []
    for metric, scaffold in zip(toy.metrics, toy.scaffolds, strict=True):
        records.append((np.asarray(metric, dtype=np.float64), scaffold))
        for item in gauge:
            records.append(
                (np.asarray(metric, dtype=np.float64),
                 _translated_scaffold(scaffold, item, layout.setting))
            )
    z_data = torch.tensor(
        np.stack([layout.chart.encode_latent(metric) for metric, _ in records]),
        dtype=torch.float64,
    )
    eta_data = torch.tensor(
        np.stack([layout.encode_exact_parameters(scaffold) for _, scaffold in records]),
        dtype=torch.float64,
    )
    count = len(records)
    return TinyFlowOracle(
        z_data,
        eta_data,
        torch.full((count,), 1.0 / count, dtype=torch.float64),
        bridge.period_basis.to(dtype=torch.float64),
        sigma_z=toy.sigma_z,
        torus_actions=[torch.eye(layout.intrinsic_dimension, dtype=torch.float64)],
        **_torus_kernel(bridge, z_data, eta_data, toy, intrinsic_kernel, reference_length),
    )


def _torus_kernel(
    bridge: TorchCoverGeometry,
    z_data: Tensor,
    eta_data: Tensor,
    toy: F10Toy,
    intrinsic_kernel: str,
    reference_length: float,
) -> dict[str, object]:
    """Oracle keywords for the production torus kernel at these records."""

    if intrinsic_kernel == "fractional-v1":
        return {"sigma_eta": toy.sigma_eta}
    if intrinsic_kernel != "cartesian-v2":
        raise DomainError("intrinsic_kernel must be fractional-v1 or cartesian-v2")
    return {
        "eta_covariances": [
            intrinsic_noise_covariance(
                bridge, z_data[index], eta_data[index], toy.sigma_eta,
                reference_length=reference_length,
            )
            for index in range(len(z_data))
        ]
    }


def _layout_gauge_oracle(
    layout: IntrinsicCoverLayout,
    toy: F10Toy,
    *,
    improper: bool,
    intrinsic_kernel: str = "fractional-v1",
    reference_length: float = 1.0,
) -> TinyFlowOracle:
    """The law of the production objective under ``normalizer-v2`` or ``-v3``.

    The objective samples the action uniformly from the cover's certified
    residual group, so the oracle carries the whole group as affine torus
    actions.  ``improper`` (normalizer-v3) adds the enantiomorph maps.
    """

    from .residual_gauge import cover_residual_gauge

    bridge = TorchCoverGeometry(layout).double()
    actions = cover_residual_gauge(layout, improper=improper).all_eta_actions()
    z_data = torch.tensor(
        np.stack(
            [layout.chart.encode_latent(np.asarray(m, dtype=np.float64)) for m in toy.metrics]
        ),
        dtype=torch.float64,
    )
    eta_data = torch.tensor(
        np.stack([layout.encode_exact_parameters(item) for item in toy.scaffolds]),
        dtype=torch.float64,
    )
    count = len(toy.scaffolds)
    return TinyFlowOracle(
        z_data,
        eta_data,
        torch.full((count,), 1.0 / count, dtype=torch.float64),
        bridge.period_basis.to(dtype=torch.float64),
        sigma_z=toy.sigma_z,
        torus_actions=[torch.as_tensor(m, dtype=torch.float64) for m, _ in actions],
        torus_translations=[torch.as_tensor(t, dtype=torch.float64) for _, t in actions],
        **_torus_kernel(bridge, z_data, eta_data, toy, intrinsic_kernel, reference_length),
    )


@dataclass(frozen=True)
class _FieldEvaluation:
    velocity: FlowVelocity
    lattice_metric: Tensor
    coordinate_metric: Tensor


class LearnedToyField:
    """The production geometry field on one cover, as a batched velocity."""

    def __init__(self, model: CrystalAtlasModel, cover: ContinuousCover) -> None:
        self.model = model
        self.layout: IntrinsicCoverLayout = model.layout(cover)
        parameter = next(model.geometry_field.parameters())
        self.dtype = parameter.dtype
        self.device = parameter.device
        self.bridge = TorchCoverGeometry(self.layout).to(device=self.device, dtype=self.dtype)
        composition = cover.composition
        self._numbers = torch.as_tensor(
            composition.support, dtype=torch.long, device=self.device
        )
        self._counts = torch.as_tensor(
            composition.coefficients, dtype=self.dtype, device=self.device
        )

    def _output(self, time: Tensor, z: Tensor, eta: Tensor) -> PackedIntrinsicFieldOutput:
        # One batched bridge pass (equal to per-record expansion up to rounding);
        # the field is the production packed field.
        batch = z.shape[0]
        time = torch.as_tensor(time, dtype=self.dtype, device=self.device)
        if time.ndim == 0:
            time = time.expand(batch)
        expanded = self.bridge.forward_batch(z, eta)
        output = self.model.geometry_field(
            time,
            z,
            eta,
            None,
            atomic_numbers=expanded.atomic_numbers,
            atom_features=expanded.atom_features,
            fractional_coordinates=expanded.fractional_coordinates,
            lattice=expanded.lattice,
            bounded_lattice_coordinates=expanded.bounded_lattice_coordinates,
            jacobian_cartesian=expanded.jacobian_cartesian,
            composition_atomic_numbers=self._numbers.expand(batch, -1),
            composition_counts=self._counts.expand(batch, -1),
            geometry_is_certified=True,
            lattice_commutant_basis=expanded.lattice_commutant_basis,
        )
        if not isinstance(output, PackedIntrinsicFieldOutput):
            raise RuntimeError("the packed geometry field returned an unpacked output")
        return output

    def _evaluate(self, time: Tensor, z: Tensor, eta: Tensor) -> _FieldEvaluation:
        output = self._output(time, z, eta)
        return _FieldEvaluation(
            FlowVelocity(output.lattice_velocity, output.intrinsic_velocity),
            self.bridge.lattice_pullback_metric_batch(z),
            output.coordinate_metric.detach(),
        )

    def __call__(self, time: Tensor, z: Tensor, eta: Tensor) -> FlowVelocity:
        output = self._output(time, z, eta)
        return FlowVelocity(output.lattice_velocity, output.intrinsic_velocity)


def train_on_toy(
    model: CrystalAtlasModel,
    toy: F10Toy,
    *,
    steps: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    final_learning_rate: float | None = None,
    intrinsic_kernel: str = "fractional-v1",
    coupling: str = "independent",
    coupling_candidates: int = 8,
) -> list[float]:
    """Adam on the production batched CFM objective; returns per-step losses.

    With ``final_learning_rate`` the rate follows a cosine from
    ``learning_rate`` to it over ``steps``; otherwise it is constant.
    ``intrinsic_kernel`` and ``coupling`` are passed to the objective (see
    :func:`toy_oracle`; the oracle's endpoint law is the same for every coupling,
    its velocity field is that of the independent coupling).
    """

    if steps < 1 or batch_size < 1 or not learning_rate > 0.0:
        raise DomainError("steps, batch_size and learning_rate must be positive")
    if final_learning_rate is not None and not 0.0 < final_learning_rate <= learning_rate:
        raise DomainError("final_learning_rate must lie in (0, learning_rate]")
    # As in training_objectives: the CFM objective performs no checkpointed inference.
    generator = LearnedCrystalGenerator(model, allow_untrained=True)
    field = model.geometry_field
    parameter = next(field.parameters())
    torch_generator = torch.Generator(device=parameter.device).manual_seed(seed)
    optimizer = torch.optim.Adam(field.parameters(), lr=learning_rate)
    field.train()
    losses = []
    for step in range(steps):
        if final_learning_rate is not None:
            progress = step / max(steps - 1, 1)
            rate = final_learning_rate + 0.5 * (learning_rate - final_learning_rate) * (
                1.0 + math.cos(math.pi * progress)
            )
            for group in optimizer.param_groups:
                group["lr"] = rate
        records = torch.randint(
            len(toy.scaffolds),
            (batch_size,),
            generator=torch_generator,
            device=parameter.device,
        ).tolist()
        time = torch.rand(
            batch_size,
            generator=torch_generator,
            dtype=parameter.dtype,
            device=parameter.device,
        )
        result = generator.geometry_cfm_objective_batch(
            tuple(toy.scaffolds[index] for index in records),
            tuple(toy.metrics[index] for index in records),
            time=time,
            sigma_z=toy.sigma_z,
            sigma_eta=toy.sigma_eta,
            generator=torch_generator,
            geometry_field=field,
            intrinsic_kernel=intrinsic_kernel,
            coupling=coupling,
            coupling_candidates=coupling_candidates,
        )
        loss = result.loss.per_sample.mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    field.eval()
    return losses


def _metric_norm(
    vz: Tensor, veta: Tensor, lattice: Tensor, coordinate: Tensor, atom_count: int
) -> Tensor:
    """The production CFM metric: the coordinate term is divided by the atom count."""

    return torch.einsum("bi,bij,bj->b", vz, lattice, vz) + torch.einsum(
        "bi,bij,bj->b", veta, coordinate, veta
    ) / atom_count


def velocity_relative_error(
    field: LearnedToyField,
    oracle: TinyFlowOracle,
    *,
    times: Sequence[float],
    samples_per_time: int,
    seed: int,
    velocity_override: VelocityField | None = None,
) -> dict[str, float]:
    """``rho_v`` in the production loss metric over production path samples.

    ``velocity_override`` replaces the learned velocity (the metric still
    comes from ``field``).  It exists to validate this harness, e.g. with the
    oracle's own velocity, which must give exactly zero.
    """

    generator = torch.Generator().manual_seed(seed)
    error = scale = 0.0
    lattice_error = lattice_scale = torus_error = torus_scale = 0.0
    for value in times:
        time = torch.full((samples_per_time,), float(value), dtype=torch.float64)
        path = oracle.sample_paths(time, generator=generator)
        z = path.state.z.to(dtype=field.dtype, device=field.device)
        eta = path.state.eta.to(dtype=field.dtype, device=field.device)
        with torch.no_grad():
            evaluation = field._evaluate(time.to(dtype=field.dtype), z, eta)
        scalar = torch.tensor(float(value), dtype=torch.float64)  # one time per oracle call
        exact = oracle.velocity(scalar, path.state.z, path.state.eta)
        predicted = (
            evaluation.velocity
            if velocity_override is None
            else velocity_override(scalar, path.state.z, path.state.eta)
        )
        delta_z = predicted.z.double().cpu() - exact.z
        delta_eta = predicted.eta.double().cpu() - exact.eta
        lattice = evaluation.lattice_metric.double().cpu()
        coordinate = evaluation.coordinate_metric.double().cpu()
        atoms = field.layout.expanded_atom_count
        zero_z, zero_eta = torch.zeros_like(delta_z), torch.zeros_like(delta_eta)
        error += float(_metric_norm(delta_z, delta_eta, lattice, coordinate, atoms).sum())
        scale += float(_metric_norm(exact.z, exact.eta, lattice, coordinate, atoms).sum())
        lattice_part = _metric_norm(delta_z, zero_eta, lattice, coordinate, atoms)
        lattice_error += float(lattice_part.sum())
        lattice_norm = _metric_norm(exact.z, zero_eta, lattice, coordinate, atoms)
        lattice_scale += float(lattice_norm.sum())
        torus_error += float(_metric_norm(zero_z, delta_eta, lattice, coordinate, atoms).sum())
        torus_scale += float(_metric_norm(zero_z, exact.eta, lattice, coordinate, atoms).sum())
    return {
        "relative_velocity_error": error / scale,
        # The pooled ratio is dominated by the larger component; report both.
        "lattice_relative_velocity_error": lattice_error / lattice_scale,
        "torus_relative_velocity_error": torus_error / torus_scale,
        "samples": len(times) * samples_per_time,
    }


def endpoint_quadrature_grid(
    oracle: TinyFlowOracle, *, z_step: float, z_margin: float, torus_points: int
) -> tuple[Tensor, Tensor, float]:
    """Trapezoid nodes and Cartesian weights over a z interval times the torus.

    Identical to the deterministic F10 gate (``tests/test_flow_oracle.py``).
    """

    low = float(oracle.z_data.min()) - z_margin
    high = float(oracle.z_data.max()) + z_margin
    z_values = torch.arange(
        math.floor(low / z_step), math.ceil(high / z_step) + 1, dtype=torch.float64
    )
    unit = (torch.arange(torus_points, dtype=torch.float64) + 0.5) / torus_points
    cells = torch.stack(
        torch.meshgrid(*([unit] * oracle.intrinsic_dimension), indexing="ij"), dim=-1
    ).reshape(-1, oracle.intrinsic_dimension)
    z = (z_step * z_values).repeat_interleave(len(cells))[:, None]
    eta = cells.repeat(len(z_values), 1) @ oracle.period_basis
    weight = z_step * abs(float(torch.linalg.det(oracle.period_basis))) / len(cells)
    return z, eta, weight


def reversed_field(field: VelocityField) -> VelocityField:
    """``-v(1 - t, q)``: integrating it from ``t = 0`` runs the flow backwards."""

    def reverse(time: Tensor, z: Tensor, eta: Tensor) -> FlowVelocity:
        velocity = field(1.0 - torch.as_tensor(time), z, eta)
        return FlowVelocity(z=-velocity.z, eta=-velocity.eta)

    return reverse


def pushforward_density_reverse_mode(
    field: VelocityField,
    oracle: TinyFlowOracle,
    z: Tensor,
    eta: Tensor,
    *,
    steps: int,
    dtype: torch.dtype = torch.float64,
    chunk_size: int = 16,
    device: torch.device | str | None = None,
) -> Tensor:
    """``p_0(x) |det dx/dy|`` with ``x`` the RK4 back-integral of node ``y``.

    ``device`` is where the field is evaluated (default CPU); the result is a
    float64 CPU tensor.

    Nodes are independent, so each row of every Jacobian is one reverse-mode
    pass of the summed output.  They are processed ``chunk_size`` at a time,
    which is exact and bounds the retained autograd graph.
    """

    if chunk_size < 1:
        raise DomainError("chunk_size must be positive")
    if len(z) > chunk_size:
        return torch.cat(
            [
                pushforward_density_reverse_mode(
                    field,
                    oracle,
                    z[start : start + chunk_size],
                    eta[start : start + chunk_size],
                    steps=steps,
                    dtype=dtype,
                    chunk_size=chunk_size,
                    device=device,
                )
                for start in range(0, len(z), chunk_size)
            ]
        )
    lattice = oracle.lattice_dimension
    target = torch.device("cpu") if device is None else torch.device(device)
    start = (
        torch.cat((z, eta), dim=1).to(dtype=dtype, device=target).detach().requires_grad_(True)
    )
    period = oracle.period_basis.to(dtype=dtype, device=target)
    end = integrate_rk4(
        reversed_field(field),
        FlowState(start[:, :lattice], start[:, lattice:]),
        period,
        num_steps=steps,
    ).final_state
    # The torus state is wrapped at each step.  Wrapping is a locally constant
    # translation, so it does not change the Jacobian.
    final = torch.cat((end.z, end.eta), dim=1)
    rows = []
    for column in range(final.shape[1]):
        (gradient,) = torch.autograd.grad(
            final[:, column].sum(), start, retain_graph=column + 1 < final.shape[1]
        )
        rows.append(gradient)
    jacobian = torch.stack(rows, dim=1)
    determinant = torch.linalg.det(jacobian.double()).abs()
    base = oracle.base_log_density(
        end.z.detach().double().cpu(), end.eta.detach().double().cpu()
    ).exp()
    return base * determinant.cpu()


def endpoint_errors(
    field: VelocityField,
    oracle: TinyFlowOracle,
    *,
    z_step: float,
    torus_points: int,
    steps: int,
    moment_floor: float = 0.05,
    dtype: torch.dtype = torch.float64,
    chunk_size: int = 16,
    device: torch.device | str | None = None,
) -> dict[str, object]:
    """Density L1 and relative moment errors of ``field``'s endpoint law."""

    z, eta, weight = endpoint_quadrature_grid(
        oracle, z_step=z_step, z_margin=6.0 * oracle.sigma_z, torus_points=torus_points
    )
    flow = pushforward_density_reverse_mode(
        field, oracle, z, eta, steps=steps, dtype=dtype, chunk_size=chunk_size, device=device
    )
    target = oracle.log_density(torch.tensor(1.0, dtype=torch.float64), z, eta).exp()
    lattice_coordinates = eta @ torch.linalg.inv(oracle.period_basis)
    moments = []
    for frequency in itertools.product(range(-2, 3), repeat=oracle.intrinsic_dimension):
        for power in (0, 1, 2):
            if not any(frequency) and power == 0:
                continue
            exact = oracle.moment(1.0, frequency, z_index=0 if power else None, z_power=power)
            if abs(exact) < moment_floor:
                continue
            phase = 2.0 * math.pi * (
                lattice_coordinates @ torch.tensor(frequency, dtype=torch.float64)
            )
            estimate = complex((weight * flow * z[:, 0] ** power * torch.exp(1j * phase)).sum())
            moments.append(
                {
                    "frequency": list(frequency),
                    "z_power": power,
                    "relative_error": abs(estimate - exact) / abs(exact),
                }
            )
    return {
        "density_l1": float(weight * (flow - target).abs().sum()),
        "flow_mass": float(weight * flow.sum()),
        "target_mass": float(weight * target.sum()),
        "max_relative_moment_error": max(item["relative_error"] for item in moments),
        "moments": moments,
    }
