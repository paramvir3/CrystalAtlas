"""Rejection-free neural sampling over exact arithmetic cover fibres."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np

from .._optional import require_torch
from ..catalogue import CertifiedCatalogue, SpaceGroupSetting
from ..composition import Composition
from ..constraints import OccupationTable, RowAction, SuffixReachability, WyckoffRowSpec
from ..cover import canonical_rows
from ..cover_legality import (
    AtomBound,
    PrimitiveCountCondition,
    feasible_formula_units,
    generic_hidden_translation_index,
    legal_settings,
)
from ..errors import DomainError, InvariantViolation
from ..geometry import ContinuousCover, OrbitType
from ..symmetry_descriptors import (
    ROW_GEOMETRY_DESCRIPTOR_DIM,
    ROW_SITE_DESCRIPTOR_DIM,
    row_geometry_descriptor,
    row_site_descriptor,
)
from .selector import ArithmeticCoverSelector

torch = require_torch()
Tensor = torch.Tensor


def operation_derived_row_features(
    setting: SpaceGroupSetting,
    rows: Sequence[WyckoffRowSpec],
    *,
    feature_dim: int,
    reference: Tensor,
    geometry: bool = False,
) -> Tensor:
    """Return the shared non-learned descriptor of each exact Wyckoff row.

    The base block is ``(log(1 + mu), d/3, log(1 + |S|), [d = 0])``.  A site
    block follows when ``feature_dim`` has room for it (review m1).  With
    ``geometry`` the exact-chart block of
    :func:`~crystalatlas.symmetry_descriptors.row_geometry_descriptor` follows
    (review 2026-10-29, M1); without it the output is the legacy one.
    """

    if not isinstance(setting, SpaceGroupSetting):
        raise TypeError("setting must be a SpaceGroupSetting")
    if feature_dim < 1:
        raise DomainError("row feature dimension must be positive")
    if not isinstance(geometry, bool):
        raise TypeError("geometry must be bool")
    geometry_dim = ROW_GEOMETRY_DESCRIPTOR_DIM if geometry else 0
    if feature_dim < 4 + geometry_dim:
        raise DomainError("row feature dimension cannot hold the geometry block")
    # The site block is keyed on the room left once the geometry block is
    # reserved, so (feature_dim, geometry) fixes the layout; legacy calls
    # (geometry=False) keep the old threshold.
    with_site = feature_dim - geometry_dim >= 4 + ROW_SITE_DESCRIPTOR_DIM
    features = reference.new_zeros(len(rows), feature_dim)
    for index, row_spec in enumerate(rows):
        row = setting.wyckoff(row_spec.label)
        values: tuple[float, ...] = (
            np.log1p(row.multiplicity),
            row.dimension / 3.0,
            np.log1p(row.site_symmetry_order),
            float(row.dimension == 0),
        )
        if with_site:
            # Site-symmetry type (review m1): rotation-type histogram of the
            # exact site stabilizer, shared by normalizer-equivalent rows.
            values = (*values, *row_site_descriptor(setting, row))
        if geometry:
            values = (*values, *row_geometry_descriptor(setting, row))
        limit = min(len(values), feature_dim)
        features[index, :limit] = reference.new_tensor(values[:limit])
    return features


@dataclass(frozen=True, slots=True)
class NeuralCoverSample:
    cover: ContinuousCover
    rows: tuple[WyckoffRowSpec, ...]
    occupation: OccupationTable
    log_probability: float
    generic_translation_rejections: int = 0
    """Tables rejected for a generic hidden translation (primitive-count track only).

    ``log_probability`` is that of the masked table law; under conditioning the
    exact conditional law also divides by the unnormalized acceptance mass,
    which is not computed.
    """


class NeuralCoverSampler:
    """Bind learned scores to exact ``nu/group/suffix`` legality masks."""

    def __init__(
        self,
        model: ArithmeticCoverSelector,
        *,
        catalogue: CertifiedCatalogue | None = None,
        max_raw_atoms: int = 40,
        max_primitive_atoms: int | None = None,
        oracle_cache_size: int = 4096,
    ) -> None:
        if catalogue is None:
            raise DomainError(
                "NeuralCoverSampler requires an explicit certified catalogue"
            )
        if max_raw_atoms <= 0:
            raise DomainError("max_raw_atoms must be positive")
        if oracle_cache_size < 1:
            raise DomainError("oracle_cache_size must be positive")
        catalogue.validate()
        if model.config.num_space_groups < max(catalogue.supported_international_numbers):
            raise DomainError("selector group vocabulary cannot represent the catalogue")
        self.model = model
        self.catalogue = catalogue
        self.max_raw_atoms = int(max_raw_atoms)
        # Shared with the training targets, so both normalize over one support.
        self.atom_bound = AtomBound(
            conventional=self.max_raw_atoms, primitive=max_primitive_atoms
        )
        self.oracle_cache_size = int(oracle_cache_size)
        self._settings = tuple(
            sorted(catalogue.settings.values(), key=lambda item: item.international_number)
        )
        self._oracle_cache: OrderedDict[
            tuple[Composition, int, str], SuffixReachability
        ] = OrderedDict()

    def _oracle(
        self, composition: Composition, nu: int, setting: SpaceGroupSetting
    ) -> SuffixReachability:
        key = (composition, nu, setting.key)
        oracle = self._oracle_cache.get(key)
        if oracle is None:
            oracle = SuffixReachability(composition, nu, canonical_rows(setting))
            self._oracle_cache[key] = oracle
            if len(self._oracle_cache) > self.oracle_cache_size:
                self._oracle_cache.popitem(last=False)
        else:
            self._oracle_cache.move_to_end(key)
        return oracle

    def feasible_nu(
        self, composition: Composition, condition: PrimitiveCountCondition | None = None
    ) -> tuple[int, ...]:
        return feasible_formula_units(
            composition,
            self._settings,
            max_formula_units=self.model.config.num_nu_choices,
            bound=self.atom_bound,
            oracle=self._oracle,
            condition=condition,
        )

    @staticmethod
    def _draw(
        logits: Tensor,
        *,
        temperature: float,
        generator: torch.Generator | None,
    ) -> tuple[int, float]:
        tempered = logits / temperature
        probabilities = torch.softmax(tempered, dim=-1)
        index = int(torch.multinomial(probabilities, 1, generator=generator).item())
        return index, float(torch.log_softmax(tempered, dim=-1)[index].item())

    def _row_features(
        self, setting: SpaceGroupSetting, rows: Sequence[WyckoffRowSpec], reference: Tensor
    ) -> Tensor:
        return operation_derived_row_features(
            setting,
            rows,
            feature_dim=self.model.config.row_feature_dim,
            reference=reference,
            geometry=self.model.config.row_geometry_features,
        )

    @torch.no_grad()
    def sample(
        self,
        composition: Composition,
        *,
        temperature: float = 1.0,
        generator: torch.Generator | None = None,
        device: torch.device | None = None,
        primitive_atom_count: int | None = None,
        max_table_attempts: int = 64,
    ) -> NeuralCoverSample:
        """Sample one exact cover; optionally condition on the primitive atom count.

        With ``primitive_atom_count`` the (nu, G) masks enforce
        ``nu |c| = I_G N_prim`` (review M6), and whole tables whose generic hidden
        translation index exceeds one are rejected and redrawn.  Without it the
        legacy masks and random stream are unchanged.
        """

        condition = (
            None if primitive_atom_count is None
            else PrimitiveCountCondition(primitive_atom_count)
        )
        for attempt in range(max_table_attempts):
            sample = self._sample_once(
                composition,
                temperature=temperature,
                generator=generator,
                device=device,
                condition=condition,
            )
            if condition is None:
                return sample
            if generic_hidden_translation_index(sample.cover, self.catalogue) == 1:
                return replace(sample, generic_translation_rejections=attempt)
        raise InvariantViolation(
            "every sampled table carried a generic hidden translation; the primitive "
            "count is not attainable within the table-attempt budget"
        )

    def _sample_once(
        self,
        composition: Composition,
        *,
        temperature: float,
        generator: torch.Generator | None,
        device: torch.device | None,
        condition: PrimitiveCountCondition | None,
    ) -> NeuralCoverSample:
        if not isinstance(composition, Composition):
            raise TypeError("composition must be a Composition")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        target_device = device or next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        atomic_numbers = torch.as_tensor(
            [composition.support], dtype=torch.long, device=target_device
        )
        counts = torch.as_tensor(
            [composition.coefficients], dtype=dtype, device=target_device
        )
        context = self.model.encode_composition(atomic_numbers, counts)
        feasible_nu = self.feasible_nu(composition, condition)
        if not feasible_nu:
            raise InvariantViolation("no formula-unit multiplier has an arithmetic cover")
        nu_mask = torch.zeros(
            (1, self.model.config.num_nu_choices),
            dtype=torch.bool,
            device=target_device,
        )
        for nu in feasible_nu:
            nu_mask[0, nu - 1] = True
        nu_logits = self.model.nu_logits(context, legal_mask=nu_mask)[0]
        nu_index, log_probability = self._draw(
            nu_logits, temperature=temperature, generator=generator
        )
        nu = nu_index + 1

        group_mask = torch.zeros(
            (1, self.model.config.num_space_groups),
            dtype=torch.bool,
            device=target_device,
        )
        reachable: dict[int, SuffixReachability] = {}
        for setting in legal_settings(
            composition,
            nu,
            self._settings,
            bound=self.atom_bound,
            oracle=self._oracle,
            condition=condition,
        ):
            group_index = setting.international_number - 1
            group_mask[0, group_index] = True
            reachable[group_index] = self._oracle(composition, nu, setting)
        group_logits = self.model.group_logits(
            context,
            torch.as_tensor([nu_index], dtype=torch.long, device=target_device),
            legal_mask=group_mask,
        )[0]
        group_index, group_log_probability = self._draw(
            group_logits, temperature=temperature, generator=generator
        )
        log_probability += group_log_probability
        try:
            oracle = reachable[group_index]
        except KeyError as exc:  # guarded by the exact group mask
            raise RuntimeError("selector sampled a masked arithmetic group") from exc
        setting = self.catalogue.get(group_index + 1)
        rows = oracle.rows
        row_features = self._row_features(setting, rows, context).unsqueeze(0)
        previous = torch.zeros(
            (1, len(rows), len(composition)), dtype=dtype, device=target_device
        )
        residual = oracle.initial_residual
        selected_actions: list[RowAction] = []
        nu_tensor = torch.as_tensor([nu_index], dtype=torch.long, device=target_device)
        group_tensor = torch.as_tensor([group_index], dtype=torch.long, device=target_device)
        for row_index in range(len(rows)):
            allowed = oracle.allowed_actions(row_index, residual)
            if not allowed:
                raise RuntimeError("reachable suffix state had no legal row action")
            action_count = max(len(allowed), 1)
            candidates = torch.zeros(
                (1, len(rows), action_count, len(composition)),
                dtype=dtype,
                device=target_device,
            )
            legal = torch.zeros(
                (1, len(rows), action_count), dtype=torch.bool, device=target_device
            )
            # Past/future rows get a harmless singleton because their logits are
            # ignored. Causal hidden state at row_index sees only selected past.
            legal[:, :, 0] = True
            for candidate_index, action in enumerate(allowed):
                candidates[0, row_index, candidate_index] = torch.as_tensor(
                    action.counts, dtype=dtype, device=target_device
                )
                legal[0, row_index, candidate_index] = True
            logits = self.model.row_logits(
                context,
                nu_tensor,
                group_tensor,
                atomic_numbers,
                row_features,
                previous,
                candidates,
                legal_mask=legal,
            )[0, row_index]
            selected_index, action_log_probability = self._draw(
                logits, temperature=temperature, generator=generator
            )
            if selected_index >= len(allowed):
                raise RuntimeError("selector sampled a padded row action")
            action = allowed[selected_index]
            log_probability += action_log_probability
            selected_actions.append(action)
            previous[0, row_index] = torch.as_tensor(
                action.counts, dtype=dtype, device=target_device
            )
            residual = oracle.advance(
                row_index, residual, action, require_completion_safe=True
            )
        occupation = OccupationTable(tuple(selected_actions))
        if not oracle.accepts(occupation):
            raise RuntimeError("neural cover path failed the exact suffix oracle")
        orbit_types: list[OrbitType] = []
        for row, action in zip(rows, occupation.actions, strict=True):
            for z, count in zip(composition.support, action.counts, strict=True):
                orbit_types.extend(OrbitType(row.label, z) for _ in range(count))
        cover = ContinuousCover(nu, setting.key, composition, tuple(orbit_types))
        return NeuralCoverSample(
            cover=cover,
            rows=rows,
            occupation=occupation,
            log_probability=log_probability,
        )


__all__ = [
    "NeuralCoverSample",
    "NeuralCoverSampler",
    "canonical_rows",
    "operation_derived_row_features",
]
