from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from test_training_data import _store

torch = pytest.importorskip("torch")

from crystalatlas.catalogue import MINIMAL_CATALOGUE  # noqa: E402
from crystalatlas.config import (  # noqa: E402
    CrystalAtlasConfig,
    DomainConfig,
    ModelConfig,
    NumericsConfig,
)
from crystalatlas.errors import DomainError  # noqa: E402
from crystalatlas.learned import CrystalAtlasModel  # noqa: E402
from crystalatlas.trainer import (  # noqa: E402
    ComponentModels,
    ScalableTrainer,
    StagedTrainingSchedule,
    TrainerConfig,
    TrainingPhase,
)
from crystalatlas.training import CheckpointError, LossWeights  # noqa: E402
from crystalatlas.training_data import (  # noqa: E402
    DeterministicBucketBatchSampler,
    StoredTrainingDataset,
    TrainingMeasure,
)
from crystalatlas.training_objectives import (  # noqa: E402
    CFMDequantization,
    CrystalAtlasTrainingObjective,
    ExactBatchCrystallographicAudit,
    ObjectiveDataMeasure,
)

pytestmark = pytest.mark.ml


def _config() -> CrystalAtlasConfig:
    return CrystalAtlasConfig(
        domain=DomainConfig(
            element_vocabulary=(6, 7, 8),
            dng_element_vocabulary=(6, 7, 8),
            max_distinct_elements=3,
            max_raw_atoms=40,
            max_formula_units=16,
            metric_radius=2.0,
            interaction_cutoff_angstrom=2.5,
        ),
        numerics=NumericsConfig(ode_steps=2, max_periodic_edges=20_000),
        model=ModelConfig(
            composition_width=24,
            composition_layers=1,
            selector_width=24,
            selector_layers=1,
            geometry_scalar_width=24,
            geometry_vector_width=12,
            geometry_layers=1,
            attention_heads=3,
            radial_features=6,
        ),
    )


def _setup(tmp_path: Path):  # type: ignore[no-untyped-def]
    store = _store(tmp_path)
    model = CrystalAtlasModel(_config(), catalogue=MINIMAL_CATALOGUE).double()
    dataset = StoredTrainingDataset(
        store,
        model.composition_codec,
        MINIMAL_CATALOGUE,
        domain=model.scientific_config.domain,
    )
    objective = CrystalAtlasTrainingObjective(
        model,
        ObjectiveDataMeasure.from_dataset(dataset),
    )
    models = ComponentModels(
        root=model,
        composition=model.composition_model,
        cover=model.cover_selector,
        geometry=model.geometry_field,
    )
    return store, model, dataset, objective, models


def _grads(module):  # type: ignore[no-untyped-def]
    return tuple(
        parameter.grad for parameter in module.parameters() if parameter.grad is not None
    )


def test_factor_objectives_are_finite_and_reach_only_the_selected_factor(
    tmp_path: Path,
) -> None:
    store, model, dataset, objective, models = _setup(tmp_path)
    try:
        unique = next(
            DeterministicBucketBatchSampler(
                dataset,
                8,
                seed=11,
                measure=TrainingMeasure.UNIQUE_COMPOSITION,
            )
        )
        structure = next(
            DeterministicBucketBatchSampler(
                dataset,
                2,
                seed=11,
                measure=TrainingMeasure.STRUCTURE,
            )
        )
        phase_batches = (
            (TrainingPhase.COMPOSITION, unique, model.composition_model),
            (TrainingPhase.COVER, structure, model.cover_selector),
            (TrainingPhase.GEOMETRY, structure, model.geometry_field),
        )
        for phase, batch, selected in phase_batches:
            model.zero_grad(set_to_none=True)
            losses = objective(models, batch, phase)
            value = losses.objective(phase, LossWeights())
            assert torch.isfinite(value)
            value.backward()
            assert _grads(selected)
            assert all(torch.isfinite(gradient).all() for gradient in _grads(selected))
            for other in (
                model.composition_model,
                model.cover_selector,
                model.geometry_field,
            ):
                if other is not selected:
                    assert not _grads(other)
    finally:
        store.close()


class _UniformSelector(torch.nn.Module):
    """Stand-in selector with equal logits on every legal action."""

    def forward(  # type: ignore[no-untyped-def]
        self,
        numbers,
        counts,
        nu_targets,
        group_targets,
        row_features,
        selected_actions,
        candidates,
        *,
        nu_legal_mask,
        group_legal_mask,
        candidate_legal_mask,
    ):
        def flat(mask):  # type: ignore[no-untyped-def]
            zeros = torch.zeros(mask.shape, dtype=counts.dtype, device=counts.device)
            return zeros.masked_fill(~mask, -torch.inf)

        return flat(nu_legal_mask), flat(group_legal_mask), flat(candidate_legal_mask)


def test_cover_factor_terms_decompose_the_cover_loss(tmp_path: Path) -> None:
    store, model, dataset, objective, models = _setup(tmp_path)
    try:
        batches = tuple(
            DeterministicBucketBatchSampler(
                dataset, 2, seed=11, measure=TrainingMeasure.STRUCTURE
            )
        )
        assert batches
        for batch in batches:
            size = len(batch.record_ids)
            with torch.no_grad():
                terms = objective.cover_factor_terms(model.cover_selector, batch)
                loss = objective(models, batch, TrainingPhase.COVER).cover_nll
            per_structure = terms.nu_nll + terms.group_nll + terms.row_nll
            assert per_structure.shape == (size,)
            # The cover loss is the mean of exactly this sum.
            assert loss.item() == per_structure.mean().item()

            cover = batch.cover
            nu_count = torch.tensor(cover.nu_legal_mask).sum(dim=-1)
            group_count = torch.tensor(cover.group_legal_mask).sum(dim=-1)
            row_count = torch.tensor(cover.candidate_legal_mask).sum(dim=-1)
            assert torch.equal(terms.nu_reference, nu_count.double().log())
            assert torch.equal(terms.group_reference, group_count.double().log())
            assert torch.equal(terms.row_reference, row_count.double().log().sum(dim=-1))
            assert bool(((terms.nu_rank >= 0) & (terms.nu_rank < nu_count)).all())
            assert bool(((terms.group_rank >= 0) & (terms.group_rank < group_count)).all())
            assert bool(
                ((terms.row_errors >= 0) & (terms.row_errors <= row_count.shape[-1])).all()
            )

            # Equal logits: every factor equals its uniform reference, no action is
            # strictly better than the target, and no row is in error.
            flat = objective.cover_factor_terms(_UniformSelector(), batch)
            for name in ("nu", "group", "row"):
                torch.testing.assert_close(
                    getattr(flat, f"{name}_nll"),
                    getattr(flat, f"{name}_reference"),
                    rtol=0.0,
                    atol=1e-12,
                )
            assert not bool(flat.nu_rank.any())
            assert not bool(flat.group_rank.any())
            assert not bool(flat.row_errors.any())
    finally:
        store.close()


def test_cover_rank_counts_strictly_better_legal_actions() -> None:
    from crystalatlas.training_objectives import _strictly_better

    logits = torch.tensor(
        [[0.5, 2.0, -torch.inf, 1.0], [3.0, 3.0, 9.0, 3.0]], dtype=torch.float64
    )
    legal = torch.tensor([[True, True, False, True], [True, True, False, True]])
    # Row 0: target 3 (logit 1.0) is beaten by action 1 only.  Row 1: the larger
    # logit of action 2 is illegal and ties are not strictly better.
    ranks = _strictly_better(logits, torch.tensor([3, 0]), legal)
    assert ranks.tolist() == [1, 0]
    assert _strictly_better(logits, torch.tensor([0, 3]), legal).tolist() == [2, 0]


def test_joint_objective_uses_the_declared_population_measures(tmp_path: Path) -> None:
    store, model, dataset, objective, models = _setup(tmp_path)
    try:
        unique = next(
            DeterministicBucketBatchSampler(
                dataset,
                8,
                seed=17,
                measure=TrainingMeasure.UNIQUE_COMPOSITION,
            )
        )
        structure = next(
            DeterministicBucketBatchSampler(
                dataset,
                2,
                seed=17,
                measure=TrainingMeasure.STRUCTURE,
            )
        )
        unique_nll = objective(models, unique, TrainingPhase.COMPOSITION).composition_nll
        joint_losses = objective(models, structure, TrainingPhase.JOINT)
        assert unique_nll is not None
        assert joint_losses.composition_nll is not None
        # Both stored polymorphs have the same composition.  The inverse
        # multiplicity and N_structure/N_composition factor must therefore
        # reproduce the unique-composition expectation exactly.
        assert torch.allclose(
            unique_nll,
            joint_losses.composition_nll,
            rtol=1.0e-12,
            atol=1.0e-12,
        )
        total = joint_losses.objective(TrainingPhase.JOINT, LossWeights())
        assert torch.isfinite(total)
        total.backward()
        assert _grads(model.composition_model)
        assert _grads(model.cover_selector)
        assert _grads(model.geometry_field)
    finally:
        store.close()


def test_exact_batch_audit_redecodes_and_roundtrips_in_float64(tmp_path: Path) -> None:
    store, model, dataset, _, _ = _setup(tmp_path)
    try:
        batch = next(
            DeterministicBucketBatchSampler(
                dataset,
                2,
                seed=29,
                measure=TrainingMeasure.STRUCTURE,
            )
        )
        result = ExactBatchCrystallographicAudit(model)(batch, torch.float64)
        assert result.residuals.dtype == torch.float64
        assert result.passed
        assert float(result.residuals.abs().max().item()) <= result.tolerance
    finally:
        store.close()


def test_memoized_audit_equals_recomputation_and_audits_changed_inputs(
    tmp_path: Path,
) -> None:
    store, model, dataset, _, _ = _setup(tmp_path)
    try:
        batch = next(
            DeterministicBucketBatchSampler(
                dataset,
                2,
                seed=29,
                measure=TrainingMeasure.STRUCTURE,
            )
        )
        with pytest.raises(TypeError):
            ExactBatchCrystallographicAudit(model, memoize=1)  # type: ignore[arg-type]
        plain = ExactBatchCrystallographicAudit(model)
        memo = ExactBatchCrystallographicAudit(model, memoize=True)
        assert plain.memoize is False
        decodes = 0
        compute = memo._record_residuals

        def counted(*args: object) -> tuple[np.ndarray, ...]:
            nonlocal decodes
            decodes += 1
            return compute(*args)

        memo._record_residuals = counted  # type: ignore[method-assign]
        records = len(batch.geometry.scaffolds)
        expected = plain(batch, torch.float64)
        first = memo(batch, torch.float64)
        assert decodes == records
        second = memo(batch, torch.float64)
        assert decodes == records
        assert first.passed and second.passed
        assert torch.equal(first.residuals, expected.residuals)
        assert torch.equal(second.residuals, expected.residuals)
        assert plain._memo == {}

        # A changed stored input is a different key: it is audited afresh and
        # rejected exactly as by the audit without a memo.
        latent = np.array(batch.geometry.lattice_latent, dtype=np.float64)
        latent[0, 0] += 1.0e-3
        tampered = replace(batch, geometry=replace(batch.geometry, lattice_latent=latent))
        rejected = memo(tampered, torch.float64)
        assert decodes == records + 1
        assert not rejected.passed
        assert torch.equal(rejected.residuals, plain(tampered, torch.float64).residuals)
        # The original record is still served from its own entry.
        assert torch.equal(memo(batch, torch.float64).residuals, expected.residuals)
        assert decodes == records + 1
    finally:
        store.close()


def test_real_objective_runs_factor_first_then_joint_training(tmp_path: Path) -> None:
    store, model, dataset, objective, _ = _setup(tmp_path)
    try:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-5)
        trainer = ScalableTrainer(
            model,
            optimizer,
            StagedTrainingSchedule(1, 1, 1, 1),
            config=TrainerConfig(
                accumulation_steps=2,
                ema_decay=0.99,
                transactional_optimizer_step=False,
            ),
        )
        audit = ExactBatchCrystallographicAudit(model)
        observed: list[TrainingPhase] = []
        for phase in (
            TrainingPhase.COMPOSITION,
            TrainingPhase.COVER,
            TrainingPhase.GEOMETRY,
            TrainingPhase.JOINT,
        ):
            measure = (
                TrainingMeasure.UNIQUE_COMPOSITION
                if phase is TrainingPhase.COMPOSITION
                else TrainingMeasure.STRUCTURE
            )
            sampler = DeterministicBucketBatchSampler(
                dataset,
                2,
                seed=101,
                measure=measure,
            )
            report = trainer.train_epoch(
                sampler,
                objective,
                crystallographic_audit=audit,
            )
            observed.append(report.phase)
            assert report.global_structures > 0
            assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
        assert observed == [
            TrainingPhase.COMPOSITION,
            TrainingPhase.COVER,
            TrainingPhase.GEOMETRY,
            TrainingPhase.JOINT,
        ]
        assert trainer.cursor.epoch == trainer.schedule.total_epochs
    finally:
        store.close()


def test_objective_contract_binds_dataset_dequantization_and_root_model(
    tmp_path: Path,
) -> None:
    store, model, dataset, objective, models = _setup(tmp_path)
    try:
        assert objective.contract["data_measure"]["dataset_fingerprint"] == (
            dataset.fingerprint
        )
        trainer = ScalableTrainer(
            model,
            torch.optim.AdamW(model.parameters(), lr=1.0e-5),
            StagedTrainingSchedule(1, 0, 0),
            config=TrainerConfig(ema_decay=None),
        )
        sampler = DeterministicBucketBatchSampler(
            dataset,
            8,
            seed=303,
            measure=TrainingMeasure.UNIQUE_COMPOSITION,
        )
        batch = next(sampler)
        trainer.train_window((batch,), objective)
        state = trainer.state_dict(sampler=sampler)
        assert state["objective_contract"]["fingerprint"] == (objective.contract_fingerprint)

        class WrongDatasetSampler:
            def state_dict(self):  # type: ignore[no-untyped-def]
                payload = dict(sampler.state_dict())
                payload["dataset_fingerprint"] = "f" * 64
                return payload

            def load_state_dict(self, payload):  # type: ignore[no-untyped-def]
                del payload

        with pytest.raises(CheckpointError, match="sampler dataset differs"):
            trainer.state_dict(sampler=WrongDatasetSampler())

        changed = CrystalAtlasTrainingObjective(
            model,
            ObjectiveDataMeasure.from_dataset(dataset),
            dequantization=CFMDequantization(
                sigma_lattice=0.051,
                sigma_intrinsic=0.03,
            ),
        )
        with pytest.raises(DomainError, match="objective contract"):
            trainer.train_window((batch,), changed)

        other_model = CrystalAtlasModel(_config(), catalogue=MINIMAL_CATALOGUE).double()
        wrong_root = CrystalAtlasTrainingObjective(
            other_model,
            ObjectiveDataMeasure.from_dataset(dataset),
        )
        with pytest.raises(DomainError, match="identical root model"):
            wrong_root(models, batch, TrainingPhase.COMPOSITION)
    finally:
        store.close()


def test_objective_rejects_a_batch_from_a_different_dataset_view(
    tmp_path: Path,
) -> None:
    store, model, dataset, _, models = _setup(tmp_path)
    try:
        batch = next(
            DeterministicBucketBatchSampler(
                dataset,
                8,
                seed=401,
                measure=TrainingMeasure.UNIQUE_COMPOSITION,
            )
        )
        mismatched = CrystalAtlasTrainingObjective(
            model,
            ObjectiveDataMeasure(
                structure_count=len(dataset),
                unique_composition_count=dataset.unique_composition_count,
                dataset_fingerprint="f" * 64,
            ),
        )
        with pytest.raises(DomainError, match="different datasets"):
            mismatched(models, batch, TrainingPhase.COMPOSITION)
    finally:
        store.close()
