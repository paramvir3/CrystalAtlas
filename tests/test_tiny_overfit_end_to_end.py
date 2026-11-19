"""Bounded proof that the production tiny-overfit path is actually executable.

This is an integration smoke test, not an assertion that four optimizer updates
fit the fixture and emphatically not evidence of predictive performance.
"""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
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
from crystalatlas.decoder import ExactCrystalDecoder  # noqa: E402
from crystalatlas.diagnostics import (  # noqa: E402
    AcceptableTarget,
    BootstrapProtocol,
    CompositionForecast,
    DiagnosticEvaluationContract,
    EquivalenceMatcherContract,
    GeometryDecompositionCase,
    RankedAttempt,
    RankedRetrievalCase,
    TinyOverfitEvaluation,
    TinyOverfitGateProtocol,
    TinyOverfitThresholds,
    composition_forecasts_sha256,
    evaluate_composition_forecasts,
    evaluate_geometry_decomposition,
    evaluate_scaffold_retrieval,
    geometry_corpus_fingerprints,
    retrieval_corpus_fingerprints,
    run_tiny_dataset_overfit_gate,
)
from crystalatlas.errors import CrystalAtlasError  # noqa: E402
from crystalatlas.geometry import ContinuousCover  # noqa: E402
from crystalatlas.learned import CrystalAtlasModel, LearnedCrystalGenerator  # noqa: E402
from crystalatlas.physical_matching import (  # noqa: E402
    encode_physical_structure,
    physical_structure_match,
    physical_structure_matcher_contract,
)
from crystalatlas.pipeline import GenerationMode, GenerationRequest  # noqa: E402
from crystalatlas.provenance import ProvenanceRecord  # noqa: E402
from crystalatlas.trainer import (  # noqa: E402
    ComponentModels,
    ScalableTrainer,
    StagedTrainingSchedule,
    TrainerConfig,
    TrainingPhase,
)
from crystalatlas.training import LossWeights  # noqa: E402
from crystalatlas.training_data import (  # noqa: E402
    DeterministicBucketBatchSampler,
    StoredTrainingDataset,
    TrainingBatch,
    TrainingMeasure,
)
from crystalatlas.training_objectives import (  # noqa: E402
    CrystalAtlasTrainingObjective,
    ExactBatchCrystallographicAudit,
    ObjectiveDataMeasure,
)

pytestmark = pytest.mark.ml

def _fixture_scaffold_match(candidate: bytes, target: bytes) -> bool:
    return candidate.partition(b"\n")[0] == target.partition(b"\n")[0]


def _prefixed_physical_match(candidate: bytes, target: bytes) -> bool:
    """Certified physical equivalence of the structure record after the scaffold line.

    The tiny gate binds one target corpus for scaffold and geometry evidence,
    so both arms share a payload: a scaffold-identity line followed by the
    canonical primitive-cell record of :mod:`crystalatlas.physical_matching`.
    """

    return physical_structure_match(
        candidate.partition(b"\n")[2], target.partition(b"\n")[2]
    )


_SCAFFOLD_MATCHER = EquivalenceMatcherContract.from_callable(
    _fixture_scaffold_match,
    algorithm="local-fixture-exact-cover-equivalence",
    version="1",
    cell_policy="canonical-P1-cover-only",
    parameters={"comparison": "nu-group-composition-ordered-orbit-types"},
)
_GEOMETRY_MATCHER = EquivalenceMatcherContract.from_callable(
    _prefixed_physical_match,
    algorithm="scaffold-prefixed-certified-periodic-structure-equivalence",
    version="1",
    cell_policy=physical_structure_matcher_contract().cell_policy,
    parameters={
        "physical_matcher_contract": physical_structure_matcher_contract().to_manifest(),
        "prefix": "one-scaffold-identity-line",
    },
)


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


def _cover_identity(cover: ContinuousCover) -> bytes:
    composition = ",".join(
        f"{atomic_number}:{count}" for atomic_number, count in cover.composition.counts
    )
    orbits = ",".join(f"{orbit.wyckoff_label}:{orbit.atomic_number}" for orbit in cover.orbits)
    return f"{cover.nu}|{cover.space_group}|{composition}|{orbits}".encode("ascii")


def _structure_payload(
    cover: ContinuousCover,
    lattice_values: np.ndarray,
    fractional_values: np.ndarray,
    atomic_numbers: np.ndarray,
) -> bytes:
    return (
        _cover_identity(cover)
        + b"\n"
        + encode_physical_structure(lattice_values, fractional_values, atomic_numbers)
    )


def _target_payload(batch: TrainingBatch) -> tuple[bytes, ContinuousCover]:
    scaffold = batch.geometry.scaffolds[0]
    decoded = ExactCrystalDecoder(MINIMAL_CATALOGUE).decode(
        scaffold,
        np.asarray(batch.geometry.metric[0], dtype=np.float64),
    )
    cover = ContinuousCover.from_exact_scaffold(scaffold)
    species = np.asarray(decoded.species, dtype=np.int64)
    return (
        _structure_payload(
            cover,
            decoded.lattice,
            decoded.fractional_positions_float,
            species,
        ),
        cover,
    )


def _structure_batch(dataset: StoredTrainingDataset) -> TrainingBatch:
    return next(
        DeterministicBucketBatchSampler(
            dataset,
            batch_size=1,
            seed=701,
            measure=TrainingMeasure.STRUCTURE,
        )
    )


def _composition_forecast(
    model: CrystalAtlasModel,
    batch: TrainingBatch,
) -> CompositionForecast:
    supervision = batch.composition
    device = next(model.parameters()).device
    tokens = torch.tensor(supervision.token_ids, dtype=torch.long, device=device)
    token_types = torch.tensor(supervision.token_types, dtype=torch.long, device=device)
    legal = torch.tensor(supervision.legal_mask, dtype=torch.bool, device=device)
    sequence = torch.tensor(supervision.sequence_mask, dtype=torch.bool, device=device)
    with torch.no_grad():
        per_token = model.composition_model(
            tokens,
            token_types,
            legal_mask=legal,
            sequence_mask=sequence,
        )
    target_nll = float((per_token * sequence.to(dtype=per_token.dtype)).sum().item())
    return CompositionForecast(
        query_id=batch.record_ids[0],
        target_log_probability=-target_nll,
        token_count=int(sequence.sum().item()),
    )


def test_authenticated_factor_first_training_reaches_real_diagnostics(
    tmp_path: Path,
) -> None:
    torch.manual_seed(607)
    with _store(tmp_path) as store:
        config = _config()
        model = CrystalAtlasModel(config, catalogue=MINIMAL_CATALOGUE).double()
        dataset = StoredTrainingDataset(
            store,
            model.composition_codec,
            MINIMAL_CATALOGUE,
            domain=config.domain,
            record_ids=(store.record_ids[0],),
        )
        objective = CrystalAtlasTrainingObjective(
            model,
            ObjectiveDataMeasure.from_dataset(dataset),
        )
        components = ComponentModels(
            root=model,
            composition=model.composition_model,
            cover=model.cover_selector,
            geometry=model.geometry_field,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-4)
        schedule = StagedTrainingSchedule(1, 1, 1, 1)
        trainer = ScalableTrainer(
            model,
            optimizer,
            schedule,
            config=TrainerConfig(
                accumulation_steps=1,
                ema_decay=None,
                transactional_optimizer_step=True,
            ),
        )
        audit = ExactBatchCrystallographicAudit(model)
        evaluation_batch = _structure_batch(dataset)
        target_payload, target_cover = _target_payload(evaluation_batch)
        targets = (AcceptableTarget(dataset.record_ids[0], target_payload),)
        target_cases = (RankedRetrievalCase(dataset.record_ids[0], targets, ()),)
        target_corpus_sha256 = retrieval_corpus_fingerprints(target_cases).target_corpus_sha256
        split_sha256 = store.manifest.splits[0].file.sha256
        generation_protocol_sha256 = sha256(
            b"crystalatlas.local-fixture-generation.v1|one-attempt|no-retry"
        ).hexdigest()
        base_policy = DiagnosticEvaluationContract(
            dataset_manifest_sha256=store.manifest.build_manifest_sha256,
            split_manifest_sha256=split_sha256,
            checkpoint_sha256="0" * 64,
            target_corpus_sha256=target_corpus_sha256,
            generated_attempt_corpus_sha256="0" * 64,
            generation_protocol_sha256=generation_protocol_sha256,
            generation_seed=809,
            attempt_budget=1,
            score_policy="raw-ancestral-cover-log-probability",
            validity_policy="exact-decoder-audit-pass-no-retry-no-repair",
        )
        protocol = TinyOverfitGateProtocol(
            dataset_manifest_sha256=base_policy.dataset_manifest_sha256,
            evaluation_policy_sha256=base_policy.policy_fingerprint,
            scaffold_matcher_contract_sha256=_SCAFFOLD_MATCHER.fingerprint,
            geometry_matcher_contract_sha256=_GEOMETRY_MATCHER.fingerprint,
            example_ids=dataset.record_ids,
            max_steps=4,
            minimum_steps=4,
            evaluation_interval=4,
            scaffold_k=1,
            match_k=1,
            seed=607,
            # Deliberately impossible finite-run thresholds: this smoke test
            # proves execution and evidence binding, not memorization.
            thresholds=TinyOverfitThresholds(
                max_training_loss=0.0,
                max_composition_nll=0.0,
                min_scaffold_recall_at_k=1.0,
                min_oracle_match_at_k=1.0,
                min_predicted_match_at_k=1.0,
                max_invalid_attempt_rate=0.0,
                min_relative_loss_reduction=1.0,
                min_relative_composition_nll_reduction=1.0,
            ),
            minimum_consecutive_passing_evaluations=1,
        )
        scientific_manifest = model.scientific_manifest(
            data_contract={
                "build_manifest_sha256": store.manifest.build_manifest_sha256,
                "dataset_fingerprint": dataset.fingerprint,
                "record_ids": list(dataset.record_ids),
                "split_manifest_sha256": split_sha256,
            }
        )
        provenance = ProvenanceRecord(
            created_utc="2026-08-29T00:00:00+00:00",
            config_fingerprint=config.fingerprint,
            dataset_hashes=(),
            catalogue_hash=None,
            package_version="0.1.0a1",
            python_version="test",
            platform="test",
            git_revision=None,
            random_seed=607,
            task="bounded-tiny-overfit-integration-smoke",
        )
        phase_seeds = {
            TrainingPhase.COMPOSITION: 811,
            TrainingPhase.COVER: 821,
            TrainingPhase.GEOMETRY: 823,
            TrainingPhase.JOINT: 827,
        }

        def sampler_for_current_phase() -> DeterministicBucketBatchSampler:
            phase = trainer.current_phase
            measure = (
                TrainingMeasure.UNIQUE_COMPOSITION
                if phase is TrainingPhase.COMPOSITION
                else TrainingMeasure.STRUCTURE
            )
            return DeterministicBucketBatchSampler(
                dataset,
                batch_size=1,
                seed=phase_seeds[phase],
                measure=measure,
            )

        active_sampler: DeterministicBucketBatchSampler | None = sampler_for_current_phase()
        observed_phases: list[TrainingPhase] = []

        def train_step(step: int) -> float:
            nonlocal active_sampler
            assert active_sampler is not None
            assert step == trainer.cursor.epoch + 1
            report = trainer.train_epoch(
                active_sampler,
                objective,
                crystallographic_audit=audit,
            )
            observed_phases.append(report.phase)
            active_sampler = (
                sampler_for_current_phase()
                if trainer.cursor.epoch < schedule.total_epochs
                else None
            )
            return report.metrics["objective/total"]

        def evaluate(step: int) -> TinyOverfitEvaluation:
            checkpoint = tmp_path / f"tiny-overfit-step-{step}.pt"
            receipt = trainer.save_checkpoint(
                checkpoint,
                scientific_config=config,
                scientific_manifest=scientific_manifest,
                provenance=provenance,
                sampler=active_sampler,
                metadata={"diagnostic_step": step},
            )
            assert receipt is not None
            generator = LearnedCrystalGenerator.from_checkpoint(
                model,
                checkpoint,
                scientific_manifest=scientific_manifest,
                expected_sha256=receipt.sha256,
            )
            was_training = model.training
            model.eval()
            try:
                with torch.no_grad(), torch.random.fork_rng(devices=[]):
                    torch.manual_seed(907)
                    losses = objective(
                        components,
                        evaluation_batch,
                        TrainingPhase.JOINT,
                    )
                    training_loss = float(
                        losses.objective(TrainingPhase.JOINT, LossWeights()).item()
                    )
                forecast = _composition_forecast(model, evaluation_batch)
            finally:
                model.train(was_training)
            step_contract = replace(
                base_policy,
                checkpoint_sha256=receipt.sha256,
                generated_attempt_corpus_sha256="0" * 64,
                composition_forecasts_sha256=composition_forecasts_sha256((forecast,)),
            )
            composition = evaluate_composition_forecasts(
                (forecast,),
                evaluation_contract=step_contract,
                bootstrap=BootstrapProtocol(samples=100, seed=919),
            )

            predicted_trace = generator.sample(
                GenerationRequest(
                    mode=GenerationMode.CSP,
                    composition=target_cover.composition,
                    samples=1,
                    seed=base_policy.generation_seed,
                )
            )[0]
            predicted_attempt = RankedAttempt[bytes]("predicted-0", 0.0, False)
            if predicted_trace.sample is not None:
                try:
                    predicted_attempt = RankedAttempt(
                        "predicted-0",
                        predicted_trace.sample.cover_sample.log_probability,
                        True,
                        _structure_payload(
                            predicted_trace.sample.decoded.cover,
                            predicted_trace.sample.decoded.lattice,
                            predicted_trace.sample.decoded.fractional_positions,
                            predicted_trace.sample.decoded.atomic_numbers,
                        ),
                    )
                except CrystalAtlasError:
                    # A structure without a certified canonical record is invalid.
                    pass
            scaffold_cases = (
                RankedRetrievalCase(
                    dataset.record_ids[0],
                    targets,
                    (predicted_attempt,),
                ),
            )
            scaffold = evaluate_scaffold_retrieval(
                scaffold_cases,
                ks=(1,),
                matcher=_fixture_scaffold_match,
                matcher_contract=_SCAFFOLD_MATCHER,
                evaluation_contract=retrieval_corpus_fingerprints(scaffold_cases).bind(
                    step_contract
                ),
                bootstrap=BootstrapProtocol(samples=100, seed=919),
            )

            try:
                _, oracle_decoded = generator.sample_geometry_given_cover(
                    target_cover,
                    seed=base_policy.generation_seed,
                )
                oracle_attempt = RankedAttempt(
                    "oracle-0",
                    0.0,
                    True,
                    _structure_payload(
                        oracle_decoded.cover,
                        oracle_decoded.lattice,
                        oracle_decoded.fractional_positions,
                        oracle_decoded.atomic_numbers,
                    ),
                )
            except CrystalAtlasError:
                # The no-retry diagnostic must preserve a failed decoder attempt.
                # Unexpected programming errors must abort the diagnostic instead
                # of being laundered into an ordinary invalid sample.
                oracle_attempt = RankedAttempt[bytes]("oracle-0", 0.0, False)
            geometry_cases = (
                GeometryDecompositionCase(
                    dataset.record_ids[0],
                    targets,
                    (oracle_attempt,),
                    (predicted_attempt,),
                ),
            )
            geometry = evaluate_geometry_decomposition(
                geometry_cases,
                ks=(1,),
                matcher=_prefixed_physical_match,
                matcher_contract=_GEOMETRY_MATCHER,
                evaluation_contract=geometry_corpus_fingerprints(geometry_cases).bind(
                    step_contract
                ),
                bootstrap=BootstrapProtocol(samples=100, seed=919),
            )
            return TinyOverfitEvaluation(
                step=step,
                training_loss=training_loss,
                composition=composition,
                scaffold=scaffold,
                geometry=geometry,
            )

        result = run_tiny_dataset_overfit_gate(
            protocol,
            evaluation_policy=base_policy,
            scaffold_matcher_contract=_SCAFFOLD_MATCHER,
            geometry_matcher_contract=_GEOMETRY_MATCHER,
            train_step=train_step,
            evaluate=evaluate,
            stop_on_pass=False,
        )

        assert observed_phases == [
            TrainingPhase.COMPOSITION,
            TrainingPhase.COVER,
            TrainingPhase.GEOMETRY,
            TrainingPhase.JOINT,
        ]
        assert [snapshot.step for snapshot in result.history] == [0, 4]
        assert result.final_snapshot.scaffold_matcher_contract_sha256 == (
            _SCAFFOLD_MATCHER.fingerprint
        )
        assert result.final_snapshot.geometry_matcher_contract_sha256 == (
            _GEOMETRY_MATCHER.fingerprint
        )
        assert not result.passed
        assert result.final_failures
