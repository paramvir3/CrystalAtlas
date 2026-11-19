from __future__ import annotations

import json
import math
from dataclasses import replace
from hashlib import sha256

import pytest

from crystalatlas.diagnostics import (
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
from crystalatlas.errors import DomainError

BOOTSTRAP = BootstrapProtocol(samples=200, confidence=0.90, seed=731)


def _phase_match(candidate: str, target: str) -> bool:
    """Test matcher that accepts equivalent settings of the same phase."""

    return candidate.partition("@")[0] == target.partition("@")[0]


MATCHER_CONTRACT = EquivalenceMatcherContract.from_callable(
    _phase_match,
    algorithm="test-phase-prefix-equivalence",
    version="1",
    cell_policy="test-string-payload-no-cell-transform",
    parameters={"delimiter": "@", "comparison": "prefix-exact"},
)
EVALUATION_CONTRACT = DiagnosticEvaluationContract(
    dataset_manifest_sha256="1" * 64,
    split_manifest_sha256="2" * 64,
    checkpoint_sha256="3" * 64,
    target_corpus_sha256="4" * 64,
    generated_attempt_corpus_sha256="b" * 64,
    generation_protocol_sha256="5" * 64,
    generation_seed=17,
    attempt_budget=4,
    score_policy="test-score-descending",
    validity_policy="test-invalid-attempts-retained",
)


def test_composition_nll_calibration_brier_and_manifest_are_exact() -> None:
    forecasts = (
        CompositionForecast.from_distribution("q-1", "A", {"A": 0.8, "B": 0.2}),
        CompositionForecast.from_distribution("q-2", "B", {"A": 0.6, "B": 0.4}),
    )
    result = evaluate_composition_forecasts(
        forecasts,
        calibration_bins=2,
        evaluation_contract=replace(
            EVALUATION_CONTRACT,
            composition_forecasts_sha256=composition_forecasts_sha256(forecasts),
        ),
        bootstrap=BOOTSTRAP,
    )

    assert result.mean_composition_nll.value == pytest.approx(
        (-math.log(0.8) - math.log(0.4)) / 2.0
    )
    assert result.corpus_nll_per_token.value == result.mean_composition_nll.value
    assert result.top_label_accuracy is not None
    assert result.top_label_accuracy.value == pytest.approx(0.5)
    assert result.mean_top_confidence is not None
    assert result.mean_top_confidence.value == pytest.approx(0.7)
    assert result.top_label_ece is not None
    assert result.top_label_ece.value == pytest.approx(0.2)
    assert result.multiclass_brier is not None
    assert result.multiclass_brier.value == pytest.approx(0.4)
    assert [item.count for item in result.calibration_bins] == [0, 2]

    manifest = json.loads(result.to_json())
    assert manifest["schema"] == "crystalatlas.composition-diagnostics.v4"
    assert manifest["evaluation_contract"]["schema"] == (
        "crystalatlas.diagnostic-evaluation-contract.v2"
    )
    assert result.forecasts_sha256 == result.evaluation_contract.composition_forecasts_sha256
    assert manifest["bootstrap"]["seed"] == 731
    assert json.dumps(result.to_manifest(), allow_nan=False)


def test_composition_sequence_nll_without_inapplicable_brier() -> None:
    forecasts = (
        CompositionForecast("q-1", math.log(0.25), token_count=2),
        CompositionForecast("q-2", math.log(0.5), token_count=1),
    )
    result = evaluate_composition_forecasts(
        forecasts,
        evaluation_contract=replace(
            EVALUATION_CONTRACT,
            composition_forecasts_sha256=composition_forecasts_sha256(forecasts),
        ),
        bootstrap=BOOTSTRAP,
    )
    assert result.mean_composition_nll.value == pytest.approx(
        (-math.log(0.25) - math.log(0.5)) / 2.0
    )
    assert result.corpus_nll_per_token.value == pytest.approx(
        (-math.log(0.25) - math.log(0.5)) / 3.0
    )
    assert result.top_label_ece is None
    assert result.multiclass_brier is None


def test_composition_metrics_refuse_partial_or_improper_forecasts() -> None:
    calibrated = CompositionForecast.from_distribution("q-1", "A", {"A": 0.7, "B": 0.3})
    uncalibrated = CompositionForecast("q-2", math.log(0.5))
    with pytest.raises(DomainError, match="every query or none"):
        evaluate_composition_forecasts(
            (calibrated, uncalibrated),
            evaluation_contract=replace(
                EVALUATION_CONTRACT,
                composition_forecasts_sha256=composition_forecasts_sha256(
                    (calibrated, uncalibrated)
                ),
            ),
            bootstrap=BOOTSTRAP,
        )
    with pytest.raises(DomainError, match="strictly positive"):
        CompositionForecast.from_distribution("q", "B", {"A": 1.0, "B": 0.0})


def test_scaffold_recall_retains_invalid_slots_polymorphs_and_ties() -> None:
    cases = (
        RankedRetrievalCase(
            "q-1",
            (
                AcceptableTarget("rocksalt", "rocksalt@origin-1"),
                AcceptableTarget("zincblende", "zincblende@origin-1"),
            ),
            # Same score: stable attempt_id puts the invalid attempt first,
            # regardless of callback emission order.
            (
                RankedAttempt("b-hit", 0.9, True, "rocksalt@origin-2"),
                RankedAttempt[str]("a-invalid", 0.9, False),
            ),
        ),
        RankedRetrievalCase(
            "q-2",
            (AcceptableTarget("rutile", "rutile@setting-1"),),
            (RankedAttempt("a-miss", 0.8, True, "anatase@setting-1"),),
        ),
        RankedRetrievalCase(
            "q-3",
            (AcceptableTarget("perovskite", "perovskite@setting-1"),),
            (),
        ),
    )
    result = evaluate_scaffold_retrieval(
        cases,
        ks=(2, 1, 2),
        matcher=_phase_match,
        matcher_contract=MATCHER_CONTRACT,
        evaluation_contract=retrieval_corpus_fingerprints(cases).bind(EVALUATION_CONTRACT),
        bootstrap=BOOTSTRAP,
    )

    one, two = result.at_k
    assert one.k == 1
    assert one.recall.value == 0.0
    assert one.valid_attempt_rate.value == pytest.approx(1.0 / 3.0)
    assert two.recall.value == pytest.approx(1.0 / 3.0)
    assert two.polymorph_coverage.value == pytest.approx(1.0 / 4.0)
    assert two.target_count == 4
    assert two.attempt_slots == 6
    assert two.valid_attempt_rate.value == pytest.approx(2.0 / 6.0)
    assert json.loads(result.to_json())["invalid_or_missing_attempts_occupy_slots"]
    retrieval_manifest = json.loads(result.to_json())
    assert retrieval_manifest["schema"] == "crystalatlas.scaffold-retrieval-diagnostics.v3"
    assert retrieval_manifest["matcher"]["fingerprint"] == MATCHER_CONTRACT.fingerprint


def test_geometry_oracle_predicted_decomposition_is_paired() -> None:
    cases = (
        GeometryDecompositionCase(
            "q-1",
            (
                AcceptableTarget("phase-a", "phase-a@reference"),
                AcceptableTarget("phase-b", "phase-b@reference"),
            ),
            (RankedAttempt("oracle", 1.0, True, "phase-b@equivalent"),),
            (RankedAttempt("selected", 1.0, True, "wrong@candidate"),),
        ),
        GeometryDecompositionCase(
            "q-2",
            (AcceptableTarget("phase-c", "phase-c@reference"),),
            (RankedAttempt("oracle", 1.0, True, "phase-c@equivalent"),),
            (RankedAttempt[str]("failed", 1.0, False),),
        ),
    )
    result = evaluate_geometry_decomposition(
        cases,
        ks=(1,),
        matcher=_phase_match,
        matcher_contract=MATCHER_CONTRACT,
        evaluation_contract=geometry_corpus_fingerprints(cases).bind(EVALUATION_CONTRACT),
        bootstrap=BOOTSTRAP,
    )
    metric = result.at_k[0]
    assert metric.oracle_scaffold_match.value == 1.0
    assert metric.predicted_scaffold_match.value == 0.0
    assert metric.delta_k.value == 1.0
    assert metric.oracle_valid_attempt_rate.value == 1.0
    assert metric.predicted_valid_attempt_rate.value == 0.5
    assert metric.delta_k.lower == 1.0
    assert metric.delta_k.upper == 1.0
    manifest = json.loads(result.to_json())
    assert manifest["schema"] == "crystalatlas.geometry-decomposition-diagnostics.v3"
    assert manifest["paired_by_query"] is True
    assert manifest["bootstrap"]["seed"] == 731


def _thresholds() -> TinyOverfitThresholds:
    return TinyOverfitThresholds(
        max_training_loss=0.02,
        max_composition_nll=0.02,
        min_scaffold_recall_at_k=1.0,
        min_oracle_match_at_k=1.0,
        min_predicted_match_at_k=1.0,
        max_invalid_attempt_rate=0.0,
        min_relative_loss_reduction=0.95,
        min_relative_composition_nll_reduction=0.95,
    )


def _tiny_policy(example_ids: tuple[str, ...]) -> DiagnosticEvaluationContract:
    empty_cases = tuple(
        RankedRetrievalCase(
            query_id,
            (AcceptableTarget("target", "target@reference"),),
            (),
        )
        for query_id in example_ids
    )
    target_digest = retrieval_corpus_fingerprints(empty_cases).target_corpus_sha256
    return replace(EVALUATION_CONTRACT, target_corpus_sha256=target_digest)


def _protocol(
    policy: DiagnosticEvaluationContract,
    *,
    max_steps: int = 4,
) -> TinyOverfitGateProtocol:
    return TinyOverfitGateProtocol(
        dataset_manifest_sha256=policy.dataset_manifest_sha256,
        evaluation_policy_sha256=policy.policy_fingerprint,
        scaffold_matcher_contract_sha256=MATCHER_CONTRACT.fingerprint,
        geometry_matcher_contract_sha256=MATCHER_CONTRACT.fingerprint,
        example_ids=("tiny-1", "tiny-2"),
        max_steps=max_steps,
        minimum_steps=2,
        evaluation_interval=1,
        scaffold_k=4,
        match_k=4,
        seed=19,
        thresholds=_thresholds(),
    )


def _tiny_evaluation(
    step: int,
    *,
    passing: bool,
    protocol: TinyOverfitGateProtocol,
    policy: DiagnosticEvaluationContract,
) -> TinyOverfitEvaluation:
    checkpoint = sha256(f"tiny-checkpoint-{step}".encode()).hexdigest()
    step_contract = replace(
        policy,
        checkpoint_sha256=checkpoint,
        generated_attempt_corpus_sha256=sha256(f"composition-{step}".encode()).hexdigest(),
    )
    forecasts = tuple(
        CompositionForecast(query_id, -0.01 if passing else -1.0)
        for query_id in protocol.example_ids
    )
    step_contract = replace(
        step_contract,
        composition_forecasts_sha256=composition_forecasts_sha256(forecasts),
    )
    composition = evaluate_composition_forecasts(
        forecasts,
        evaluation_contract=step_contract,
        bootstrap=BootstrapProtocol(samples=100, seed=step),
    )

    def attempts(query_id: str, count: int) -> tuple[RankedAttempt[str], ...]:
        if not passing:
            return tuple(
                RankedAttempt[str](f"{query_id}-invalid-{index}", float(count - index), False)
                for index in range(count)
            )
        return tuple(
            RankedAttempt(
                f"{query_id}-valid-{index}",
                float(count - index),
                True,
                "target@candidate" if index == 0 else "other@candidate",
            )
            for index in range(count)
        )

    scaffold_cases = tuple(
        RankedRetrievalCase(
            query_id,
            (AcceptableTarget("target", "target@reference"),),
            attempts(query_id, protocol.scaffold_k),
        )
        for query_id in protocol.example_ids
    )
    scaffold_contract = retrieval_corpus_fingerprints(scaffold_cases).bind(step_contract)
    scaffold = evaluate_scaffold_retrieval(
        scaffold_cases,
        ks=(protocol.scaffold_k,),
        matcher=_phase_match,
        matcher_contract=MATCHER_CONTRACT,
        evaluation_contract=scaffold_contract,
        bootstrap=BootstrapProtocol(samples=100, seed=step),
    )

    geometry_cases = tuple(
        GeometryDecompositionCase(
            query_id,
            (AcceptableTarget("target", "target@reference"),),
            attempts(f"{query_id}-oracle", protocol.match_k),
            attempts(f"{query_id}-predicted", protocol.match_k),
        )
        for query_id in protocol.example_ids
    )
    geometry_contract = geometry_corpus_fingerprints(geometry_cases).bind(step_contract)
    geometry = evaluate_geometry_decomposition(
        geometry_cases,
        ks=(protocol.match_k,),
        matcher=_phase_match,
        matcher_contract=MATCHER_CONTRACT,
        evaluation_contract=geometry_contract,
        bootstrap=BootstrapProtocol(samples=100, seed=step),
    )
    return TinyOverfitEvaluation(
        step=step,
        training_loss=0.01 if passing else 1.0,
        composition=composition,
        scaffold=scaffold,
        geometry=geometry,
    )


def test_tiny_overfit_gate_passes_only_all_predeclared_thresholds() -> None:
    train_calls: list[int] = []
    policy = _tiny_policy(("tiny-1", "tiny-2"))
    protocol = _protocol(policy)

    def train_step(step: int) -> float:
        train_calls.append(step)
        return 10.0 ** (-step)

    def evaluate(step: int) -> TinyOverfitEvaluation:
        return _tiny_evaluation(
            step,
            passing=step >= 2,
            protocol=protocol,
            policy=policy,
        )

    result = run_tiny_dataset_overfit_gate(
        protocol,
        evaluation_policy=policy,
        scaffold_matcher_contract=MATCHER_CONTRACT,
        geometry_matcher_contract=MATCHER_CONTRACT,
        train_step=train_step,
        evaluate=evaluate,
    )
    assert result.passed
    assert result.first_passing_step == 3
    assert result.relative_loss_reduction == pytest.approx(0.99)
    assert result.relative_composition_nll_reduction == pytest.approx(0.99)
    assert result.final_failures == ()
    assert train_calls == [1, 2, 3]
    assert [snapshot.step for snapshot in result.history] == [0, 1, 2, 3]
    manifest = json.loads(result.to_json())
    assert manifest["schema"] == "crystalatlas.tiny-overfit-gate.v4.result"
    assert manifest["protocol"]["seed"] == 19
    assert manifest["protocol"]["minimum_consecutive_passing_evaluations"] == 2
    assert manifest["protocol"]["thresholds"]["threshold_comparison"] == "inclusive"


def test_tiny_overfit_gate_failure_names_the_unsatisfied_metrics() -> None:
    policy = _tiny_policy(("tiny-1", "tiny-2"))
    protocol = _protocol(policy, max_steps=2)

    def evaluate(step: int) -> TinyOverfitEvaluation:
        return _tiny_evaluation(
            step,
            passing=False,
            protocol=protocol,
            policy=policy,
        )

    result = run_tiny_dataset_overfit_gate(
        protocol,
        evaluation_policy=policy,
        scaffold_matcher_contract=MATCHER_CONTRACT,
        geometry_matcher_contract=MATCHER_CONTRACT,
        train_step=lambda _: 1.0,
        evaluate=evaluate,
    )
    assert not result.passed
    assert result.first_passing_step is None
    assert "composition_nll" in result.final_failures
    assert "predicted_match_at_k" in result.final_failures
    assert "invalid_attempt_rate" in result.final_failures
    assert "relative_loss_reduction" in result.final_failures
    assert "relative_composition_nll_reduction" in result.final_failures


def test_tiny_overfit_gate_rejects_wrong_callback_step() -> None:
    policy = _tiny_policy(("tiny-1", "tiny-2"))
    protocol = _protocol(policy)

    def wrong_step(_: int) -> TinyOverfitEvaluation:
        return _tiny_evaluation(
            7,
            passing=True,
            protocol=protocol,
            policy=policy,
        )

    with pytest.raises(DomainError, match="wrong step"):
        run_tiny_dataset_overfit_gate(
            protocol,
            evaluation_policy=policy,
            scaffold_matcher_contract=MATCHER_CONTRACT,
            geometry_matcher_contract=MATCHER_CONTRACT,
            train_step=lambda _: None,
            evaluate=wrong_step,
        )
