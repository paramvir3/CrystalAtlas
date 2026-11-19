"""Adversarial checks for the scientific contracts of Step-4 diagnostics."""

from __future__ import annotations

import json
import math
import sys
from dataclasses import replace
from hashlib import sha256

import numpy as np
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
    matcher_callable_sha256,
    retrieval_corpus_fingerprints,
    run_tiny_dataset_overfit_gate,
)
from crystalatlas.errors import DomainError

BOOTSTRAP = BootstrapProtocol(samples=400, confidence=0.90, seed=20260827)


def _exact_match(candidate: str, target: str) -> bool:
    return candidate == target


_STATEFUL_MATCH_VALUE = False
_GLOBAL_MATCH_LIMIT = 0.0


def _stateful_match(candidate: str, target: str) -> bool:
    del candidate, target
    global _STATEFUL_MATCH_VALUE
    _STATEFUL_MATCH_VALUE = not _STATEFUL_MATCH_VALUE
    return _STATEFUL_MATCH_VALUE


def _limited_numeric_match(candidate: float, target: float) -> bool:
    return abs(candidate - target) <= _GLOBAL_MATCH_LIMIT


def _module_dependent_match(candidate: float, target: float) -> bool:
    return math.isclose(candidate, target)


def _genexpr_helper(value: str) -> str:
    return value


def _genexpr_match(candidate: str, target: str) -> bool:
    # The helper is referenced only inside a generator-expression code object.
    return len(candidate) == len(target) and all(
        _genexpr_helper(left) == right for left, right in zip(candidate, target, strict=True)
    )


def _nested_import_match(candidate: str, target: str) -> bool:
    def inner() -> float:
        import math as hidden

        return hidden.pi

    return candidate == target and inner() > 3.0


def _local_import_match(candidate: object, target: object) -> bool:
    from operator import eq

    return eq(candidate, target)


MATCHER_CONTRACT = EquivalenceMatcherContract.from_callable(
    _exact_match,
    algorithm="test-exact-payload-equivalence",
    version="1",
    cell_policy="test-string-payload-no-cell-transform",
    parameters={"comparison": "python-string-equality", "tolerance": 0.0},
)
GEOMETRY_MATCHER_CONTRACT = EquivalenceMatcherContract.from_callable(
    _exact_match,
    algorithm="test-exact-full-geometry-equivalence",
    version="1",
    cell_policy="test-string-payload-no-cell-transform",
    parameters={"comparison": "python-string-equality", "tolerance": 0.0},
)
EVALUATION_CONTRACT = DiagnosticEvaluationContract(
    dataset_manifest_sha256="6" * 64,
    split_manifest_sha256="7" * 64,
    checkpoint_sha256="8" * 64,
    target_corpus_sha256="9" * 64,
    generated_attempt_corpus_sha256="b" * 64,
    generation_protocol_sha256="a" * 64,
    generation_seed=29,
    attempt_budget=4,
    score_policy="test-score-descending",
    validity_policy="test-invalid-attempts-retained",
)


def test_matcher_contract_binds_tolerances_and_requires_canonical_finite_json() -> None:
    changed = EquivalenceMatcherContract.from_parameters(
        algorithm=MATCHER_CONTRACT.algorithm,
        version=MATCHER_CONTRACT.version,
        implementation_sha256=MATCHER_CONTRACT.implementation_sha256,
        cell_policy=MATCHER_CONTRACT.cell_policy,
        parameters={"comparison": "python-string-equality", "tolerance": 1.0e-6},
    )
    assert changed.fingerprint != MATCHER_CONTRACT.fingerprint
    with pytest.raises(DomainError, match="canonical JSON"):
        EquivalenceMatcherContract(
            algorithm="test",
            version="1",
            implementation_sha256="a" * 64,
            cell_policy="test",
            parameters_json='{ "tolerance": 0.0 }',
        )
    with pytest.raises(DomainError, match="non-finite"):
        EquivalenceMatcherContract.from_parameters(
            algorithm="test",
            version="1",
            implementation_sha256="a" * 64,
            cell_policy="test",
            parameters={"tolerance": float("nan")},
        )
    with pytest.raises(DomainError, match="keys must be strings"):
        EquivalenceMatcherContract.from_parameters(
            algorithm="test",
            version="1",
            implementation_sha256="a" * 64,
            cell_policy="test",
            parameters={1: "coercion-forbidden"},  # type: ignore[dict-item]
        )
    with pytest.raises(DomainError, match="immutable canonical bytes"):
        AcceptableTarget("mutable", [1])
    with pytest.raises(DomainError, match="must not perform local imports"):
        EquivalenceMatcherContract.from_callable(
            _local_import_match,
            algorithm="test-local-import-rejection",
            version="1",
            cell_policy="test",
            parameters={},
        )


def test_invalid_and_missing_generations_cannot_improve_recall_denominator() -> None:
    """Filtering invalid attempts would turn Recall@1 from zero into one-half."""

    cases = (
        RankedRetrievalCase(
            "has-late-hit",
            (AcceptableTarget("target", "correct"),),
            (
                RankedAttempt[str]("invalid-high-score", 2.0, False),
                RankedAttempt("valid-late-hit", 1.0, True, "correct"),
            ),
        ),
        RankedRetrievalCase(
            "no-attempts",
            (AcceptableTarget("target", "correct"),),
            (),
        ),
    )

    result = evaluate_scaffold_retrieval(
        cases,
        ks=(1, 2),
        matcher=_exact_match,
        matcher_contract=MATCHER_CONTRACT,
        evaluation_contract=retrieval_corpus_fingerprints(cases).bind(EVALUATION_CONTRACT),
        bootstrap=BOOTSTRAP,
    )
    at_one, at_two = result.at_k

    assert at_one.recall.value == 0.0
    assert at_one.valid_attempt_rate.value == 0.0
    assert at_one.attempt_slots == 2
    assert at_two.recall.value == pytest.approx(0.5)
    assert at_two.valid_attempt_rate.value == pytest.approx(1.0 / 4.0)
    assert at_two.attempt_slots == 4


def test_ranked_diagnostics_bind_budget_cases_and_match_decisions() -> None:
    base = RankedRetrievalCase(
        "q",
        (AcceptableTarget("target", "correct"),),
        (RankedAttempt("candidate", 1.0, True, "correct"),),
    )
    changed_score = RankedRetrievalCase(
        "q",
        (AcceptableTarget("target", "correct"),),
        (RankedAttempt("candidate", 0.5, True, "correct"),),
    )
    first = evaluate_scaffold_retrieval(
        (base,),
        ks=(1,),
        matcher=_exact_match,
        matcher_contract=MATCHER_CONTRACT,
        evaluation_contract=retrieval_corpus_fingerprints((base,)).bind(EVALUATION_CONTRACT),
        bootstrap=BOOTSTRAP,
    )
    second = evaluate_scaffold_retrieval(
        (changed_score,),
        ks=(1,),
        matcher=_exact_match,
        matcher_contract=MATCHER_CONTRACT,
        evaluation_contract=retrieval_corpus_fingerprints((changed_score,)).bind(
            EVALUATION_CONTRACT
        ),
        bootstrap=BOOTSTRAP,
    )
    assert first.cases_sha256 != second.cases_sha256
    assert first.match_decisions_sha256 == second.match_decisions_sha256

    over_budget = RankedRetrievalCase(
        "q",
        (AcceptableTarget("target", "correct"),),
        tuple(RankedAttempt(f"a-{index}", 1.0, True, "correct") for index in range(5)),
    )
    with pytest.raises(DomainError, match="exceeds.*attempt budget"):
        evaluate_scaffold_retrieval(
            (over_budget,),
            ks=(1,),
            matcher=_exact_match,
            matcher_contract=MATCHER_CONTRACT,
            evaluation_contract=retrieval_corpus_fingerprints((over_budget,)).bind(
                EVALUATION_CONTRACT
            ),
            bootstrap=BOOTSTRAP,
        )

    stateful_contract = EquivalenceMatcherContract.from_callable(
        _stateful_match,
        algorithm="test-stateful-matcher-rejection",
        version="1",
        cell_policy="test",
        parameters={},
    )
    global _STATEFUL_MATCH_VALUE
    _STATEFUL_MATCH_VALUE = False
    with pytest.raises(DomainError, match="stateful, nondeterministic, or asymmetric"):
        evaluate_scaffold_retrieval(
            (base,),
            ks=(1,),
            matcher=_stateful_match,
            matcher_contract=stateful_contract,
            evaluation_contract=retrieval_corpus_fingerprints((base,)).bind(
                EVALUATION_CONTRACT
            ),
            bootstrap=BOOTSTRAP,
        )


def test_ranked_payload_receipts_and_matcher_global_dependencies_fail_closed() -> None:
    alpha = RankedRetrievalCase(
        "q",
        (AcceptableTarget("target", "alpha"),),
        (RankedAttempt("candidate", 1.0, True, "alpha"),),
    )
    beta = RankedRetrievalCase(
        "q",
        (AcceptableTarget("target", "beta"),),
        (RankedAttempt("candidate", 1.0, True, "beta"),),
    )
    changed_attempt = RankedRetrievalCase(
        "q",
        (AcceptableTarget("target", "alpha"),),
        (RankedAttempt("candidate", 1.0, True, "beta"),),
    )
    alpha_contract = retrieval_corpus_fingerprints((alpha,)).bind(EVALUATION_CONTRACT)
    alpha_result = evaluate_scaffold_retrieval(
        (alpha,),
        ks=(1,),
        matcher=_exact_match,
        matcher_contract=MATCHER_CONTRACT,
        evaluation_contract=alpha_contract,
        bootstrap=BOOTSTRAP,
    )
    with pytest.raises(DomainError, match="target corpus differs"):
        evaluate_scaffold_retrieval(
            (beta,),
            ks=(1,),
            matcher=_exact_match,
            matcher_contract=MATCHER_CONTRACT,
            evaluation_contract=alpha_contract,
            bootstrap=BOOTSTRAP,
        )
    with pytest.raises(DomainError, match="attempt corpus differs"):
        evaluate_scaffold_retrieval(
            (changed_attempt,),
            ks=(1,),
            matcher=_exact_match,
            matcher_contract=MATCHER_CONTRACT,
            evaluation_contract=alpha_contract,
            bootstrap=BOOTSTRAP,
        )
    beta_result = evaluate_scaffold_retrieval(
        (beta,),
        ks=(1,),
        matcher=_exact_match,
        matcher_contract=MATCHER_CONTRACT,
        evaluation_contract=retrieval_corpus_fingerprints((beta,)).bind(EVALUATION_CONTRACT),
        bootstrap=BOOTSTRAP,
    )
    assert alpha_result.to_json() != beta_result.to_json()

    global _GLOBAL_MATCH_LIMIT
    _GLOBAL_MATCH_LIMIT = 0.0
    limited_contract = EquivalenceMatcherContract.from_callable(
        _limited_numeric_match,
        algorithm="test-global-limit",
        version="1",
        cell_policy="test",
        parameters={},
    )
    numeric = RankedRetrievalCase(
        "numeric",
        (AcceptableTarget("target", 0.0),),
        (RankedAttempt("candidate", 1.0, True, 0.5),),
    )
    numeric_contract = retrieval_corpus_fingerprints((numeric,)).bind(EVALUATION_CONTRACT)
    _GLOBAL_MATCH_LIMIT = 1.0
    try:
        with pytest.raises(DomainError, match="differs from its declared implementation"):
            evaluate_scaffold_retrieval(
                (numeric,),
                ks=(1,),
                matcher=_limited_numeric_match,
                matcher_contract=limited_contract,
                evaluation_contract=numeric_contract,
                bootstrap=BOOTSTRAP,
            )
    finally:
        _GLOBAL_MATCH_LIMIT = 0.0

    with pytest.raises(DomainError, match="is a module"):
        EquivalenceMatcherContract.from_callable(
            _module_dependent_match,
            algorithm="test-module-rejection",
            version="1",
            cell_policy="test",
            parameters={},
        )


def test_geometry_delta_uses_paired_query_resampling_not_two_independent_arms() -> None:
    """The attribution interval must resample each query's oracle/predicted pair."""

    hit = (RankedAttempt("hit", 1.0, True, "target"),)
    miss = (RankedAttempt("miss", 1.0, True, "other"),)
    cases = (
        GeometryDecompositionCase("oracle-only", (AcceptableTarget("t", "target"),), hit, miss),
        GeometryDecompositionCase(
            "predicted-only", (AcceptableTarget("t", "target"),), miss, hit
        ),
        GeometryDecompositionCase("both", (AcceptableTarget("t", "target"),), hit, hit),
        GeometryDecompositionCase("neither", (AcceptableTarget("t", "target"),), miss, miss),
    )

    result = evaluate_geometry_decomposition(
        cases,
        ks=(1,),
        matcher=_exact_match,
        matcher_contract=MATCHER_CONTRACT,
        evaluation_contract=geometry_corpus_fingerprints(cases).bind(EVALUATION_CONTRACT),
        bootstrap=BOOTSTRAP,
    ).at_k[0]

    assert result.oracle_scaffold_match.value == 0.5
    assert result.predicted_scaffold_match.value == 0.5
    assert result.delta_k.value == 0.0

    paired_query_deltas = np.asarray((1.0, -1.0, 0.0, 0.0))
    rng = np.random.default_rng(BOOTSTRAP.seed)
    draws = np.asarray(
        [
            paired_query_deltas[
                rng.integers(0, len(cases), size=len(cases), dtype=np.int64)
            ].mean()
            for _ in range(BOOTSTRAP.samples)
        ]
    )
    tail = (1.0 - BOOTSTRAP.confidence) / 2.0
    expected_lower, expected_upper = np.quantile(draws, (tail, 1.0 - tail), method="linear")
    assert result.delta_k.lower == pytest.approx(expected_lower)
    assert result.delta_k.upper == pytest.approx(expected_upper)


def test_calibration_boundaries_and_label_ties_are_deterministic() -> None:
    forecasts = (
        CompositionForecast(
            "lower-bin", math.log(0.1), top_confidence=0.25, top_is_correct=False
        ),
        CompositionForecast("boundary", math.log(0.5), top_confidence=0.5, top_is_correct=True),
        CompositionForecast("unit-confidence", 0.0, top_confidence=1.0, top_is_correct=True),
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
    lower, upper = result.calibration_bins
    assert lower.count == 1
    assert lower.mean_confidence == 0.25
    assert lower.accuracy == 0.0
    assert upper.count == 2
    assert upper.mean_confidence == 0.75
    assert upper.accuracy == 1.0
    assert result.top_label_ece is not None
    assert result.top_label_ece.value == pytest.approx(0.25)

    tied = CompositionForecast.from_distribution("tie", "B", {"B": 0.5, "A": 0.5})
    assert tied.top_confidence == 0.5
    assert tied.top_is_correct is False  # Unicode-first label A wins the tie.

    direct_tie = CompositionForecast(
        "direct-tie",
        math.log(0.5),
        top_confidence=0.5,
        top_is_correct=False,
        categorical_probabilities=(0.5, 0.5),
        target_index=1,
    )
    assert direct_tie.top_is_correct is False  # Supplied vector index zero wins.


def test_composition_evaluation_requires_an_explicit_frozen_corpus_pin() -> None:
    forecasts = (CompositionForecast("q-a", -0.5), CompositionForecast("q-b", -2.0))
    with pytest.raises(DomainError, match="requires a pinned forecast corpus"):
        evaluate_composition_forecasts(
            forecasts,
            evaluation_contract=EVALUATION_CONTRACT,
            bootstrap=BOOTSTRAP,
        )
    digest = composition_forecasts_sha256(forecasts)
    contract = replace(EVALUATION_CONTRACT, composition_forecasts_sha256=digest)
    result = evaluate_composition_forecasts(
        forecasts, evaluation_contract=contract, bootstrap=BOOTSTRAP
    )
    assert result.forecasts_sha256 == digest
    assert result.evaluation_contract == contract
    # Forecasts vary by checkpoint; the fixed population/generation policy does not.
    assert contract.policy_fingerprint == EVALUATION_CONTRACT.policy_fingerprint
    assert contract.fingerprint != EVALUATION_CONTRACT.fingerprint

    substitutions = (
        (replace(forecasts[0], query_id="replacement-query"), forecasts[1]),
        (replace(forecasts[0], target_log_probability=-0.001), forecasts[1]),
        (replace(forecasts[0], token_count=2), forecasts[1]),
        tuple(reversed(forecasts)),
        forecasts[:1],
        (*forecasts, CompositionForecast("extra-query", -0.1)),
    )
    for substituted in substitutions:
        with pytest.raises(DomainError, match="forecast corpus differs"):
            evaluate_composition_forecasts(
                substituted,
                evaluation_contract=contract,
                bootstrap=BOOTSTRAP,
            )


def test_composition_corpus_pin_rejects_changed_true_class_and_invalid_digest() -> None:
    original = CompositionForecast.from_distribution("q", "A", {"A": 0.5, "B": 0.5})
    substituted = CompositionForecast.from_distribution("q", "B", {"A": 0.5, "B": 0.5})
    contract = replace(
        EVALUATION_CONTRACT,
        composition_forecasts_sha256=composition_forecasts_sha256((original,)),
    )
    with pytest.raises(DomainError, match="forecast corpus differs"):
        evaluate_composition_forecasts(
            (substituted,), evaluation_contract=contract, bootstrap=BOOTSTRAP
        )
    for invalid in ("", "0" * 63, "G" * 64, 42):
        with pytest.raises(DomainError, match="composition_forecasts_sha256"):
            replace(EVALUATION_CONTRACT, composition_forecasts_sha256=invalid)


def _overfit_policy() -> DiagnosticEvaluationContract:
    empty_cases = tuple(
        RankedRetrievalCase(
            query_id,
            (AcceptableTarget("target", "target"),),
            (),
        )
        for query_id in ("tiny-a", "tiny-b")
    )
    return replace(
        EVALUATION_CONTRACT,
        target_corpus_sha256=retrieval_corpus_fingerprints(empty_cases).target_corpus_sha256,
    )


def _overfit_protocol(policy: DiagnosticEvaluationContract) -> TinyOverfitGateProtocol:
    return TinyOverfitGateProtocol(
        dataset_manifest_sha256=policy.dataset_manifest_sha256,
        evaluation_policy_sha256=policy.policy_fingerprint,
        scaffold_matcher_contract_sha256=MATCHER_CONTRACT.fingerprint,
        geometry_matcher_contract_sha256=GEOMETRY_MATCHER_CONTRACT.fingerprint,
        example_ids=("tiny-a", "tiny-b"),
        max_steps=4,
        minimum_steps=2,
        evaluation_interval=1,
        scaffold_k=2,
        match_k=2,
        seed=17,
        thresholds=TinyOverfitThresholds(
            max_training_loss=0.02,
            max_composition_nll=0.02,
            min_scaffold_recall_at_k=1.0,
            min_oracle_match_at_k=1.0,
            min_predicted_match_at_k=1.0,
            max_invalid_attempt_rate=0.0,
            min_relative_loss_reduction=0.95,
            min_relative_composition_nll_reduction=0.95,
        ),
    )


def _overfit_evaluation(
    step: int,
    *,
    passing: bool,
    policy: DiagnosticEvaluationContract,
    protocol: TinyOverfitGateProtocol,
) -> TinyOverfitEvaluation:
    checkpoint = sha256(f"rigor-checkpoint-{step}".encode()).hexdigest()
    step_contract = replace(
        policy,
        checkpoint_sha256=checkpoint,
        generated_attempt_corpus_sha256=sha256(f"rigor-{step}".encode()).hexdigest(),
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

    def attempts(prefix: str, count: int) -> tuple[RankedAttempt[str], ...]:
        if not passing:
            return tuple(
                RankedAttempt[str](f"{prefix}-invalid-{index}", float(count - index), False)
                for index in range(count)
            )
        return tuple(
            RankedAttempt(
                f"{prefix}-valid-{index}",
                float(count - index),
                True,
                "target" if index == 0 else "other",
            )
            for index in range(count)
        )

    scaffold_cases = tuple(
        RankedRetrievalCase(
            query_id,
            (AcceptableTarget("target", "target"),),
            attempts(query_id, protocol.scaffold_k),
        )
        for query_id in protocol.example_ids
    )
    scaffold = evaluate_scaffold_retrieval(
        scaffold_cases,
        ks=(protocol.scaffold_k,),
        matcher=_exact_match,
        matcher_contract=MATCHER_CONTRACT,
        evaluation_contract=retrieval_corpus_fingerprints(scaffold_cases).bind(step_contract),
        bootstrap=BootstrapProtocol(samples=100, seed=step),
    )
    geometry_cases = tuple(
        GeometryDecompositionCase(
            query_id,
            (AcceptableTarget("target", "target"),),
            attempts(f"{query_id}-oracle", protocol.match_k),
            attempts(f"{query_id}-predicted", protocol.match_k),
        )
        for query_id in protocol.example_ids
    )
    geometry = evaluate_geometry_decomposition(
        geometry_cases,
        ks=(protocol.match_k,),
        matcher=_exact_match,
        matcher_contract=GEOMETRY_MATCHER_CONTRACT,
        evaluation_contract=geometry_corpus_fingerprints(geometry_cases).bind(step_contract),
        bootstrap=BootstrapProtocol(samples=100, seed=step),
    )
    return TinyOverfitEvaluation(
        step=step,
        training_loss=0.01 if passing else 1.0,
        composition=composition,
        scaffold=scaffold,
        geometry=geometry,
    )


def test_tiny_overfit_gate_records_first_crossing_and_requires_final_persistence() -> None:
    policy = _overfit_policy()
    protocol = _overfit_protocol(policy)

    recovered = run_tiny_dataset_overfit_gate(
        protocol,
        evaluation_policy=policy,
        scaffold_matcher_contract=MATCHER_CONTRACT,
        geometry_matcher_contract=GEOMETRY_MATCHER_CONTRACT,
        train_step=lambda _step: None,
        evaluate=lambda step: _overfit_evaluation(
            step,
            passing=step in (2, 3, 4),
            policy=policy,
            protocol=protocol,
        ),
        stop_on_pass=False,
    )
    assert recovered.passed
    assert recovered.first_passing_step == 3
    assert [item.step for item in recovered.history] == [0, 1, 2, 3, 4]

    regressed = run_tiny_dataset_overfit_gate(
        protocol,
        evaluation_policy=policy,
        scaffold_matcher_contract=MATCHER_CONTRACT,
        geometry_matcher_contract=GEOMETRY_MATCHER_CONTRACT,
        train_step=lambda _step: None,
        evaluate=lambda step: _overfit_evaluation(
            step,
            passing=step == 2,
            policy=policy,
            protocol=protocol,
        ),
        stop_on_pass=False,
    )
    assert not regressed.passed
    assert regressed.first_passing_step is None
    assert "composition_nll" in regressed.final_failures

    # The result is strict canonical JSON and persists the corpus identity,
    # complete history, protocol, and terminal decision without NaN values.
    persisted = regressed.to_json()
    assert persisted == regressed.to_json()
    payload = json.loads(persisted)
    assert payload["protocol"]["dataset_manifest_sha256"] == policy.dataset_manifest_sha256
    assert [item["step"] for item in payload["history"]] == [0, 1, 2, 3, 4]
    assert payload["passed"] is False


def test_tiny_gate_rejects_budget_and_source_diagnostic_contract_mismatches() -> None:
    policy = _overfit_policy()
    narrow_policy = replace(policy, attempt_budget=1)
    narrow_protocol = _overfit_protocol(narrow_policy)
    with pytest.raises(DomainError, match="K cannot exceed"):
        run_tiny_dataset_overfit_gate(
            narrow_protocol,
            evaluation_policy=narrow_policy,
            scaffold_matcher_contract=MATCHER_CONTRACT,
            geometry_matcher_contract=GEOMETRY_MATCHER_CONTRACT,
            train_step=lambda _step: None,
            evaluate=lambda step: _overfit_evaluation(
                step,
                passing=False,
                policy=narrow_policy,
                protocol=narrow_protocol,
            ),
        )

    protocol = _overfit_protocol(policy)
    valid = _overfit_evaluation(
        0,
        passing=False,
        policy=policy,
        protocol=protocol,
    )
    wrong_checkpoint = replace(
        valid.composition,
        evaluation_contract=replace(
            valid.composition.evaluation_contract,
            checkpoint_sha256="f" * 64,
        ),
    )
    mixed = replace(valid, composition=wrong_checkpoint)
    with pytest.raises(DomainError, match="share one checkpoint"):
        run_tiny_dataset_overfit_gate(
            protocol,
            evaluation_policy=policy,
            scaffold_matcher_contract=MATCHER_CONTRACT,
            geometry_matcher_contract=GEOMETRY_MATCHER_CONTRACT,
            train_step=lambda _step: None,
            evaluate=lambda _step: mixed,
        )


def test_partial_composition_forecast_must_be_probability_consistent() -> None:
    with pytest.raises(DomainError, match="must equal the target probability"):
        CompositionForecast(
            "impossible-correct",
            math.log(0.1),
            top_confidence=0.9,
            top_is_correct=True,
        )
    with pytest.raises(DomainError, match="cannot be below target probability"):
        CompositionForecast(
            "impossible-incorrect",
            math.log(0.8),
            top_confidence=0.7,
            top_is_correct=False,
        )


def test_matcher_fingerprint_covers_globals_and_imports_in_nested_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``inspect.getclosurevars`` ignores nested code objects; the contract must not."""

    this_module = sys.modules[_genexpr_match.__module__]
    original = matcher_callable_sha256(_genexpr_match)
    monkeypatch.setattr(this_module, "_genexpr_helper", lambda value: value[::-1])
    assert matcher_callable_sha256(_genexpr_match) != original
    with pytest.raises(DomainError, match="must not perform local imports"):
        matcher_callable_sha256(_nested_import_match)
