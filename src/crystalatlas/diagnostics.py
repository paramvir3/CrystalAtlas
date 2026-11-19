"""Fail-closed diagnostics for the three-factor crystal generator.

The functions in this module are deliberately independent of PyTorch and of a
particular structure matcher.  Callers supply a certified matcher for scaffold
or structure equivalence.  This matters because an acceptable answer is a set
of polymorphs/equivalent canonical presentations, not necessarily one raw cover
object.

All ranked metrics use a frozen attempt budget.  Invalid attempts and missing
attempts therefore occupy top-``K`` slots; filtering them before evaluation is
forbidden.  Score ties are resolved by the UTF-8/Unicode ordering of the stable
``attempt_id``.  Bootstrap intervals resample queries (the experimental unit),
use an explicit seed, and use paired samples for oracle-versus-predicted
comparisons.
"""

from __future__ import annotations

import dis
import inspect
import json
import marshal
import math
import re
import sys
import types
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path
from statistics import NormalDist
from typing import Generic, TypeAlias, TypeVar

import numpy as np
import numpy.typing as npt

from .errors import DomainError

_COMPOSITION_SCHEMA = "crystalatlas.composition-diagnostics.v4"
_SCAFFOLD_SCHEMA = "crystalatlas.scaffold-retrieval-diagnostics.v3"
_GEOMETRY_SCHEMA = "crystalatlas.geometry-decomposition-diagnostics.v3"
_OVERFIT_SCHEMA = "crystalatlas.tiny-overfit-gate.v4"
_MATCHER_CONTRACT_SCHEMA = "crystalatlas.equivalence-matcher-contract.v1"
_EVALUATION_CONTRACT_SCHEMA = "crystalatlas.diagnostic-evaluation-contract.v2"
_TIE_POLICY = "score_desc_then_attempt_id_unicode_asc"
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")

T = TypeVar("T")
FloatArray: TypeAlias = npt.NDArray[np.float64]
IntArray: TypeAlias = npt.NDArray[np.int64]


def _require_identifier(value: str, name: str) -> None:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise DomainError(f"{name} must be a nonempty string without NUL bytes")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise DomainError(f"{name} must be valid UTF-8 text") from exc


def _require_probability(value: float, name: str) -> None:
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise DomainError(f"{name} must be finite and lie in [0, 1]")


def _json_text(payload: Mapping[str, object]) -> str:
    """Return strict, canonical JSON (NaN and infinity are refused)."""

    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _strict_json_value(value: object, *, path: str = "parameters") -> object:
    """Validate without Python-to-JSON type coercion, then deep-copy."""

    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str):
        try:
            value.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise DomainError(f"{path} contains invalid UTF-8 text") from exc
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise DomainError(f"{path} contains a non-finite float")
        return value
    if isinstance(value, list):
        return [
            _strict_json_value(child, path=f"{path}[{index}]")
            for index, child in enumerate(value)
        ]
    if isinstance(value, Mapping):
        normalized: dict[str, object] = {}
        for key, child in value.items():
            if not isinstance(key, str):
                raise DomainError(f"{path} mapping keys must be strings")
            _require_identifier(key, f"{path} mapping key")
            normalized[key] = _strict_json_value(child, path=f"{path}.{key}")
        return normalized
    raise DomainError(f"{path} contains unsupported non-JSON type {type(value).__qualname__}")


def _identifier_fingerprint(identifiers: Sequence[str]) -> str:
    payload = _json_text({"ordered_query_ids": list(identifiers)}).encode("utf-8")
    return sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class EquivalenceMatcherContract:
    """Reproducible identity of the equivalence relation used by a metric.

    The callable itself is intentionally not serialized.  Its frozen algorithm
    name/version, implementation digest, cell policy, and strict JSON
    parameters (including every numerical tolerance) are.  Changing any of
    them therefore changes the metric manifest and fingerprint.
    """

    algorithm: str
    version: str
    implementation_sha256: str
    cell_policy: str
    parameters_json: str

    def __post_init__(self) -> None:
        _require_identifier(self.algorithm, "matcher algorithm")
        _require_identifier(self.version, "matcher version")
        _require_identifier(self.cell_policy, "matcher cell policy")
        if _SHA256_PATTERN.fullmatch(self.implementation_sha256) is None:
            raise DomainError("matcher implementation fingerprint must be lowercase SHA-256")
        if not isinstance(self.parameters_json, str):
            raise TypeError("matcher parameters_json must be text")
        try:
            parameters = json.loads(
                self.parameters_json,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError(f"nonfinite JSON constant {value}")
                ),
            )
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise DomainError("matcher parameters must be strict JSON") from exc
        if not isinstance(parameters, dict):
            raise DomainError("matcher parameters must be a JSON object")
        canonical_value = _strict_json_value(parameters)
        assert isinstance(canonical_value, dict)
        canonical = _json_text(canonical_value)
        if canonical != self.parameters_json:
            raise DomainError("matcher parameters_json must use canonical JSON encoding")

    @classmethod
    def from_parameters(
        cls,
        *,
        algorithm: str,
        version: str,
        implementation_sha256: str,
        cell_policy: str,
        parameters: Mapping[str, object],
    ) -> EquivalenceMatcherContract:
        if not isinstance(parameters, Mapping):
            raise TypeError("matcher parameters must be a mapping")
        try:
            normalized = _strict_json_value(parameters)
            assert isinstance(normalized, dict)
            parameters_json = _json_text(normalized)
        except DomainError:
            raise
        except (TypeError, ValueError) as exc:
            raise DomainError("matcher parameters must be finite strict-JSON values") from exc
        return cls(
            algorithm=algorithm,
            version=version,
            implementation_sha256=implementation_sha256,
            cell_policy=cell_policy,
            parameters_json=parameters_json,
        )

    @classmethod
    def from_callable(
        cls,
        matcher: Callable[..., bool],
        *,
        algorithm: str,
        version: str,
        cell_policy: str,
        parameters: Mapping[str, object],
    ) -> EquivalenceMatcherContract:
        return cls.from_parameters(
            algorithm=algorithm,
            version=version,
            implementation_sha256=matcher_callable_sha256(matcher),
            cell_policy=cell_policy,
            parameters=parameters,
        )

    @property
    def fingerprint(self) -> str:
        return sha256(_json_text(self._payload()).encode("utf-8")).hexdigest()

    def _payload(self) -> dict[str, object]:
        return {
            "algorithm": self.algorithm,
            "cell_policy": self.cell_policy,
            "implementation_sha256": self.implementation_sha256,
            "parameters": json.loads(self.parameters_json),
            "schema": _MATCHER_CONTRACT_SCHEMA,
            "version": self.version,
        }

    def to_manifest(self) -> dict[str, object]:
        payload = self._payload()
        payload["fingerprint"] = self.fingerprint
        return payload


def _module_artifact_identity(value: object) -> dict[str, object]:
    module = inspect.getmodule(value)
    module_name = getattr(value, "__module__", None)
    qualname = getattr(value, "__qualname__", None)
    if not isinstance(module_name, str) or not isinstance(qualname, str):
        raise DomainError("matcher dependency lacks a stable module and qualified name")
    payload: dict[str, object] = {
        "module": module_name,
        "qualname": qualname,
    }
    module_file = getattr(module, "__file__", None)
    if isinstance(module_file, str):
        path = Path(module_file)
        try:
            payload["module_file_sha256"] = sha256(path.read_bytes()).hexdigest()
        except OSError as exc:
            raise DomainError(
                f"cannot read matcher dependency module artifact {module_file!r}"
            ) from exc
    module_version = getattr(module, "__version__", None)
    if module_version is not None:
        payload["module_version"] = str(module_version)
    return payload


def _matcher_dependency_identity(
    value: object,
    *,
    name: str,
    active_functions: set[tuple[str, str]],
) -> object:
    if value is None or isinstance(value, (bool, str, int, float)):
        normalized = _strict_json_value(value, path=f"matcher global {name}")
        return {"kind": "strict_json", "value": normalized}

    if inspect.isfunction(value):
        return {
            "kind": "python_function",
            "identity": _matcher_function_payload(value, active_functions=active_functions),
        }
    if inspect.isbuiltin(value):
        return {
            "kind": "module_artifact_symbol",
            "identity": _module_artifact_identity(value),
        }
    if inspect.isclass(value):
        raise DomainError(
            f"matcher global {name!r} is a class with mutable runtime state; "
            "bind a verified dependency receipt outside this execution hash"
        )
    if inspect.ismodule(value):
        raise DomainError(
            f"matcher global {name!r} is a module; import and reference the exact "
            "function or class so its artifact can be fingerprinted"
        )
    raise DomainError(
        f"matcher global {name!r} has unsupported reproducibility type "
        f"{type(value).__qualname__}"
    )


def _code_objects(code: types.CodeType) -> Iterator[types.CodeType]:
    """Yield ``code`` and every nested code object (comprehensions, lambdas, defs)."""

    yield code
    for constant in code.co_consts:
        if isinstance(constant, types.CodeType):
            yield from _code_objects(constant)


def _matcher_function_payload(
    matcher: Callable[..., object],
    *,
    active_functions: set[tuple[str, str]],
) -> dict[str, object]:
    if not inspect.isfunction(matcher):
        raise TypeError("certified matcher and its Python helpers must be functions")
    if matcher.__closure__:
        raise DomainError("certified matcher must not capture mutable closure state")
    # ``inspect.getclosurevars`` reads only the outer code object.  Generator
    # expressions, lambdas, and nested functions keep their own instructions,
    # so imports and global references are collected from every nested level.
    nested_globals: set[str] = set()
    for code in _code_objects(matcher.__code__):
        for instruction in dis.get_instructions(code):
            if instruction.opname in {"IMPORT_NAME", "IMPORT_FROM"}:
                raise DomainError(
                    "certified matcher functions must not perform local imports; "
                    "bind an exact global helper or a verified dependency receipt"
                )
            if code is not matcher.__code__ and instruction.opname in {
                "LOAD_GLOBAL",
                "LOAD_NAME",
            }:
                nested_globals.add(str(instruction.argval))
    key = (matcher.__module__, matcher.__qualname__)
    if key in active_functions:
        return {"recursive_reference": list(key)}
    active_functions.add(key)
    try:
        closure = inspect.getclosurevars(matcher)
        if closure.nonlocals:
            raise DomainError("certified matcher must not capture nonlocal state")
        referenced = dict(closure.globals)
        for name in nested_globals:
            if name in matcher.__globals__:
                referenced[name] = matcher.__globals__[name]
        globals_payload = {
            name: _matcher_dependency_identity(
                value,
                name=name,
                active_functions=active_functions,
            )
            for name, value in sorted(referenced.items())
        }
        defaults = _strict_json_value(list(matcher.__defaults__ or ()), path="matcher defaults")
        kwdefaults = _strict_json_value(
            dict(matcher.__kwdefaults__ or {}), path="matcher keyword defaults"
        )
        return {
            "code_sha256": sha256(marshal.dumps(matcher.__code__)).hexdigest(),
            "defaults": defaults,
            "global_dependencies": globals_payload,
            "keyword_defaults": kwdefaults,
            "module": matcher.__module__,
            "python_implementation": sys.implementation.name,
            "python_version": list(sys.version_info[:3]),
            "qualname": matcher.__qualname__,
        }
    finally:
        active_functions.remove(key)


def matcher_callable_sha256(matcher: Callable[..., bool]) -> str:
    """Fingerprint a closure-free matcher and every referenced global helper.

    Module globals are rejected: callers must import the exact function or
    class used by a matcher.  This prevents a small wrapper from appearing
    unchanged while its tolerance, helper code, or binary dependency changes.
    """

    payload = _matcher_function_payload(matcher, active_functions=set())
    return sha256(_json_text(payload).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class DiagnosticEvaluationContract:
    """Externally anchored population and generation protocol for ranked metrics."""

    dataset_manifest_sha256: str
    split_manifest_sha256: str
    checkpoint_sha256: str
    target_corpus_sha256: str
    generated_attempt_corpus_sha256: str
    generation_protocol_sha256: str
    generation_seed: int
    attempt_budget: int
    score_policy: str
    validity_policy: str
    composition_forecasts_sha256: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "dataset_manifest_sha256",
            "split_manifest_sha256",
            "checkpoint_sha256",
            "target_corpus_sha256",
            "generated_attempt_corpus_sha256",
            "generation_protocol_sha256",
        ):
            if _SHA256_PATTERN.fullmatch(getattr(self, name)) is None:
                raise DomainError(f"{name} must be a lowercase SHA-256 digest")
        if (
            isinstance(self.generation_seed, bool)
            or not isinstance(self.generation_seed, int)
            or not 0 <= self.generation_seed < 2**64
        ):
            raise DomainError("generation_seed must be an unsigned 64-bit integer")
        if (
            isinstance(self.attempt_budget, bool)
            or not isinstance(self.attempt_budget, int)
            or self.attempt_budget < 1
        ):
            raise DomainError("attempt_budget must be a positive integer")
        _require_identifier(self.score_policy, "score policy")
        _require_identifier(self.validity_policy, "validity policy")
        if self.composition_forecasts_sha256 is not None and (
            not isinstance(self.composition_forecasts_sha256, str)
            or _SHA256_PATTERN.fullmatch(self.composition_forecasts_sha256) is None
        ):
            raise DomainError("composition_forecasts_sha256 must be a lowercase SHA-256 digest")

    @property
    def fingerprint(self) -> str:
        return sha256(_json_text(self._payload()).encode("utf-8")).hexdigest()

    @property
    def policy_fingerprint(self) -> str:
        """Identity of fields that remain invariant across training checkpoints."""

        return sha256(_json_text(self._policy_payload()).encode("utf-8")).hexdigest()

    def _policy_payload(self) -> dict[str, object]:
        return {
            "attempt_budget": self.attempt_budget,
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "generation_protocol_sha256": self.generation_protocol_sha256,
            "generation_seed": self.generation_seed,
            "schema": "crystalatlas.diagnostic-evaluation-policy.v1",
            "score_policy": self.score_policy,
            "split_manifest_sha256": self.split_manifest_sha256,
            "target_corpus_sha256": self.target_corpus_sha256,
            "validity_policy": self.validity_policy,
        }

    def _payload(self) -> dict[str, object]:
        return {
            "attempt_budget": self.attempt_budget,
            "checkpoint_sha256": self.checkpoint_sha256,
            "composition_forecasts_sha256": self.composition_forecasts_sha256,
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "generated_attempt_corpus_sha256": self.generated_attempt_corpus_sha256,
            "generation_protocol_sha256": self.generation_protocol_sha256,
            "generation_seed": self.generation_seed,
            "schema": _EVALUATION_CONTRACT_SCHEMA,
            "score_policy": self.score_policy,
            "split_manifest_sha256": self.split_manifest_sha256,
            "target_corpus_sha256": self.target_corpus_sha256,
            "validity_policy": self.validity_policy,
        }

    def to_manifest(self) -> dict[str, object]:
        payload = self._payload()
        payload["fingerprint"] = self.fingerprint
        payload["policy_fingerprint"] = self.policy_fingerprint
        return payload


@dataclass(frozen=True, slots=True)
class DiagnosticCorpusFingerprints:
    """Digests of the exact target and generated-attempt payload corpora."""

    target_corpus_sha256: str
    generated_attempt_corpus_sha256: str

    def __post_init__(self) -> None:
        for name in ("target_corpus_sha256", "generated_attempt_corpus_sha256"):
            if _SHA256_PATTERN.fullmatch(getattr(self, name)) is None:
                raise DomainError(f"{name} must be a lowercase SHA-256 digest")

    def bind(self, contract: DiagnosticEvaluationContract) -> DiagnosticEvaluationContract:
        if not isinstance(contract, DiagnosticEvaluationContract):
            raise TypeError("contract must be a DiagnosticEvaluationContract")
        return replace(
            contract,
            target_corpus_sha256=self.target_corpus_sha256,
            generated_attempt_corpus_sha256=self.generated_attempt_corpus_sha256,
        )


@dataclass(frozen=True, slots=True)
class BootstrapProtocol:
    """Frozen percentile-bootstrap protocol with query-level resampling."""

    samples: int = 10_000
    confidence: float = 0.95
    seed: int = 0

    def __post_init__(self) -> None:
        if (
            isinstance(self.samples, bool)
            or not isinstance(self.samples, int)
            or self.samples < 100
        ):
            raise DomainError("bootstrap samples must be an integer of at least 100")
        if not math.isfinite(self.confidence) or not 0.0 < self.confidence < 1.0:
            raise DomainError("bootstrap confidence must lie strictly between zero and one")
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or not 0 <= self.seed < 2**64
        ):
            raise DomainError("bootstrap seed must be an unsigned 64-bit integer")

    def to_manifest(self) -> dict[str, object]:
        return {
            "bit_generator": "PCG64",
            "confidence": self.confidence,
            "interval": "percentile",
            "quantile_method": "linear",
            "resampling_unit": "query",
            "samples": self.samples,
            "seed": self.seed,
        }


def per_group_calibration(
    probabilities: npt.ArrayLike,
    observed: Sequence[int],
    *,
    group_numbers: Sequence[int],
) -> dict[int, dict[str, float | int | None]]:
    """Per-space-group calibration of normalized group forecasts (review m1).

    ``probabilities[i, g]`` is the forecast for query ``i`` and column ``g``,
    which names ``group_numbers[g]``; ``observed[i]`` is the column of the true
    group.  For each group the report gives the observed frequency, the mean
    forecast probability (equal in expectation under calibration), and the mean
    negative log-likelihood on the queries where it was observed.
    """

    table = np.asarray(probabilities, dtype=np.float64)
    if table.ndim != 2 or table.shape[1] != len(group_numbers) or not len(table):
        raise DomainError("probabilities must be a nonempty [queries, groups] table")
    if np.any(~np.isfinite(table)) or np.any(table < 0.0) or not np.allclose(
        table.sum(axis=1), 1.0, atol=1e-12, rtol=0.0
    ):
        raise DomainError("each forecast row must be a normalized probability vector")
    labels = np.asarray(observed, dtype=np.int64)
    if labels.shape != (len(table),) or np.any(labels < 0) or np.any(labels >= table.shape[1]):
        raise DomainError("observed labels must index the forecast columns")
    report: dict[int, dict[str, float | int | None]] = {}
    for column, number in enumerate(group_numbers):
        hits = labels == column
        report[int(number)] = {
            "observed": int(hits.sum()),
            "observed_frequency": float(hits.mean()),
            "mean_predicted": float(table[:, column].mean()),
            "nll_when_observed": (
                float(-np.mean(np.log(table[hits, column]))) if hits.any() else None
            ),
        }
    return report


def paired_discordance_sample_size(
    *,
    effect: float,
    discordant_fraction: float,
    alpha: float = 0.05,
    power: float = 0.8,
) -> int:
    """Queries needed to detect a paired difference ``effect`` (review m7).

    For paired binary outcomes, such as oracle-versus-predicted Match@K on the
    same queries, the McNemar-type normal approximation gives
    ``n = (z_{1-alpha/2} + z_{power})^2 p_d / effect^2``, where ``p_d`` is the
    fraction of discordant pairs.  The difference of paired proportions cannot
    exceed ``p_d``.  Fix n and the number of seeds before any gated run.
    """

    for name, value in (("alpha", alpha), ("power", power)):
        if not math.isfinite(value) or not 0.0 < value < 1.0:
            raise DomainError(f"{name} must lie strictly between 0 and 1")
    if not math.isfinite(discordant_fraction) or not 0.0 < discordant_fraction <= 1.0:
        raise DomainError("discordant_fraction must lie in (0, 1]")
    if not math.isfinite(effect) or not 0.0 < abs(effect) <= discordant_fraction:
        raise DomainError("effect must be nonzero and at most the discordant fraction")
    quantile = NormalDist().inv_cdf
    z = quantile(1.0 - alpha / 2.0) + quantile(power)
    return math.ceil(z * z * discordant_fraction / (effect * effect))


def _select_bootstrap(protocol: BootstrapProtocol | None) -> BootstrapProtocol:
    if protocol is None:
        return BootstrapProtocol()
    if not isinstance(protocol, BootstrapProtocol):
        raise TypeError("bootstrap must be a BootstrapProtocol")
    return protocol


@dataclass(frozen=True, slots=True)
class MetricEstimate:
    """Point estimate and percentile confidence interval."""

    value: float
    lower: float
    upper: float

    def __post_init__(self) -> None:
        if any(not math.isfinite(value) for value in (self.value, self.lower, self.upper)):
            raise DomainError("metric estimates and bounds must be finite")
        if self.lower > self.upper:
            raise DomainError("metric lower bound cannot exceed its upper bound")

    def to_manifest(self) -> dict[str, object]:
        return {"estimate": self.value, "lower": self.lower, "upper": self.upper}


def _estimate(
    value: float, bootstrap_values: Sequence[float], protocol: BootstrapProtocol
) -> MetricEstimate:
    values = np.asarray(bootstrap_values, dtype=np.float64)
    if values.shape != (protocol.samples,) or np.any(~np.isfinite(values)):
        raise DomainError("bootstrap produced nonfinite values or the wrong sample count")
    tail = (1.0 - protocol.confidence) / 2.0
    bounds = np.asarray(
        np.quantile(values, (tail, 1.0 - tail), method="linear"),
        dtype=np.float64,
    )
    if bounds.shape != (2,):
        raise DomainError("bootstrap quantile calculation returned the wrong shape")
    return MetricEstimate(float(value), float(bounds[0]), float(bounds[1]))


def _bootstrap_indices(count: int, protocol: BootstrapProtocol) -> Iterator[IntArray]:
    if count <= 0:
        raise DomainError("diagnostics require at least one query")
    rng = np.random.Generator(np.random.PCG64(protocol.seed))
    # Stream samples: materializing an N x 10,000 matrix is unreasonable for a
    # production corpus.  Reinitializing this generator with the same protocol
    # produces the identical draws used to pair arms and K values.
    for _ in range(protocol.samples):
        yield rng.integers(0, count, size=count, dtype=np.int64)


@dataclass(frozen=True, slots=True)
class CompositionForecast:
    """One composition-level proper-scoring observation.

    ``target_log_probability`` is ``log p(c_true)`` for the complete reduced
    composition, not a conveniently selected token.  Top-label calibration is
    optional because exhaustive composition probabilities may be unavailable
    for an autoregressive grammar.  A multiclass Brier score is reported only
    when the complete categorical distribution is supplied.
    """

    query_id: str
    target_log_probability: float
    token_count: int = 1
    top_confidence: float | None = None
    top_is_correct: bool | None = None
    categorical_probabilities: tuple[float, ...] | None = None
    target_index: int | None = None

    def __post_init__(self) -> None:
        _require_identifier(self.query_id, "query_id")
        if (
            not math.isfinite(self.target_log_probability)
            or self.target_log_probability > 1.0e-12
        ):
            raise DomainError("target log-probability must be finite and nonpositive")
        if (
            isinstance(self.token_count, bool)
            or not isinstance(self.token_count, int)
            or self.token_count <= 0
        ):
            raise DomainError("composition token_count must be a positive integer")
        if (self.top_confidence is None) != (self.top_is_correct is None):
            raise DomainError("top confidence and correctness must be supplied together")
        if self.top_confidence is not None:
            _require_probability(self.top_confidence, "top confidence")
            if not isinstance(self.top_is_correct, bool):
                raise DomainError("top-label correctness must be Boolean")
            target_probability = math.exp(self.target_log_probability)
            if self.top_is_correct and not math.isclose(
                self.top_confidence,
                target_probability,
                rel_tol=1.0e-10,
                abs_tol=1.0e-12,
            ):
                raise DomainError(
                    "a correct top-label confidence must equal the target probability"
                )
            if not self.top_is_correct and self.top_confidence < target_probability:
                raise DomainError(
                    "an incorrect top-label confidence cannot be below target probability"
                )

        probabilities = self.categorical_probabilities
        if (probabilities is None) != (self.target_index is None):
            raise DomainError("categorical probabilities and target index must coexist")
        if probabilities is None:
            return
        values = tuple(float(value) for value in probabilities)
        object.__setattr__(self, "categorical_probabilities", values)
        if not values or any(not math.isfinite(value) or value < 0.0 for value in values):
            raise DomainError("categorical probabilities must be nonempty and nonnegative")
        if not math.isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=1.0e-10):
            raise DomainError("categorical probabilities must sum to one")
        if (
            isinstance(self.target_index, bool)
            or not isinstance(self.target_index, int)
            or not 0 <= self.target_index < len(values)
        ):
            raise DomainError("target index lies outside the categorical distribution")
        target_probability = values[self.target_index]
        if target_probability <= 0.0:
            raise DomainError("a zero target probability has infinite, non-JSON NLL")
        if not math.isclose(
            math.log(target_probability),
            self.target_log_probability,
            rel_tol=1.0e-10,
            abs_tol=1.0e-12,
        ):
            raise DomainError("target log-probability disagrees with categorical probabilities")
        if self.top_confidence is None:
            raise DomainError("a full categorical forecast must include top-label calibration")
        maximum = max(values)
        top_index = values.index(maximum)  # stable class-index tie rule
        if not math.isclose(self.top_confidence, maximum, rel_tol=0.0, abs_tol=1.0e-12):
            raise DomainError("top confidence disagrees with categorical probabilities")
        if self.top_is_correct != (top_index == self.target_index):
            raise DomainError("top correctness disagrees with categorical probabilities")

    @classmethod
    def from_distribution(
        cls,
        query_id: str,
        target_label: str,
        probabilities: Mapping[str, float],
        *,
        token_count: int = 1,
    ) -> CompositionForecast:
        """Construct from a complete categorical law.

        Labels are sorted before converting to a vector; tied top probabilities
        are therefore resolved lexicographically and reproducibly.
        """

        _require_identifier(target_label, "target_label")
        if not probabilities:
            raise DomainError("the categorical composition law cannot be empty")
        items = sorted(probabilities.items(), key=lambda item: item[0])
        for label, probability in items:
            _require_identifier(label, "composition label")
            _require_probability(float(probability), "composition probability")
        labels = tuple(label for label, _ in items)
        if target_label not in probabilities:
            raise DomainError("target composition is absent from the categorical law")
        values = tuple(float(probability) for _, probability in items)
        if not math.isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=1.0e-10):
            raise DomainError("categorical composition probabilities must sum to one")
        target_index = labels.index(target_label)
        target_probability = values[target_index]
        if target_probability <= 0.0:
            raise DomainError("target composition probability must be strictly positive")
        top_index = min(range(len(values)), key=lambda index: (-values[index], labels[index]))
        return cls(
            query_id=query_id,
            target_log_probability=math.log(target_probability),
            token_count=token_count,
            top_confidence=values[top_index],
            top_is_correct=top_index == target_index,
            categorical_probabilities=values,
            target_index=target_index,
        )


@dataclass(frozen=True, slots=True)
class CalibrationBin:
    index: int
    lower: float
    upper: float
    upper_inclusive: bool
    count: int
    mean_confidence: float | None
    accuracy: float | None

    def __post_init__(self) -> None:
        if isinstance(self.index, bool) or not isinstance(self.index, int) or self.index < 0:
            raise DomainError("calibration-bin index must be a nonnegative integer")
        if (
            not math.isfinite(self.lower)
            or not math.isfinite(self.upper)
            or not 0.0 <= self.lower < self.upper <= 1.0
        ):
            raise DomainError("calibration-bin endpoints must form an interval in [0, 1]")
        if not isinstance(self.upper_inclusive, bool):
            raise DomainError("calibration-bin closure flag must be Boolean")
        if isinstance(self.count, bool) or not isinstance(self.count, int) or self.count < 0:
            raise DomainError("calibration-bin count must be a nonnegative integer")
        if self.count == 0:
            if self.mean_confidence is not None or self.accuracy is not None:
                raise DomainError("an empty calibration bin cannot have empirical statistics")
        elif self.mean_confidence is None or self.accuracy is None:
            raise DomainError("a populated calibration bin requires empirical statistics")
        else:
            _require_probability(self.mean_confidence, "bin mean confidence")
            _require_probability(self.accuracy, "bin accuracy")

    def to_manifest(self) -> dict[str, object]:
        return {
            "accuracy": self.accuracy,
            "count": self.count,
            "index": self.index,
            "lower": self.lower,
            "mean_confidence": self.mean_confidence,
            "upper": self.upper,
            "upper_inclusive": self.upper_inclusive,
        }


def _calibration_bins(
    confidence: FloatArray, correct: FloatArray, count: int
) -> tuple[CalibrationBin, ...]:
    indices = np.minimum(np.floor(confidence * count).astype(np.int64), count - 1)
    result: list[CalibrationBin] = []
    for index in range(count):
        selected = indices == index
        bin_count = int(np.count_nonzero(selected))
        result.append(
            CalibrationBin(
                index=index,
                lower=index / count,
                upper=(index + 1) / count,
                upper_inclusive=index == count - 1,
                count=bin_count,
                mean_confidence=(float(confidence[selected].mean()) if bin_count else None),
                accuracy=(float(correct[selected].mean()) if bin_count else None),
            )
        )
    return tuple(result)


def _ece(confidence: FloatArray, correct: FloatArray, bin_count: int) -> float:
    bins = _calibration_bins(confidence, correct, bin_count)
    total = confidence.size
    return float(
        sum(
            item.count / total * abs(item.accuracy - item.mean_confidence)
            for item in bins
            if item.count and item.accuracy is not None and item.mean_confidence is not None
        )
    )


@dataclass(frozen=True, slots=True)
class CompositionDiagnostics:
    observation_count: int
    total_token_count: int
    query_ids_sha256: str
    calibration_bin_count: int
    mean_composition_nll: MetricEstimate
    corpus_nll_per_token: MetricEstimate
    top_label_accuracy: MetricEstimate | None
    mean_top_confidence: MetricEstimate | None
    top_label_ece: MetricEstimate | None
    multiclass_brier: MetricEstimate | None
    calibration_bins: tuple[CalibrationBin, ...]
    bootstrap: BootstrapProtocol
    evaluation_contract: DiagnosticEvaluationContract
    forecasts_sha256: str

    def to_manifest(self) -> dict[str, object]:
        return {
            "bootstrap": self.bootstrap.to_manifest(),
            "calibration_bin_count": self.calibration_bin_count,
            "calibration_binning": "equal_width_left_closed_right_open_last_closed",
            "calibration_bins": [item.to_manifest() for item in self.calibration_bins],
            "corpus_nll_per_token": self.corpus_nll_per_token.to_manifest(),
            "evaluation_contract": self.evaluation_contract.to_manifest(),
            "forecasts_sha256": self.forecasts_sha256,
            "mean_composition_nll": self.mean_composition_nll.to_manifest(),
            "mean_top_confidence": _optional_estimate_manifest(self.mean_top_confidence),
            "multiclass_brier_definition": "mean(sum_j((p_j-y_j)^2))",
            "multiclass_brier": _optional_estimate_manifest(self.multiclass_brier),
            "nll_definition": "mean(-log(p_complete_true_composition))",
            "observation_count": self.observation_count,
            "query_ids_sha256": self.query_ids_sha256,
            "schema": _COMPOSITION_SCHEMA,
            "top_label_tie_policy": (
                "lowest_supplied_class_index; from_distribution sorts labels_unicode_asc"
            ),
            "top_label_accuracy": _optional_estimate_manifest(self.top_label_accuracy),
            "top_label_ece": _optional_estimate_manifest(self.top_label_ece),
            "total_token_count": self.total_token_count,
        }

    def to_json(self) -> str:
        return _json_text(self.to_manifest())


def _optional_estimate_manifest(
    value: MetricEstimate | None,
) -> dict[str, object] | None:
    return None if value is None else value.to_manifest()


def composition_forecasts_sha256(forecasts: Sequence[CompositionForecast]) -> str:
    """Bind every ordered query, score and calibration value in a forecast corpus.

    Freeze this digest in the evaluation contract when the checkpoint's
    complete forecast artifact is produced. The evaluator requires that pin;
    recomputing it after selecting or replacing observations defeats the trust
    boundary, just as replacing any externally trusted corpus fingerprint does.
    """

    observations = tuple(forecasts)
    if not observations or any(
        not isinstance(item, CompositionForecast) for item in observations
    ):
        raise DomainError("composition corpus requires nonempty CompositionForecast values")
    if len({item.query_id for item in observations}) != len(observations):
        raise DomainError("composition diagnostic query identifiers must be unique")
    return sha256(
        _json_text(
            {
                "forecasts": [
                    {
                        "categorical_probabilities_hex": (
                            [float(value).hex() for value in item.categorical_probabilities]
                            if item.categorical_probabilities is not None
                            else None
                        ),
                        "query_id": item.query_id,
                        "target_index": item.target_index,
                        "target_log_probability_hex": float(item.target_log_probability).hex(),
                        "token_count": item.token_count,
                        "top_confidence_hex": (
                            float(item.top_confidence).hex()
                            if item.top_confidence is not None
                            else None
                        ),
                        "top_is_correct": item.top_is_correct,
                    }
                    for item in observations
                ]
            }
        ).encode("utf-8")
    ).hexdigest()


def evaluate_composition_forecasts(
    forecasts: Sequence[CompositionForecast],
    *,
    calibration_bins: int = 15,
    evaluation_contract: DiagnosticEvaluationContract,
    bootstrap: BootstrapProtocol | None = None,
) -> CompositionDiagnostics:
    """Evaluate composition NLL, top-label calibration, and Brier score.

    Calibration or full-class probabilities must be available for every query
    or for none.  This prevents silently reporting a favourable subset.
    """

    selected_bootstrap = _select_bootstrap(bootstrap)
    if not isinstance(evaluation_contract, DiagnosticEvaluationContract):
        raise TypeError("evaluation_contract must be a DiagnosticEvaluationContract")
    observations = tuple(forecasts)
    if not observations:
        raise DomainError("composition diagnostics require at least one observation")
    if len({item.query_id for item in observations}) != len(observations):
        raise DomainError("composition diagnostic query identifiers must be unique")
    corpus_digest = composition_forecasts_sha256(observations)
    if evaluation_contract.composition_forecasts_sha256 is None:
        raise DomainError(
            "composition evaluation requires a pinned forecast corpus fingerprint"
        )
    if corpus_digest != evaluation_contract.composition_forecasts_sha256:
        raise DomainError("composition forecast corpus differs from the evaluation contract")
    if (
        isinstance(calibration_bins, bool)
        or not isinstance(calibration_bins, int)
        or calibration_bins <= 0
    ):
        raise DomainError("calibration_bins must be a positive integer")

    has_calibration = [item.top_confidence is not None for item in observations]
    has_brier = [item.categorical_probabilities is not None for item in observations]
    if len(set(has_calibration)) != 1:
        raise DomainError("top-label calibration must cover every query or none")
    if len(set(has_brier)) != 1:
        raise DomainError("Brier distributions must cover every query or none")

    nll = np.asarray([-item.target_log_probability for item in observations])
    tokens = np.asarray([item.token_count for item in observations], dtype=np.float64)
    confidence = (
        np.asarray([item.top_confidence for item in observations], dtype=np.float64)
        if has_calibration[0]
        else None
    )
    correct = (
        np.asarray([item.top_is_correct for item in observations], dtype=np.float64)
        if has_calibration[0]
        else None
    )
    brier: FloatArray | None = None
    if has_brier[0]:
        brier_values: list[float] = []
        for item in observations:
            assert item.categorical_probabilities is not None
            assert item.target_index is not None
            probabilities = np.asarray(item.categorical_probabilities, dtype=np.float64)
            one_hot = np.zeros_like(probabilities)
            one_hot[item.target_index] = 1.0
            brier_values.append(float(np.square(probabilities - one_hot).sum()))
        brier = np.asarray(brier_values, dtype=np.float64)

    nll_bootstrap: list[float] = []
    token_nll_bootstrap: list[float] = []
    accuracy_bootstrap: list[float] = []
    confidence_bootstrap: list[float] = []
    ece_bootstrap: list[float] = []
    brier_bootstrap: list[float] = []
    for indices in _bootstrap_indices(len(observations), selected_bootstrap):
        sampled_nll = nll[indices]
        sampled_tokens = tokens[indices]
        nll_bootstrap.append(float(sampled_nll.mean()))
        token_nll_bootstrap.append(float(sampled_nll.sum() / sampled_tokens.sum()))
        if confidence is not None and correct is not None:
            sampled_confidence = confidence[indices]
            sampled_correct = correct[indices]
            accuracy_bootstrap.append(float(sampled_correct.mean()))
            confidence_bootstrap.append(float(sampled_confidence.mean()))
            ece_bootstrap.append(_ece(sampled_confidence, sampled_correct, calibration_bins))
        if brier is not None:
            brier_bootstrap.append(float(brier[indices].mean()))

    calibration_bin_payload: tuple[CalibrationBin, ...] = ()
    top_accuracy: MetricEstimate | None = None
    mean_confidence: MetricEstimate | None = None
    top_ece: MetricEstimate | None = None
    if confidence is not None and correct is not None:
        calibration_bin_payload = _calibration_bins(confidence, correct, calibration_bins)
        top_accuracy = _estimate(float(correct.mean()), accuracy_bootstrap, selected_bootstrap)
        mean_confidence = _estimate(
            float(confidence.mean()), confidence_bootstrap, selected_bootstrap
        )
        top_ece = _estimate(
            _ece(confidence, correct, calibration_bins),
            ece_bootstrap,
            selected_bootstrap,
        )
    brier_estimate = (
        _estimate(float(brier.mean()), brier_bootstrap, selected_bootstrap)
        if brier is not None
        else None
    )
    return CompositionDiagnostics(
        observation_count=len(observations),
        total_token_count=int(tokens.sum()),
        query_ids_sha256=_identifier_fingerprint(tuple(item.query_id for item in observations)),
        calibration_bin_count=calibration_bins,
        mean_composition_nll=_estimate(float(nll.mean()), nll_bootstrap, selected_bootstrap),
        corpus_nll_per_token=_estimate(
            float(nll.sum() / tokens.sum()),
            token_nll_bootstrap,
            selected_bootstrap,
        ),
        top_label_accuracy=top_accuracy,
        mean_top_confidence=mean_confidence,
        top_label_ece=top_ece,
        multiclass_brier=brier_estimate,
        calibration_bins=calibration_bin_payload,
        bootstrap=selected_bootstrap,
        evaluation_contract=evaluation_contract,
        forecasts_sha256=corpus_digest,
    )


def _resolve_payload_sha256(
    payload: object | None,
    declared: str | None,
    *,
    name: str,
) -> str | None:
    if payload is None:
        if declared is not None:
            raise DomainError(f"{name} cannot declare a payload digest without a payload")
        return None
    computed: str | None = None
    if isinstance(payload, bytes):
        computed = sha256(payload).hexdigest()
    elif isinstance(payload, (bool, str, int, float)):
        normalized = _strict_json_value(payload, path=name)
        computed = sha256(_json_text({"payload": normalized}).encode("utf-8")).hexdigest()
    else:
        raise DomainError(
            f"{name} must be immutable canonical bytes or a strict-JSON scalar; "
            "serialize structured matcher inputs before evaluation"
        )
    if declared is None:
        assert computed is not None
        return computed
    if _SHA256_PATTERN.fullmatch(declared) is None:
        raise DomainError(f"{name} digest must be a lowercase SHA-256")
    if computed is not None and declared != computed:
        raise DomainError(f"{name} digest disagrees with its canonical payload bytes")
    return declared


@dataclass(frozen=True)
class AcceptableTarget(Generic[T]):
    """One member of the complete acceptable polymorph set for a query."""

    target_id: str
    payload: T
    payload_sha256: str | None = None

    def __post_init__(self) -> None:
        _require_identifier(self.target_id, "target_id")
        if self.payload is None:
            raise DomainError("acceptable target payload cannot be None")
        object.__setattr__(
            self,
            "payload_sha256",
            _resolve_payload_sha256(
                self.payload,
                self.payload_sha256,
                name="acceptable target payload",
            ),
        )


@dataclass(frozen=True)
class RankedAttempt(Generic[T]):
    """One generated attempt; invalid attempts deliberately retain their rank."""

    attempt_id: str
    score: float
    valid: bool
    payload: T | None = None
    payload_sha256: str | None = None

    def __post_init__(self) -> None:
        _require_identifier(self.attempt_id, "attempt_id")
        if not math.isfinite(self.score):
            raise DomainError("attempt score must be finite")
        if not isinstance(self.valid, bool):
            raise DomainError("attempt validity must be Boolean")
        if self.valid and self.payload is None:
            raise DomainError("a valid attempt must carry a matcher payload")
        object.__setattr__(
            self,
            "payload_sha256",
            _resolve_payload_sha256(
                self.payload,
                self.payload_sha256,
                name="generated attempt payload",
            ),
        )


@dataclass(frozen=True)
class RankedRetrievalCase(Generic[T]):
    """One query, all target polymorphs, and unfiltered generated attempts."""

    query_id: str
    acceptable_targets: tuple[AcceptableTarget[T], ...]
    attempts: tuple[RankedAttempt[T], ...]

    def __post_init__(self) -> None:
        _require_identifier(self.query_id, "query_id")
        targets = tuple(self.acceptable_targets)
        attempts = tuple(self.attempts)
        object.__setattr__(self, "acceptable_targets", targets)
        object.__setattr__(self, "attempts", attempts)
        if not targets:
            raise DomainError("each retrieval query needs at least one acceptable polymorph")
        if len({item.target_id for item in targets}) != len(targets):
            raise DomainError("acceptable target identifiers must be unique within a query")
        if len({item.attempt_id for item in attempts}) != len(attempts):
            raise DomainError("attempt identifiers must be unique within a query")


def _ranked(attempts: Sequence[RankedAttempt[T]]) -> tuple[RankedAttempt[T], ...]:
    return tuple(sorted(attempts, key=lambda item: (-item.score, item.attempt_id)))


def _validate_ks(ks: Sequence[int]) -> tuple[int, ...]:
    result = tuple(sorted(set(ks)))
    if not result or any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in result
    ):
        raise DomainError("K values must be one or more positive integers")
    return result


def _attempt_identity(attempt: RankedAttempt[T]) -> dict[str, object]:
    current_digest = _resolve_payload_sha256(
        attempt.payload,
        attempt.payload_sha256,
        name="generated attempt payload",
    )
    return {
        "attempt_id": attempt.attempt_id,
        "payload_sha256": current_digest,
        "score_hex": float(attempt.score).hex(),
        "valid": attempt.valid,
    }


def _target_identity(target: AcceptableTarget[T]) -> dict[str, object]:
    current_digest = _resolve_payload_sha256(
        target.payload,
        target.payload_sha256,
        name="acceptable target payload",
    )
    return {
        "payload_sha256": current_digest,
        "target_id": target.target_id,
    }


def _target_corpus_payload(cases: Sequence[RankedRetrievalCase[T]]) -> dict[str, object]:
    return {
        "queries": [
            {
                "query_id": case.query_id,
                "targets": [_target_identity(target) for target in case.acceptable_targets],
            }
            for case in cases
        ],
        "schema": "crystalatlas.diagnostic-target-corpus.v1",
    }


def _attempt_corpus_payload(
    cases: Sequence[RankedRetrievalCase[T]],
    *,
    arm: str,
) -> dict[str, object]:
    return {
        "arm": arm,
        "queries": [
            {
                "attempts": [_attempt_identity(attempt) for attempt in case.attempts],
                "query_id": case.query_id,
            }
            for case in cases
        ],
        "schema": "crystalatlas.diagnostic-attempt-corpus.v1",
    }


def _payload_fingerprint(payload: Mapping[str, object]) -> str:
    return sha256(_json_text(payload).encode("utf-8")).hexdigest()


def retrieval_corpus_fingerprints(
    cases: Sequence[RankedRetrievalCase[T]],
) -> DiagnosticCorpusFingerprints:
    """Fingerprint the exact ordered targets and generated retrieval attempts."""

    materialized = tuple(cases)
    if not materialized:
        raise DomainError("retrieval corpus fingerprints require at least one query")
    return DiagnosticCorpusFingerprints(
        target_corpus_sha256=_payload_fingerprint(_target_corpus_payload(materialized)),
        generated_attempt_corpus_sha256=_payload_fingerprint(
            _attempt_corpus_payload(materialized, arm="retrieval")
        ),
    )


def _retrieval_cases_fingerprint(cases: Sequence[RankedRetrievalCase[T]]) -> str:
    payload = {
        "attempt_corpus": _attempt_corpus_payload(cases, arm="retrieval"),
        "schema": "crystalatlas.retrieval-cases.v1",
        "target_corpus": _target_corpus_payload(cases),
    }
    return _payload_fingerprint(payload)


def _precompute_case_matches(
    case: RankedRetrievalCase[T],
    matcher: Callable[[T, T], bool],
    *,
    arm: str,
) -> tuple[dict[str, frozenset[int]], list[dict[str, object]]]:
    matches: dict[str, frozenset[int]] = {}
    decisions: list[dict[str, object]] = []
    for attempt in case.attempts:
        matched: set[int] = set()
        if attempt.valid:
            assert attempt.payload is not None
            for target_index, target in enumerate(case.acceptable_targets):
                forward = matcher(attempt.payload, target.payload)
                repeated = matcher(attempt.payload, target.payload)
                reverse = matcher(target.payload, attempt.payload)
                if not all(isinstance(value, bool) for value in (forward, repeated, reverse)):
                    raise DomainError("equivalence matcher must return Python bool values")
                if forward != repeated or forward != reverse:
                    raise DomainError(
                        "equivalence matcher is stateful, nondeterministic, or asymmetric"
                    )
                if forward:
                    matched.add(target_index)
                decisions.append(
                    {
                        "arm": arm,
                        "attempt_id": attempt.attempt_id,
                        "match": forward,
                        "query_id": case.query_id,
                        "target_id": target.target_id,
                    }
                )
        matches[attempt.attempt_id] = frozenset(matched)
    return matches, decisions


def _case_hits(
    case: RankedRetrievalCase[T],
    k: int,
    matches: Mapping[str, frozenset[int]],
) -> tuple[bool, frozenset[int], int]:
    hit_targets: set[int] = set()
    valid_count = 0
    for attempt in _ranked(case.attempts)[:k]:
        if not attempt.valid:
            continue
        assert attempt.payload is not None
        valid_count += 1
        hit_targets.update(matches[attempt.attempt_id])
    return bool(hit_targets), frozenset(hit_targets), valid_count


@dataclass(frozen=True, slots=True)
class RetrievalAtK:
    k: int
    recall: MetricEstimate
    polymorph_coverage: MetricEstimate
    valid_attempt_rate: MetricEstimate
    query_hits: int
    target_hits: int
    target_count: int
    attempt_slots: int

    def to_manifest(self) -> dict[str, object]:
        return {
            "attempt_slots": self.attempt_slots,
            "k": self.k,
            "polymorph_coverage": self.polymorph_coverage.to_manifest(),
            "query_hits": self.query_hits,
            "recall": self.recall.to_manifest(),
            "target_count": self.target_count,
            "target_hits": self.target_hits,
            "valid_attempt_rate": self.valid_attempt_rate.to_manifest(),
        }


@dataclass(frozen=True, slots=True)
class ScaffoldRetrievalDiagnostics:
    query_count: int
    query_ids_sha256: str
    at_k: tuple[RetrievalAtK, ...]
    bootstrap: BootstrapProtocol
    matcher_contract: EquivalenceMatcherContract
    evaluation_contract: DiagnosticEvaluationContract
    cases_sha256: str
    match_decisions_sha256: str

    def to_manifest(self) -> dict[str, object]:
        return {
            "at_k": [item.to_manifest() for item in self.at_k],
            "bootstrap": self.bootstrap.to_manifest(),
            "cases_sha256": self.cases_sha256,
            "evaluation_contract": self.evaluation_contract.to_manifest(),
            "invalid_or_missing_attempts_occupy_slots": True,
            "matcher": self.matcher_contract.to_manifest(),
            "match_decisions_sha256": self.match_decisions_sha256,
            "polymorph_targets": "all_acceptable_targets_retained",
            "query_count": self.query_count,
            "query_ids_sha256": self.query_ids_sha256,
            "schema": _SCAFFOLD_SCHEMA,
            "tie_policy": _TIE_POLICY,
        }

    def to_json(self) -> str:
        return _json_text(self.to_manifest())


def evaluate_scaffold_retrieval(
    cases: Sequence[RankedRetrievalCase[T]],
    *,
    ks: Sequence[int],
    matcher: Callable[[T, T], bool],
    matcher_contract: EquivalenceMatcherContract,
    evaluation_contract: DiagnosticEvaluationContract,
    bootstrap: BootstrapProtocol | None = None,
) -> ScaffoldRetrievalDiagnostics:
    """Compute legal-scaffold Recall@K and acceptable-polymorph coverage."""

    selected_bootstrap = _select_bootstrap(bootstrap)
    queries = tuple(cases)
    if not queries:
        raise DomainError("scaffold diagnostics require at least one query")
    if len({case.query_id for case in queries}) != len(queries):
        raise DomainError("scaffold diagnostic query identifiers must be unique")
    if not callable(matcher):
        raise TypeError("matcher must be callable")
    if not isinstance(matcher_contract, EquivalenceMatcherContract):
        raise TypeError("matcher_contract must be an EquivalenceMatcherContract")
    if not isinstance(evaluation_contract, DiagnosticEvaluationContract):
        raise TypeError("evaluation_contract must be a DiagnosticEvaluationContract")
    if matcher_callable_sha256(matcher) != matcher_contract.implementation_sha256:
        raise DomainError("matcher callable differs from its declared implementation")
    k_values = _validate_ks(ks)
    if max(k_values) > evaluation_contract.attempt_budget:
        raise DomainError("K cannot exceed the frozen generation attempt budget")
    if any(len(case.attempts) > evaluation_contract.attempt_budget for case in queries):
        raise DomainError("a retrieval case exceeds the frozen generation attempt budget")
    corpus = retrieval_corpus_fingerprints(queries)
    if corpus.target_corpus_sha256 != evaluation_contract.target_corpus_sha256:
        raise DomainError("retrieval target corpus differs from the evaluation contract")
    if (
        corpus.generated_attempt_corpus_sha256
        != evaluation_contract.generated_attempt_corpus_sha256
    ):
        raise DomainError("retrieval attempt corpus differs from the evaluation contract")
    case_matches: list[dict[str, frozenset[int]]] = []
    decision_payload: list[dict[str, object]] = []
    for case in queries:
        matches, decisions = _precompute_case_matches(case, matcher, arm="retrieval")
        case_matches.append(matches)
        decision_payload.extend(decisions)
    if matcher_callable_sha256(matcher) != matcher_contract.implementation_sha256:
        raise DomainError("matcher implementation state changed during evaluation")
    if retrieval_corpus_fingerprints(queries) != corpus:
        raise DomainError("retrieval payload corpus changed during evaluation")
    results: list[RetrievalAtK] = []
    for k in k_values:
        hits: list[bool] = []
        target_hits: list[int] = []
        target_counts: list[int] = []
        valid_counts: list[int] = []
        for case, matches in zip(queries, case_matches, strict=True):
            hit, matched_targets, valid = _case_hits(case, k, matches)
            hits.append(hit)
            target_hits.append(len(matched_targets))
            target_counts.append(len(case.acceptable_targets))
            valid_counts.append(valid)
        hit_array = np.asarray(hits, dtype=np.float64)
        target_hit_array = np.asarray(target_hits, dtype=np.float64)
        target_count_array = np.asarray(target_counts, dtype=np.float64)
        valid_array = np.asarray(valid_counts, dtype=np.float64)
        recall_bootstrap: list[float] = []
        coverage_bootstrap: list[float] = []
        valid_bootstrap: list[float] = []
        for indices in _bootstrap_indices(len(queries), selected_bootstrap):
            recall_bootstrap.append(float(hit_array[indices].mean()))
            coverage_bootstrap.append(
                float(target_hit_array[indices].sum() / target_count_array[indices].sum())
            )
            valid_bootstrap.append(float(valid_array[indices].sum() / (len(queries) * k)))
        total_targets = int(target_count_array.sum())
        attempt_slots = len(queries) * k
        results.append(
            RetrievalAtK(
                k=k,
                recall=_estimate(float(hit_array.mean()), recall_bootstrap, selected_bootstrap),
                polymorph_coverage=_estimate(
                    float(target_hit_array.sum() / total_targets),
                    coverage_bootstrap,
                    selected_bootstrap,
                ),
                valid_attempt_rate=_estimate(
                    float(valid_array.sum() / attempt_slots),
                    valid_bootstrap,
                    selected_bootstrap,
                ),
                query_hits=int(hit_array.sum()),
                target_hits=int(target_hit_array.sum()),
                target_count=total_targets,
                attempt_slots=attempt_slots,
            )
        )
    return ScaffoldRetrievalDiagnostics(
        len(queries),
        _identifier_fingerprint(tuple(case.query_id for case in queries)),
        tuple(results),
        selected_bootstrap,
        matcher_contract,
        evaluation_contract,
        _retrieval_cases_fingerprint(queries),
        sha256(_json_text({"decisions": decision_payload}).encode("utf-8")).hexdigest(),
    )


@dataclass(frozen=True)
class GeometryDecompositionCase(Generic[T]):
    """Paired geometry trials with oracle and selected scaffold conditioning."""

    query_id: str
    acceptable_targets: tuple[AcceptableTarget[T], ...]
    oracle_scaffold_attempts: tuple[RankedAttempt[T], ...]
    predicted_scaffold_attempts: tuple[RankedAttempt[T], ...]

    def __post_init__(self) -> None:
        oracle_case = RankedRetrievalCase(
            self.query_id, self.acceptable_targets, self.oracle_scaffold_attempts
        )
        predicted_case = RankedRetrievalCase(
            self.query_id, self.acceptable_targets, self.predicted_scaffold_attempts
        )
        object.__setattr__(self, "acceptable_targets", oracle_case.acceptable_targets)
        object.__setattr__(self, "oracle_scaffold_attempts", oracle_case.attempts)
        object.__setattr__(self, "predicted_scaffold_attempts", predicted_case.attempts)


def _geometry_retrieval_cases(
    cases: Sequence[GeometryDecompositionCase[T]],
) -> tuple[tuple[RankedRetrievalCase[T], ...], tuple[RankedRetrievalCase[T], ...]]:
    oracle = tuple(
        RankedRetrievalCase(
            case.query_id,
            case.acceptable_targets,
            case.oracle_scaffold_attempts,
        )
        for case in cases
    )
    predicted = tuple(
        RankedRetrievalCase(
            case.query_id,
            case.acceptable_targets,
            case.predicted_scaffold_attempts,
        )
        for case in cases
    )
    return oracle, predicted


def _geometry_attempt_corpus_payload(
    cases: Sequence[GeometryDecompositionCase[T]],
) -> dict[str, object]:
    oracle, predicted = _geometry_retrieval_cases(cases)
    return {
        "arms": [
            _attempt_corpus_payload(oracle, arm="oracle"),
            _attempt_corpus_payload(predicted, arm="predicted"),
        ],
        "schema": "crystalatlas.geometry-attempt-corpus.v1",
    }


def geometry_corpus_fingerprints(
    cases: Sequence[GeometryDecompositionCase[T]],
) -> DiagnosticCorpusFingerprints:
    """Fingerprint exact targets and both ordered geometry-generation arms."""

    materialized = tuple(cases)
    if not materialized:
        raise DomainError("geometry corpus fingerprints require at least one query")
    oracle, _ = _geometry_retrieval_cases(materialized)
    return DiagnosticCorpusFingerprints(
        target_corpus_sha256=_payload_fingerprint(_target_corpus_payload(oracle)),
        generated_attempt_corpus_sha256=_payload_fingerprint(
            _geometry_attempt_corpus_payload(materialized)
        ),
    )


def _geometry_cases_fingerprint(cases: Sequence[GeometryDecompositionCase[T]]) -> str:
    oracle, _ = _geometry_retrieval_cases(cases)
    return _payload_fingerprint(
        {
            "attempt_corpus": _geometry_attempt_corpus_payload(cases),
            "schema": "crystalatlas.geometry-cases.v1",
            "target_corpus": _target_corpus_payload(oracle),
        }
    )


@dataclass(frozen=True, slots=True)
class GeometryAtK:
    k: int
    oracle_scaffold_match: MetricEstimate
    predicted_scaffold_match: MetricEstimate
    delta_k: MetricEstimate
    oracle_valid_attempt_rate: MetricEstimate
    predicted_valid_attempt_rate: MetricEstimate
    oracle_query_hits: int
    predicted_query_hits: int
    attempt_slots_per_arm: int

    def to_manifest(self) -> dict[str, object]:
        return {
            "attempt_slots_per_arm": self.attempt_slots_per_arm,
            "delta_k": self.delta_k.to_manifest(),
            "k": self.k,
            "oracle_query_hits": self.oracle_query_hits,
            "oracle_scaffold_match": self.oracle_scaffold_match.to_manifest(),
            "oracle_valid_attempt_rate": self.oracle_valid_attempt_rate.to_manifest(),
            "predicted_query_hits": self.predicted_query_hits,
            "predicted_scaffold_match": self.predicted_scaffold_match.to_manifest(),
            "predicted_valid_attempt_rate": self.predicted_valid_attempt_rate.to_manifest(),
        }


@dataclass(frozen=True, slots=True)
class GeometryDecompositionDiagnostics:
    query_count: int
    query_ids_sha256: str
    at_k: tuple[GeometryAtK, ...]
    bootstrap: BootstrapProtocol
    matcher_contract: EquivalenceMatcherContract
    evaluation_contract: DiagnosticEvaluationContract
    cases_sha256: str
    match_decisions_sha256: str

    def to_manifest(self) -> dict[str, object]:
        return {
            "at_k": [item.to_manifest() for item in self.at_k],
            "bootstrap": self.bootstrap.to_manifest(),
            "cases_sha256": self.cases_sha256,
            "delta_definition": (
                "oracle_scaffold_match_at_k-minus-predicted_scaffold_match_at_k"
            ),
            "invalid_or_missing_attempts_occupy_slots": True,
            "evaluation_contract": self.evaluation_contract.to_manifest(),
            "matcher": self.matcher_contract.to_manifest(),
            "match_decisions_sha256": self.match_decisions_sha256,
            "paired_by_query": True,
            "polymorph_targets": "all_acceptable_targets_retained",
            "query_count": self.query_count,
            "query_ids_sha256": self.query_ids_sha256,
            "schema": _GEOMETRY_SCHEMA,
            "tie_policy": _TIE_POLICY,
        }

    def to_json(self) -> str:
        return _json_text(self.to_manifest())


def evaluate_geometry_decomposition(
    cases: Sequence[GeometryDecompositionCase[T]],
    *,
    ks: Sequence[int],
    matcher: Callable[[T, T], bool],
    matcher_contract: EquivalenceMatcherContract,
    evaluation_contract: DiagnosticEvaluationContract,
    bootstrap: BootstrapProtocol | None = None,
) -> GeometryDecompositionDiagnostics:
    """Compare geometry Match@K under oracle and predicted scaffolds.

    The confidence interval for ``Delta_K`` is paired: the same resampled query
    indices are used in the oracle and predicted arms.
    """

    selected_bootstrap = _select_bootstrap(bootstrap)
    queries = tuple(cases)
    if not queries:
        raise DomainError("geometry diagnostics require at least one paired query")
    if len({case.query_id for case in queries}) != len(queries):
        raise DomainError("geometry diagnostic query identifiers must be unique")
    if not callable(matcher):
        raise TypeError("matcher must be callable")
    if not isinstance(matcher_contract, EquivalenceMatcherContract):
        raise TypeError("matcher_contract must be an EquivalenceMatcherContract")
    if not isinstance(evaluation_contract, DiagnosticEvaluationContract):
        raise TypeError("evaluation_contract must be a DiagnosticEvaluationContract")
    if matcher_callable_sha256(matcher) != matcher_contract.implementation_sha256:
        raise DomainError("matcher callable differs from its declared implementation")
    k_values = _validate_ks(ks)
    if max(k_values) > evaluation_contract.attempt_budget:
        raise DomainError("K cannot exceed the frozen generation attempt budget")
    if any(
        len(case.oracle_scaffold_attempts) > evaluation_contract.attempt_budget
        or len(case.predicted_scaffold_attempts) > evaluation_contract.attempt_budget
        for case in queries
    ):
        raise DomainError("a geometry case exceeds the frozen per-arm attempt budget")
    corpus = geometry_corpus_fingerprints(queries)
    if corpus.target_corpus_sha256 != evaluation_contract.target_corpus_sha256:
        raise DomainError("geometry target corpus differs from the evaluation contract")
    if (
        corpus.generated_attempt_corpus_sha256
        != evaluation_contract.generated_attempt_corpus_sha256
    ):
        raise DomainError("geometry attempt corpus differs from the evaluation contract")
    oracle_matches: list[dict[str, frozenset[int]]] = []
    predicted_matches: list[dict[str, frozenset[int]]] = []
    decision_payload: list[dict[str, object]] = []
    for case in queries:
        oracle_case = RankedRetrievalCase(
            case.query_id, case.acceptable_targets, case.oracle_scaffold_attempts
        )
        predicted_case = RankedRetrievalCase(
            case.query_id, case.acceptable_targets, case.predicted_scaffold_attempts
        )
        oracle_match, oracle_decisions = _precompute_case_matches(
            oracle_case, matcher, arm="oracle"
        )
        predicted_match, predicted_decisions = _precompute_case_matches(
            predicted_case, matcher, arm="predicted"
        )
        oracle_matches.append(oracle_match)
        predicted_matches.append(predicted_match)
        decision_payload.extend(oracle_decisions)
        decision_payload.extend(predicted_decisions)
    if matcher_callable_sha256(matcher) != matcher_contract.implementation_sha256:
        raise DomainError("matcher implementation state changed during evaluation")
    if geometry_corpus_fingerprints(queries) != corpus:
        raise DomainError("geometry payload corpus changed during evaluation")
    results: list[GeometryAtK] = []
    for k in k_values:
        oracle_hits: list[bool] = []
        predicted_hits: list[bool] = []
        oracle_valid: list[int] = []
        predicted_valid: list[int] = []
        for case, oracle_match, predicted_match in zip(
            queries, oracle_matches, predicted_matches, strict=True
        ):
            oracle_case = RankedRetrievalCase(
                case.query_id, case.acceptable_targets, case.oracle_scaffold_attempts
            )
            predicted_case = RankedRetrievalCase(
                case.query_id,
                case.acceptable_targets,
                case.predicted_scaffold_attempts,
            )
            oracle_hit, _, oracle_count = _case_hits(oracle_case, k, oracle_match)
            predicted_hit, _, predicted_count = _case_hits(predicted_case, k, predicted_match)
            oracle_hits.append(oracle_hit)
            predicted_hits.append(predicted_hit)
            oracle_valid.append(oracle_count)
            predicted_valid.append(predicted_count)
        oracle_array = np.asarray(oracle_hits, dtype=np.float64)
        predicted_array = np.asarray(predicted_hits, dtype=np.float64)
        delta_array = oracle_array - predicted_array
        oracle_valid_array = np.asarray(oracle_valid, dtype=np.float64)
        predicted_valid_array = np.asarray(predicted_valid, dtype=np.float64)
        oracle_bootstrap: list[float] = []
        predicted_bootstrap: list[float] = []
        delta_bootstrap: list[float] = []
        oracle_valid_bootstrap: list[float] = []
        predicted_valid_bootstrap: list[float] = []
        slots = len(queries) * k
        for indices in _bootstrap_indices(len(queries), selected_bootstrap):
            oracle_bootstrap.append(float(oracle_array[indices].mean()))
            predicted_bootstrap.append(float(predicted_array[indices].mean()))
            delta_bootstrap.append(float(delta_array[indices].mean()))
            oracle_valid_bootstrap.append(float(oracle_valid_array[indices].sum() / slots))
            predicted_valid_bootstrap.append(
                float(predicted_valid_array[indices].sum() / slots)
            )
        results.append(
            GeometryAtK(
                k=k,
                oracle_scaffold_match=_estimate(
                    float(oracle_array.mean()), oracle_bootstrap, selected_bootstrap
                ),
                predicted_scaffold_match=_estimate(
                    float(predicted_array.mean()),
                    predicted_bootstrap,
                    selected_bootstrap,
                ),
                delta_k=_estimate(
                    float(delta_array.mean()), delta_bootstrap, selected_bootstrap
                ),
                oracle_valid_attempt_rate=_estimate(
                    float(oracle_valid_array.sum() / slots),
                    oracle_valid_bootstrap,
                    selected_bootstrap,
                ),
                predicted_valid_attempt_rate=_estimate(
                    float(predicted_valid_array.sum() / slots),
                    predicted_valid_bootstrap,
                    selected_bootstrap,
                ),
                oracle_query_hits=int(oracle_array.sum()),
                predicted_query_hits=int(predicted_array.sum()),
                attempt_slots_per_arm=slots,
            )
        )
    return GeometryDecompositionDiagnostics(
        len(queries),
        _identifier_fingerprint(tuple(case.query_id for case in queries)),
        tuple(results),
        selected_bootstrap,
        matcher_contract,
        evaluation_contract,
        _geometry_cases_fingerprint(queries),
        sha256(_json_text({"decisions": decision_payload}).encode("utf-8")).hexdigest(),
    )


@dataclass(frozen=True, slots=True)
class TinyOverfitThresholds:
    """Predeclared, inclusive pass thresholds for the tiny-data gate."""

    max_training_loss: float
    max_composition_nll: float
    min_scaffold_recall_at_k: float
    min_oracle_match_at_k: float
    min_predicted_match_at_k: float
    max_invalid_attempt_rate: float
    min_relative_loss_reduction: float
    min_relative_composition_nll_reduction: float

    def __post_init__(self) -> None:
        for name in ("max_training_loss", "max_composition_nll"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise DomainError(f"{name} must be finite and nonnegative")
        for name in (
            "min_scaffold_recall_at_k",
            "min_oracle_match_at_k",
            "min_predicted_match_at_k",
            "max_invalid_attempt_rate",
            "min_relative_loss_reduction",
            "min_relative_composition_nll_reduction",
        ):
            _require_probability(getattr(self, name), name)

    def to_manifest(self) -> dict[str, object]:
        return {
            "max_composition_nll": self.max_composition_nll,
            "max_invalid_attempt_rate": self.max_invalid_attempt_rate,
            "max_training_loss": self.max_training_loss,
            "min_oracle_match_at_k": self.min_oracle_match_at_k,
            "min_predicted_match_at_k": self.min_predicted_match_at_k,
            "min_relative_composition_nll_reduction": (
                self.min_relative_composition_nll_reduction
            ),
            "min_relative_loss_reduction": self.min_relative_loss_reduction,
            "min_scaffold_recall_at_k": self.min_scaffold_recall_at_k,
            "threshold_comparison": "inclusive",
        }


@dataclass(frozen=True, slots=True)
class TinyOverfitGateProtocol:
    """Frozen tiny-corpus identity, schedule, seeds, and pass contract."""

    dataset_manifest_sha256: str
    evaluation_policy_sha256: str
    scaffold_matcher_contract_sha256: str
    geometry_matcher_contract_sha256: str
    example_ids: tuple[str, ...]
    max_steps: int
    minimum_steps: int
    evaluation_interval: int
    scaffold_k: int
    match_k: int
    seed: int
    thresholds: TinyOverfitThresholds
    minimum_consecutive_passing_evaluations: int = 2

    def __post_init__(self) -> None:
        for name in (
            "dataset_manifest_sha256",
            "evaluation_policy_sha256",
            "scaffold_matcher_contract_sha256",
            "geometry_matcher_contract_sha256",
        ):
            if _SHA256_PATTERN.fullmatch(getattr(self, name)) is None:
                raise DomainError(f"{name} must be a lowercase SHA-256 digest")
        identifiers = tuple(self.example_ids)
        object.__setattr__(self, "example_ids", identifiers)
        if not identifiers:
            raise DomainError("the tiny overfit corpus cannot be empty")
        for identifier in identifiers:
            _require_identifier(identifier, "example_id")
        if len(set(identifiers)) != len(identifiers):
            raise DomainError("tiny overfit example identifiers must be unique")
        integer_values = {
            "max_steps": self.max_steps,
            "minimum_steps": self.minimum_steps,
            "evaluation_interval": self.evaluation_interval,
            "scaffold_k": self.scaffold_k,
            "match_k": self.match_k,
            "minimum_consecutive_passing_evaluations": (
                self.minimum_consecutive_passing_evaluations
            ),
        }
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in integer_values.values()
        ):
            raise DomainError("overfit schedule and K values must be positive integers")
        if self.minimum_steps > self.max_steps:
            raise DomainError("minimum_steps cannot exceed max_steps")
        if (
            isinstance(self.minimum_consecutive_passing_evaluations, bool)
            or not isinstance(self.minimum_consecutive_passing_evaluations, int)
            or self.minimum_consecutive_passing_evaluations < 1
        ):
            raise DomainError(
                "minimum_consecutive_passing_evaluations must be a positive integer"
            )
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or not 0 <= self.seed < 2**64
        ):
            raise DomainError("overfit seed must be an unsigned 64-bit integer")

    def to_manifest(self) -> dict[str, object]:
        return {
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "evaluation_policy_sha256": self.evaluation_policy_sha256,
            "evaluation_interval": self.evaluation_interval,
            "example_ids": list(self.example_ids),
            "match_k": self.match_k,
            "max_steps": self.max_steps,
            "geometry_matcher_contract_sha256": (self.geometry_matcher_contract_sha256),
            "minimum_consecutive_passing_evaluations": (
                self.minimum_consecutive_passing_evaluations
            ),
            "minimum_steps": self.minimum_steps,
            "scaffold_matcher_contract_sha256": (self.scaffold_matcher_contract_sha256),
            "scaffold_k": self.scaffold_k,
            "schema": _OVERFIT_SCHEMA + ".protocol",
            "seed": self.seed,
            "thresholds": self.thresholds.to_manifest(),
        }

    def to_json(self) -> str:
        return _json_text(self.to_manifest())


@dataclass(frozen=True, slots=True)
class TinyOverfitEvaluation:
    """Source diagnostic bundle returned by a tiny-run evaluation callback."""

    step: int
    training_loss: float
    composition: CompositionDiagnostics
    scaffold: ScaffoldRetrievalDiagnostics
    geometry: GeometryDecompositionDiagnostics

    def __post_init__(self) -> None:
        if isinstance(self.step, bool) or not isinstance(self.step, int) or self.step < 0:
            raise DomainError("overfit evaluation step must be a nonnegative integer")
        if not math.isfinite(self.training_loss) or self.training_loss < 0.0:
            raise DomainError("overfit evaluation training loss must be finite and nonnegative")
        if not isinstance(self.composition, CompositionDiagnostics):
            raise TypeError("composition must be CompositionDiagnostics")
        if not isinstance(self.scaffold, ScaffoldRetrievalDiagnostics):
            raise TypeError("scaffold must be ScaffoldRetrievalDiagnostics")
        if not isinstance(self.geometry, GeometryDecompositionDiagnostics):
            raise TypeError("geometry must be GeometryDecompositionDiagnostics")


@dataclass(frozen=True, slots=True)
class TinyOverfitSnapshot:
    """Metrics derived from manifest-bound source diagnostics."""

    step: int
    training_loss: float
    composition_nll: float
    scaffold_recall_at_k: float
    oracle_match_at_k: float
    predicted_match_at_k: float
    invalid_attempt_rate: float
    checkpoint_sha256: str
    evaluation_policy_sha256: str
    scaffold_matcher_contract_sha256: str
    geometry_matcher_contract_sha256: str
    composition_diagnostics_sha256: str
    scaffold_diagnostics_sha256: str
    geometry_diagnostics_sha256: str

    def __post_init__(self) -> None:
        if isinstance(self.step, bool) or not isinstance(self.step, int) or self.step < 0:
            raise DomainError("overfit snapshot step must be a nonnegative integer")
        for name in ("training_loss", "composition_nll"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise DomainError(f"snapshot {name} must be finite and nonnegative")
        for name in (
            "scaffold_recall_at_k",
            "oracle_match_at_k",
            "predicted_match_at_k",
            "invalid_attempt_rate",
        ):
            _require_probability(getattr(self, name), f"snapshot {name}")
        for name in (
            "checkpoint_sha256",
            "evaluation_policy_sha256",
            "scaffold_matcher_contract_sha256",
            "geometry_matcher_contract_sha256",
            "composition_diagnostics_sha256",
            "scaffold_diagnostics_sha256",
            "geometry_diagnostics_sha256",
        ):
            if _SHA256_PATTERN.fullmatch(getattr(self, name)) is None:
                raise DomainError(f"snapshot {name} must be a lowercase SHA-256")

    def to_manifest(self) -> dict[str, object]:
        return {
            "composition_nll": self.composition_nll,
            "composition_diagnostics_sha256": self.composition_diagnostics_sha256,
            "checkpoint_sha256": self.checkpoint_sha256,
            "evaluation_policy_sha256": self.evaluation_policy_sha256,
            "geometry_matcher_contract_sha256": (self.geometry_matcher_contract_sha256),
            "geometry_diagnostics_sha256": self.geometry_diagnostics_sha256,
            "invalid_attempt_rate": self.invalid_attempt_rate,
            "oracle_match_at_k": self.oracle_match_at_k,
            "predicted_match_at_k": self.predicted_match_at_k,
            "scaffold_diagnostics_sha256": self.scaffold_diagnostics_sha256,
            "scaffold_matcher_contract_sha256": (self.scaffold_matcher_contract_sha256),
            "scaffold_recall_at_k": self.scaffold_recall_at_k,
            "step": self.step,
            "training_loss": self.training_loss,
        }


def _manifest_sha256(value: object) -> str:
    to_json = getattr(value, "to_json", None)
    if not callable(to_json):
        raise TypeError("diagnostic evidence must provide canonical to_json()")
    payload = to_json()
    if not isinstance(payload, str):
        raise DomainError("diagnostic to_json() must return text")
    return sha256(payload.encode("utf-8")).hexdigest()


def _retrieval_at_k(
    diagnostics: ScaffoldRetrievalDiagnostics,
    k: int,
) -> RetrievalAtK:
    matches = tuple(item for item in diagnostics.at_k if item.k == k)
    if len(matches) != 1:
        raise DomainError(f"scaffold diagnostics must contain K={k} exactly once")
    return matches[0]


def _geometry_at_k(
    diagnostics: GeometryDecompositionDiagnostics,
    k: int,
) -> GeometryAtK:
    matches = tuple(item for item in diagnostics.at_k if item.k == k)
    if len(matches) != 1:
        raise DomainError(f"geometry diagnostics must contain K={k} exactly once")
    return matches[0]


def _snapshot_from_evaluation(
    evaluation: TinyOverfitEvaluation,
    protocol: TinyOverfitGateProtocol,
    evaluation_policy: DiagnosticEvaluationContract,
    scaffold_matcher_contract: EquivalenceMatcherContract,
    geometry_matcher_contract: EquivalenceMatcherContract,
) -> TinyOverfitSnapshot:
    evidence = (evaluation.composition, evaluation.scaffold, evaluation.geometry)
    expected_policy = evaluation_policy.policy_fingerprint
    if any(item.evaluation_contract.policy_fingerprint != expected_policy for item in evidence):
        raise DomainError("tiny-gate diagnostics do not share the frozen evaluation policy")
    checkpoints = {item.evaluation_contract.checkpoint_sha256 for item in evidence}
    if len(checkpoints) != 1:
        raise DomainError("tiny-gate diagnostics must share one checkpoint at each step")
    checkpoint = next(iter(checkpoints))
    if (
        evaluation.scaffold.matcher_contract.fingerprint
        != scaffold_matcher_contract.fingerprint
    ):
        raise DomainError("scaffold diagnostics use the wrong equivalence matcher contract")
    if (
        evaluation.geometry.matcher_contract.fingerprint
        != geometry_matcher_contract.fingerprint
    ):
        raise DomainError("geometry diagnostics use the wrong equivalence matcher contract")

    expected_query_ids = _identifier_fingerprint(protocol.example_ids)
    expected_count = len(protocol.example_ids)
    if (
        evaluation.composition.observation_count != expected_count
        or evaluation.composition.query_ids_sha256 != expected_query_ids
        or evaluation.scaffold.query_count != expected_count
        or evaluation.scaffold.query_ids_sha256 != expected_query_ids
        or evaluation.geometry.query_count != expected_count
        or evaluation.geometry.query_ids_sha256 != expected_query_ids
    ):
        raise DomainError("tiny-gate diagnostics do not match the ordered tiny corpus")

    scaffold = _retrieval_at_k(evaluation.scaffold, protocol.scaffold_k)
    geometry = _geometry_at_k(evaluation.geometry, protocol.match_k)
    invalid_rate = max(
        1.0 - scaffold.valid_attempt_rate.value,
        1.0 - geometry.oracle_valid_attempt_rate.value,
        1.0 - geometry.predicted_valid_attempt_rate.value,
    )
    return TinyOverfitSnapshot(
        step=evaluation.step,
        training_loss=evaluation.training_loss,
        composition_nll=evaluation.composition.mean_composition_nll.value,
        scaffold_recall_at_k=scaffold.recall.value,
        oracle_match_at_k=geometry.oracle_scaffold_match.value,
        predicted_match_at_k=geometry.predicted_scaffold_match.value,
        invalid_attempt_rate=invalid_rate,
        checkpoint_sha256=checkpoint,
        evaluation_policy_sha256=expected_policy,
        scaffold_matcher_contract_sha256=scaffold_matcher_contract.fingerprint,
        geometry_matcher_contract_sha256=geometry_matcher_contract.fingerprint,
        composition_diagnostics_sha256=_manifest_sha256(evaluation.composition),
        scaffold_diagnostics_sha256=_manifest_sha256(evaluation.scaffold),
        geometry_diagnostics_sha256=_manifest_sha256(evaluation.geometry),
    )


def _relative_reduction(initial: float, final: float) -> float:
    if initial == 0.0:
        return 1.0 if final == 0.0 else -1.0
    return (initial - final) / initial


def _gate_failures(
    snapshot: TinyOverfitSnapshot,
    initial_loss: float,
    initial_composition_nll: float,
    thresholds: TinyOverfitThresholds,
) -> tuple[str, ...]:
    values = (
        (snapshot.training_loss <= thresholds.max_training_loss, "training_loss"),
        (snapshot.composition_nll <= thresholds.max_composition_nll, "composition_nll"),
        (
            snapshot.scaffold_recall_at_k >= thresholds.min_scaffold_recall_at_k,
            "scaffold_recall_at_k",
        ),
        (
            snapshot.oracle_match_at_k >= thresholds.min_oracle_match_at_k,
            "oracle_match_at_k",
        ),
        (
            snapshot.predicted_match_at_k >= thresholds.min_predicted_match_at_k,
            "predicted_match_at_k",
        ),
        (
            snapshot.invalid_attempt_rate <= thresholds.max_invalid_attempt_rate,
            "invalid_attempt_rate",
        ),
        (
            _relative_reduction(initial_loss, snapshot.training_loss)
            >= thresholds.min_relative_loss_reduction,
            "relative_loss_reduction",
        ),
        (
            _relative_reduction(initial_composition_nll, snapshot.composition_nll)
            >= thresholds.min_relative_composition_nll_reduction,
            "relative_composition_nll_reduction",
        ),
    )
    return tuple(name for passed, name in values if not passed)


@dataclass(frozen=True, slots=True)
class TinyOverfitGateResult:
    passed: bool
    first_passing_step: int | None
    relative_loss_reduction: float
    relative_composition_nll_reduction: float
    final_failures: tuple[str, ...]
    history: tuple[TinyOverfitSnapshot, ...]
    protocol: TinyOverfitGateProtocol

    def __post_init__(self) -> None:
        if not self.history:
            raise DomainError("overfit result history cannot be empty")
        if not math.isfinite(self.relative_loss_reduction):
            raise DomainError("relative loss reduction must be finite")
        if not math.isfinite(self.relative_composition_nll_reduction):
            raise DomainError("relative composition NLL reduction must be finite")
        if self.passed != (self.first_passing_step is not None):
            raise DomainError("overfit pass flag and first passing step disagree")
        if self.passed and self.final_failures:
            raise DomainError("a passing overfit result cannot contain final failures")

    @property
    def final_snapshot(self) -> TinyOverfitSnapshot:
        return self.history[-1]

    def to_manifest(self) -> dict[str, object]:
        return {
            "final_failures": list(self.final_failures),
            "first_passing_step": self.first_passing_step,
            "history": [item.to_manifest() for item in self.history],
            "passed": self.passed,
            "protocol": self.protocol.to_manifest(),
            "relative_loss_reduction": self.relative_loss_reduction,
            "relative_composition_nll_reduction": (
                self.relative_composition_nll_reduction
            ),
            "schema": _OVERFIT_SCHEMA + ".result",
        }

    def to_json(self) -> str:
        return _json_text(self.to_manifest())


TrainStepCallback = Callable[[int], float | None]
EvaluationCallback = Callable[[int], TinyOverfitEvaluation]


def run_tiny_dataset_overfit_gate(
    protocol: TinyOverfitGateProtocol,
    *,
    evaluation_policy: DiagnosticEvaluationContract,
    scaffold_matcher_contract: EquivalenceMatcherContract,
    geometry_matcher_contract: EquivalenceMatcherContract,
    train_step: TrainStepCallback,
    evaluate: EvaluationCallback,
    stop_on_pass: bool = True,
) -> TinyOverfitGateResult:
    """Execute the callback-driven, framework-independent tiny-data gate.

    The initial state is evaluated at step zero.  Then exactly one update
    callback is made for each step until ``max_steps`` or, if enabled, the
    first passing scheduled evaluation at or after ``minimum_steps``.
    """

    if not isinstance(stop_on_pass, bool):
        raise DomainError("stop_on_pass must be Boolean")
    if not isinstance(evaluation_policy, DiagnosticEvaluationContract):
        raise TypeError("evaluation_policy must be a DiagnosticEvaluationContract")
    if not isinstance(scaffold_matcher_contract, EquivalenceMatcherContract):
        raise TypeError("scaffold_matcher_contract must be an EquivalenceMatcherContract")
    if not isinstance(geometry_matcher_contract, EquivalenceMatcherContract):
        raise TypeError("geometry_matcher_contract must be an EquivalenceMatcherContract")
    if evaluation_policy.policy_fingerprint != protocol.evaluation_policy_sha256:
        raise DomainError("evaluation policy differs from the tiny-gate protocol")
    if scaffold_matcher_contract.fingerprint != protocol.scaffold_matcher_contract_sha256:
        raise DomainError("scaffold matcher contract differs from the tiny-gate protocol")
    if geometry_matcher_contract.fingerprint != protocol.geometry_matcher_contract_sha256:
        raise DomainError("geometry matcher contract differs from the tiny-gate protocol")
    if evaluation_policy.dataset_manifest_sha256 != protocol.dataset_manifest_sha256:
        raise DomainError("dataset manifest differs from the tiny-gate protocol")
    if max(protocol.scaffold_k, protocol.match_k) > evaluation_policy.attempt_budget:
        raise DomainError("tiny-gate K cannot exceed the frozen generation attempt budget")

    initial_evaluation = evaluate(0)
    if initial_evaluation.step != 0:
        raise DomainError("evaluation callback returned a snapshot for the wrong step")
    initial = _snapshot_from_evaluation(
        initial_evaluation,
        protocol,
        evaluation_policy,
        scaffold_matcher_contract,
        geometry_matcher_contract,
    )
    history = [initial]
    first_passing_step: int | None = None
    consecutive_passing_evaluations = 0
    for step in range(1, protocol.max_steps + 1):
        reported_loss = train_step(step)
        if reported_loss is not None and not math.isfinite(reported_loss):
            raise DomainError("train-step callback returned a nonfinite loss")
        if step % protocol.evaluation_interval != 0 and step != protocol.max_steps:
            continue
        reported_evaluation = evaluate(step)
        if reported_evaluation.step != step:
            raise DomainError("evaluation callback returned a snapshot for the wrong step")
        snapshot = _snapshot_from_evaluation(
            reported_evaluation,
            protocol,
            evaluation_policy,
            scaffold_matcher_contract,
            geometry_matcher_contract,
        )
        history.append(snapshot)
        failures = _gate_failures(
            snapshot,
            initial.training_loss,
            initial.composition_nll,
            protocol.thresholds,
        )
        if step >= protocol.minimum_steps and not failures:
            consecutive_passing_evaluations += 1
        else:
            consecutive_passing_evaluations = 0
        if (
            consecutive_passing_evaluations
            >= protocol.minimum_consecutive_passing_evaluations
        ):
            if first_passing_step is None:
                first_passing_step = step
            if stop_on_pass:
                break

    final = history[-1]
    final_failures = _gate_failures(
        final,
        initial.training_loss,
        initial.composition_nll,
        protocol.thresholds,
    )
    # A metric crossing before minimum_steps cannot pass the protocol unless it
    # remains passing at a scheduled evaluation on/after that boundary.
    persistent_pass = (
        consecutive_passing_evaluations
        >= protocol.minimum_consecutive_passing_evaluations
    )
    if not persistent_pass:
        final_failures = (*final_failures, "consecutive_passing_evaluations")
    passed = first_passing_step is not None and persistent_pass and not final_failures
    return TinyOverfitGateResult(
        passed=passed,
        first_passing_step=first_passing_step if passed else None,
        relative_loss_reduction=_relative_reduction(initial.training_loss, final.training_loss),
        relative_composition_nll_reduction=_relative_reduction(
            initial.composition_nll,
            final.composition_nll,
        ),
        final_failures=() if passed else final_failures,
        history=tuple(history),
        protocol=protocol,
    )


__all__ = [
    "AcceptableTarget",
    "BootstrapProtocol",
    "CalibrationBin",
    "CompositionDiagnostics",
    "CompositionForecast",
    "DiagnosticCorpusFingerprints",
    "DiagnosticEvaluationContract",
    "EquivalenceMatcherContract",
    "GeometryAtK",
    "GeometryDecompositionCase",
    "GeometryDecompositionDiagnostics",
    "MetricEstimate",
    "RankedAttempt",
    "RankedRetrievalCase",
    "RetrievalAtK",
    "ScaffoldRetrievalDiagnostics",
    "TinyOverfitGateProtocol",
    "TinyOverfitGateResult",
    "TinyOverfitEvaluation",
    "TinyOverfitSnapshot",
    "TinyOverfitThresholds",
    "evaluate_composition_forecasts",
    "composition_forecasts_sha256",
    "evaluate_geometry_decomposition",
    "evaluate_scaffold_retrieval",
    "geometry_corpus_fingerprints",
    "matcher_callable_sha256",
    "paired_discordance_sample_size",
    "per_group_calibration",
    "retrieval_corpus_fingerprints",
    "run_tiny_dataset_overfit_gate",
]
