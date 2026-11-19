"""Resource exhaustion must abort, not inflate scientific invalidity counts."""

import numpy as np
import pytest
from test_pipeline import (
    _CompositionSampler,
    _CoverSampler,
    _Decoder,
    _GeometrySampler,
    _Verifier,
)

from crystalatlas.errors import DomainError, ResourceLimitError
from crystalatlas.periodic import images_within_cutoff
from crystalatlas.pipeline import GenerationEngine, GenerationMode, GenerationRequest


@pytest.mark.parametrize(
    "stage,method,failed_stage",
    [
        ("composition_sampler", "sample_unconditional", "composition"),
        ("cover_sampler", "sample_cover", "cover"),
        ("geometry_sampler", "sample_geometry", "geometry"),
        ("decoder", "decode", "decode"),
        ("verifier", "verify", "verify"),
    ],
)
@pytest.mark.parametrize("resource_failure", [True, False])
def test_failure_classification_at_every_stage(
    monkeypatch, stage, method, failed_stage, resource_failure
):
    components = dict(
        composition_sampler=_CompositionSampler(),
        cover_sampler=_CoverSampler(),
        geometry_sampler=_GeometrySampler(),
        decoder=_Decoder(),
        verifier=_Verifier(),
    )
    error = (
        ResourceLimitError("test work budget")
        if resource_failure
        else DomainError("test defect")
    )
    calls = []

    def fail(*args, **kwargs):
        calls.append(1)
        raise error

    monkeypatch.setattr(components[stage], method, fail)
    engine = GenerationEngine(**components)
    request = GenerationRequest(mode=GenerationMode.DNG, samples=3, seed=42)
    if resource_failure:
        with pytest.raises(ResourceLimitError) as caught:
            engine.sample(request)
        assert caught.value is error
        assert len(calls) == 1  # No retry or replacement of failed work.
    else:
        traces = engine.sample(request)
        assert len(traces) == len(calls) == 3
        assert all(t.failed_stage == failed_stage and not t.succeeded for t in traces)


@pytest.mark.parametrize("mode", list(GenerationMode))
def test_actual_periodic_budget_error_propagates_in_all_modes(mode):
    class LimitedGeometry(_GeometrySampler):
        def sample_geometry(self, composition, cover, rng):
            images_within_cutoff([0.0, 0.0, 0.0], np.eye(3), 1.0, max_search_points=1)
            raise AssertionError("budget exhaustion was ignored")

    engine = GenerationEngine(
        composition_sampler=_CompositionSampler(),
        cover_sampler=_CoverSampler(),
        geometry_sampler=LimitedGeometry(),
        decoder=_Decoder(),
        verifier=_Verifier(),
    )
    kwargs = (
        {"species": (3, 8)}
        if mode is GenerationMode.SPECIES_DNG
        else ({"composition": 11} if mode is GenerationMode.CSP else {})
    )
    with pytest.raises(ResourceLimitError):
        engine.sample(GenerationRequest(mode=mode, seed=7, **kwargs))
