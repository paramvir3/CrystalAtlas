from __future__ import annotations

from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from crystalatlas.catalogue import (  # noqa: E402
    FIXTURE_EXPECTED_ROWS,
    MINIMAL_CATALOGUE,
    CertifiedCatalogue,
)
from crystalatlas.composition import Composition  # noqa: E402
from crystalatlas.config import (  # noqa: E402
    CrystalAtlasConfig,
    DomainConfig,
    ModelConfig,
    NumericsConfig,
)
from crystalatlas.decoder import CrystalScaffold, OrbitAssignment  # noqa: E402
from crystalatlas.errors import (  # noqa: E402
    DomainError,
    GeneratedSampleError,
    InvariantViolation,
)
from crystalatlas.flow import canonicalize_torus  # noqa: E402
from crystalatlas.geometry import ContinuousCover, IntrinsicCoverLayout  # noqa: E402
from crystalatlas.learned import CrystalAtlasModel, LearnedCrystalGenerator  # noqa: E402
from crystalatlas.pipeline import GenerationMode, GenerationRequest  # noqa: E402
from crystalatlas.provenance import ProvenanceRecord  # noqa: E402
from crystalatlas.training import (  # noqa: E402
    CheckpointError,
    ScientificManifest,
    load_checkpoint,
    model_state_fingerprint,
    save_checkpoint,
)
from crystalatlas.training_objectives import (  # noqa: E402
    CrystalAtlasTrainingObjective,
    ObjectiveDataMeasure,
)

pytestmark = pytest.mark.ml


def _tiny_config() -> CrystalAtlasConfig:
    return CrystalAtlasConfig(
        domain=DomainConfig(
            element_vocabulary=(6, 8, 14),
            dng_element_vocabulary=(6, 8),
            max_distinct_elements=2,
            max_raw_atoms=4,
            max_formula_units=2,
            metric_radius=1.0,
            interaction_cutoff_angstrom=2.5,
            lattice_chart="ball-v1",  # fixture designed for the radius-1 ball
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


def _manifest(model: CrystalAtlasModel) -> ScientificManifest:
    return model.scientific_manifest(
        data_contract={"measure": "unit-test", "split_sha256": "1" * 64}
    )


def _provenance(config: CrystalAtlasConfig) -> ProvenanceRecord:
    return ProvenanceRecord(
        created_utc="2026-08-25T00:00:00+00:00",
        config_fingerprint=config.fingerprint,
        dataset_hashes=(),
        catalogue_hash=None,
        package_version="0.1.0a1",
        python_version="test",
        platform="test",
        git_revision=None,
        random_seed=31,
        task="learned-checkpoint-test",
    )


def test_bridge_numerical_policy_is_bound_to_model_and_objective_identity() -> None:
    model = CrystalAtlasModel.fixture(_tiny_config())
    manifest = _manifest(model)
    objective = CrystalAtlasTrainingObjective(model, ObjectiveDataMeasure(3, 1, "2" * 64))
    assert manifest.representation["schema"] == "crystalatlas.representation.v2"
    assert objective.contract["schema"] == "crystalatlas.training-objective.v4"
    policy = manifest.representation["geometry_bridge_numerics"]
    assert policy["lattice_chart"] == "canonical-float64-matrix-exp-then-master-cast.v1"
    assert policy["chart_constant_storage"] == "float64-bits-in-int64-buffers.v1"
    assert policy == objective.contract["geometry_bridge_numerics"]


def test_old_chart_checkpoint_identity_cannot_bypass_live_model_validation(
    tmp_path,
    monkeypatch,
) -> None:
    config = _tiny_config()
    model = CrystalAtlasModel.fixture(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    current = _manifest(model)
    legacy_payload = current.to_dict()
    legacy_payload["representation"]["schema"] = "crystalatlas.representation.v1"
    del legacy_payload["representation"]["geometry_bridge_numerics"]
    legacy = ScientificManifest(**legacy_payload)
    assert current.fingerprint != legacy.fingerprint
    with pytest.raises(DomainError, match="does not match the live model"):
        model.validate_scientific_manifest(legacy)
    rejected_path = tmp_path / "must-not-publish.pt"
    with pytest.raises(CheckpointError, match="does not match the live model"):
        save_checkpoint(
            rejected_path,
            model=model,
            optimizer=optimizer,
            config=config,
            scientific_manifest=legacy,
            provenance=_provenance(config),
            step=0,
        )
    assert not rejected_path.exists()

    # Emulate the old writer's declared identity, without changing package
    # source or pretending the new chart is the old numerical implementation.
    old_path = tmp_path / "old-numerical-identity.pt"
    with monkeypatch.context() as patch:
        patch.setattr(model, "validate_scientific_manifest", lambda manifest: None)
        receipt = save_checkpoint(
            old_path,
            model=model,
            optimizer=optimizer,
            config=config,
            scientific_manifest=legacy,
            provenance=_provenance(config),
            step=0,
        )
    state_before = model_state_fingerprint(model)
    rng_before = torch.get_rng_state().clone()
    for expected, message in (
        (current, "different scientific manifest"),
        (legacy, "does not match the live model"),
    ):
        with pytest.raises(CheckpointError, match=message):
            load_checkpoint(
                old_path,
                model=model,
                optimizer=optimizer,
                expected_scientific_manifest=expected,
                expected_config=config,
                expected_sha256=receipt.sha256,
                restore_rng=True,
            )
        assert model_state_fingerprint(model) == state_before
        torch.testing.assert_close(torch.get_rng_state(), rng_before, rtol=0, atol=0)


def test_learned_inference_requires_explicit_checkpoint_identity() -> None:
    model = CrystalAtlasModel(_tiny_config(), catalogue=MINIMAL_CATALOGUE).double()
    with pytest.raises(DomainError, match="checkpoint receipt"):
        LearnedCrystalGenerator(model)


def test_model_never_silently_substitutes_the_fixture_catalogue() -> None:
    with pytest.raises(DomainError, match="catalogue must be explicit"):
        CrystalAtlasModel(_tiny_config())
    fixture = CrystalAtlasModel.fixture(_tiny_config())
    assert fixture.catalogue is MINIMAL_CATALOGUE


def test_random_initialization_smoke_connects_csp_to_flow_and_decoder() -> None:
    torch.manual_seed(13)
    model = CrystalAtlasModel(_tiny_config(), catalogue=MINIMAL_CATALOGUE).double()
    generator = LearnedCrystalGenerator(model, allow_untrained=True)
    request = GenerationRequest(
        mode=GenerationMode.CSP,
        composition=Composition.from_counts({6: 1, 8: 1}),
        samples=1,
        seed=29,
    )
    samples = generator.require_samples(request)
    assert len(samples) == 1
    assert samples[0].decoded.audit.passed
    assert samples[0].physical_primitive.translation_index == (
        samples[0].decoded.translation_index
    )
    assert len(samples[0].physical_primitive.atomic_numbers) * samples[
        0
    ].physical_primitive.translation_index == len(samples[0].decoded.atomic_numbers)
    assert samples[0].checkpoint_id is None
    assert not generator.is_trained_inference
    assert model.training


def test_oracle_cover_geometry_sampling_is_a_public_single_attempt() -> None:
    torch.manual_seed(13)
    model = CrystalAtlasModel(_tiny_config(), catalogue=MINIMAL_CATALOGUE).double()
    generator = LearnedCrystalGenerator(model, allow_untrained=True)
    learned = generator.require_samples(
        GenerationRequest(
            mode=GenerationMode.CSP,
            composition=Composition.from_counts({6: 1, 8: 1}),
            samples=1,
            seed=29,
        )
    )[0]
    geometry, decoded = generator.sample_geometry_given_cover(
        learned.cover_sample.cover,
        seed=31,
    )
    np.testing.assert_array_equal(
        geometry.lattice_latent,
        decoded.geometry.lattice_latent,
    )
    np.testing.assert_array_equal(
        geometry.intrinsic_coordinates,
        decoded.geometry.intrinsic_coordinates,
    )
    assert decoded.cover == learned.cover_sample.cover
    assert decoded.audit.passed
    assert model.training


def test_trained_inference_requires_a_live_state_bound_checkpoint(tmp_path) -> None:
    config = _tiny_config()
    model = CrystalAtlasModel(config, catalogue=MINIMAL_CATALOGUE).double()
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    checkpoint = tmp_path / "model.pt"
    info = save_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        config=config,
        scientific_manifest=_manifest(model),
        provenance=_provenance(config),
        step=0,
    )
    with torch.no_grad():
        next(model.parameters()).add_(1.0)
    generator = LearnedCrystalGenerator.from_checkpoint(
        model,
        checkpoint,
        scientific_manifest=_manifest(model),
        expected_sha256=info.sha256,
    )
    assert generator.is_trained_inference
    assert generator.checkpoint_id == info.sha256
    with pytest.raises(TypeError, match="LoadedCheckpoint"):
        LearnedCrystalGenerator(model, validated_checkpoint="fake")
    with torch.no_grad():
        next(model.parameters()).add_(1.0)
    request = GenerationRequest(
        mode=GenerationMode.CSP,
        composition=Composition.from_counts({6: 1, 8: 1}),
        samples=1,
        seed=37,
    )
    with pytest.raises(InvariantViolation, match="changed after"):
        generator.sample(request)


def test_checkpoint_binding_rejects_a_different_live_catalogue(tmp_path) -> None:
    config = _tiny_config()
    model = CrystalAtlasModel(config, catalogue=MINIMAL_CATALOGUE).double()
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    checkpoint = tmp_path / "model.pt"
    info = save_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        config=config,
        scientific_manifest=_manifest(model),
        provenance=_provenance(config),
        step=0,
    )
    changed_catalogue = CertifiedCatalogue(
        MINIMAL_CATALOGUE.settings,
        FIXTURE_EXPECTED_ROWS,
        certification_scope="semantically different live catalogue",
    )
    changed_model = CrystalAtlasModel(config, catalogue=changed_catalogue).double()
    changed_manifest = changed_model.scientific_manifest(
        data_contract={"measure": "unit-test", "split_sha256": "1" * 64}
    )
    with pytest.raises(CheckpointError, match="different scientific manifest"):
        LearnedCrystalGenerator.from_checkpoint(
            changed_model,
            checkpoint,
            scientific_manifest=changed_manifest,
            expected_sha256=info.sha256,
        )


def test_csp_support_contract_rejects_out_of_vocabulary_composition() -> None:
    model = CrystalAtlasModel(_tiny_config(), catalogue=MINIMAL_CATALOGUE).double()
    generator = LearnedCrystalGenerator(model, allow_untrained=True)
    traces = generator.sample(
        GenerationRequest(
            mode=GenerationMode.CSP,
            composition=Composition.from_counts({1: 1}),
            samples=1,
            seed=41,
        )
    )
    assert traces[0].failed_stage == "composition"


def test_geometry_cfm_objective_uses_current_lattice_pullback_and_backpropagates() -> None:
    torch.manual_seed(17)
    model = CrystalAtlasModel(_tiny_config(), catalogue=MINIMAL_CATALOGUE).double()
    generator = LearnedCrystalGenerator(model, allow_untrained=True)
    scaffold = CrystalScaffold.from_mapping(
        nu=2,
        space_group="P-1",
        reduced_composition={8: 1},
        orbits=(
            OrbitAssignment(
                "2i",
                8,
                (Fraction(1, 7), Fraction(2, 7), Fraction(3, 7)),
            ),
        ),
    )
    layout = model.layout(ContinuousCover.from_exact_scaffold(scaffold))
    target_metric = layout.chart.decode_latent(np.zeros(layout.lattice_dimension))
    result = generator.geometry_cfm_objective(
        scaffold,
        target_metric,
        time=torch.tensor(0.37, dtype=torch.float64),
        sigma_z=0.05,
        sigma_eta=0.03,
        generator=torch.Generator().manual_seed(101),
    )
    assert torch.isfinite(result.loss.total)
    result.loss.total.backward()
    gradients = [
        parameter.grad
        for parameter in model.geometry_field.parameters()
        if parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_packed_geometry_cfm_batches_one_exact_cover_topology() -> None:
    torch.manual_seed(23)
    model = CrystalAtlasModel(_tiny_config(), catalogue=MINIMAL_CATALOGUE).double()
    generator = LearnedCrystalGenerator(model, allow_untrained=True)
    scaffolds = (
        CrystalScaffold.from_mapping(
            nu=2,
            space_group="P-1",
            reduced_composition={8: 1},
            orbits=(
                OrbitAssignment(
                    "2i",
                    8,
                    (Fraction(1, 7), Fraction(2, 7), Fraction(3, 7)),
                ),
            ),
        ),
        CrystalScaffold.from_mapping(
            nu=2,
            space_group="P-1",
            reduced_composition={8: 1},
            orbits=(
                OrbitAssignment(
                    "2i",
                    8,
                    (Fraction(2, 9), Fraction(1, 9), Fraction(4, 9)),
                ),
            ),
        ),
    )
    layout = model.layout(ContinuousCover.from_exact_scaffold(scaffolds[0]))
    metrics = (
        layout.chart.decode_latent(np.zeros(layout.lattice_dimension)),
        layout.chart.decode_latent(np.full(layout.lattice_dimension, 0.05)),
    )
    result = generator.geometry_cfm_objective_batch(
        scaffolds,
        metrics,
        time=torch.tensor([0.25, 0.75], dtype=torch.float64),
        sigma_z=0.05,
        sigma_eta=0.03,
        generator=torch.Generator().manual_seed(113),
    )
    assert result.loss.per_sample.shape == (2,)
    assert result.prediction.z.shape == (2, layout.lattice_dimension)
    assert result.prediction.eta.shape == (2, layout.intrinsic_dimension)
    assert torch.isfinite(result.loss.total)
    result.loss.total.backward()
    gradients = [
        parameter.grad
        for parameter in model.geometry_field.parameters()
        if parameter.grad is not None
    ]
    assert gradients
    assert all(bool(torch.isfinite(gradient).all().item()) for gradient in gradients)


def test_packed_geometry_cfm_bfloat16_autocast_preserves_master_algebra() -> None:
    """AMP is restricted to the neural map, never the crystal/CFM algebra."""

    torch.manual_seed(127)
    model = CrystalAtlasModel(_tiny_config(), catalogue=MINIMAL_CATALOGUE).float()
    generator = LearnedCrystalGenerator(model, allow_untrained=True)
    scaffold = CrystalScaffold.from_mapping(
        nu=2,
        space_group="P-1",
        reduced_composition={8: 1},
        orbits=(
            OrbitAssignment(
                "2i",
                8,
                (Fraction(1, 7), Fraction(2, 7), Fraction(3, 7)),
            ),
        ),
    )
    layout = model.layout(ContinuousCover.from_exact_scaffold(scaffold))
    metric = layout.chart.decode_latent(np.zeros(layout.lattice_dimension))
    neural_dtypes: list[torch.dtype] = []

    def observe_neural_head(_module, _inputs, output):  # type: ignore[no-untyped-def]
        neural_dtypes.append(output.dtype)

    hook = model.geometry_field.cartesian_head.register_forward_hook(observe_neural_head)
    try:
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            result = generator.geometry_cfm_objective_batch(
                (scaffold, scaffold),
                (metric, metric),
                time=torch.tensor([0.25, 0.75], dtype=torch.float32),
                sigma_z=0.05,
                sigma_eta=0.03,
                generator=torch.Generator().manual_seed(131),
            )
    finally:
        hook.remove()

    # This assertion proves the regression did execute a real BF16 neural
    # forward rather than accidentally disabling autocast around the model.
    assert neural_dtypes == [torch.bfloat16]
    master_tensors = (
        result.path.state.z,
        result.path.state.eta,
        result.path.target_velocity.z,
        result.path.target_velocity.eta,
        result.prediction.z,
        result.prediction.eta,
        result.loss.total,
        result.loss.per_sample,
    )
    assert all(value.dtype == torch.float32 for value in master_tensors)
    assert all(bool(torch.isfinite(value).all().item()) for value in master_tensors)

    result.loss.total.backward()
    gradients = tuple(
        parameter.grad
        for parameter in model.geometry_field.parameters()
        if parameter.grad is not None
    )
    assert gradients
    assert all(gradient.dtype == torch.float32 for gradient in gradients)
    assert all(bool(torch.isfinite(gradient).all().item()) for gradient in gradients)


def test_packed_geometry_cfm_preserves_each_sample_chemical_identity() -> None:
    model = CrystalAtlasModel(_tiny_config(), catalogue=MINIMAL_CATALOGUE).double()
    generator = LearnedCrystalGenerator(model, allow_untrained=True)

    def scaffold(species: int) -> CrystalScaffold:
        return CrystalScaffold.from_mapping(
            nu=2,
            space_group="P-1",
            reduced_composition={species: 1},
            orbits=(
                OrbitAssignment(
                    "2i",
                    species,
                    (Fraction(1, 7), Fraction(2, 7), Fraction(3, 7)),
                ),
            ),
        )

    scaffolds = (scaffold(6), scaffold(8))
    metrics = tuple(
        model.layout(ContinuousCover.from_exact_scaffold(item)).chart.decode_latent(np.zeros(6))
        for item in scaffolds
    )
    captured: dict[str, torch.Tensor] = {}

    def observing_field(*args, **kwargs):  # type: ignore[no-untyped-def]
        captured["atomic_numbers"] = kwargs["atomic_numbers"].detach().clone()
        captured["composition_atomic_numbers"] = (
            kwargs["composition_atomic_numbers"].detach().clone()
        )
        return model.geometry_field(*args, **kwargs)

    result = generator.geometry_cfm_objective_batch(
        scaffolds,
        metrics,
        time=torch.tensor([0.25, 0.75], dtype=torch.float64),
        sigma_z=0.05,
        sigma_eta=0.03,
        generator=torch.Generator().manual_seed(109),
        geometry_field=observing_field,
    )
    assert torch.isfinite(result.loss.total)
    torch.testing.assert_close(
        captured["atomic_numbers"],
        torch.tensor([[6, 6], [8, 8]], dtype=torch.long),
    )
    torch.testing.assert_close(
        captured["composition_atomic_numbers"],
        torch.tensor([[6], [8]], dtype=torch.long),
    )


def test_geometry_objective_rejects_specialized_and_hidden_translation_targets() -> None:
    model = CrystalAtlasModel(_tiny_config(), catalogue=MINIMAL_CATALOGUE).double()
    generator = LearnedCrystalGenerator(model, allow_untrained=True)
    specialized = CrystalScaffold.from_mapping(
        nu=2,
        space_group="P-1",
        reduced_composition={8: 1},
        orbits=(OrbitAssignment("2i", 8, (Fraction(0),) * 3),),
    )
    with pytest.raises(InvariantViolation, match="below multiplicity"):
        generator.geometry_cfm_objective(
            specialized,
            np.eye(3) * 9.0,
            time=torch.tensor(0.5, dtype=torch.float64),
            sigma_z=0.05,
            sigma_eta=0.05,
        )

    hidden_translation = CrystalScaffold.from_mapping(
        nu=2,
        space_group="P-1",
        reduced_composition={8: 1},
        orbits=(
            OrbitAssignment("1a", 8, ()),
            OrbitAssignment("1d", 8, ()),
        ),
    )
    with pytest.raises(DomainError, match="hidden translations"):
        generator.geometry_cfm_objective(
            hidden_translation,
            np.eye(3) * 9.0,
            time=torch.tensor(0.5, dtype=torch.float64),
            sigma_z=0.05,
            sigma_eta=0.05,
        )


@pytest.mark.parametrize("batched", [False, True])
def test_geometry_objective_applies_residual_action_to_complete_noisy_path(
    monkeypatch, batched
):
    import crystalatlas.learned as learned_module

    config = _tiny_config()
    config = replace(config, domain=replace(config.domain, max_formula_units=3))
    model = CrystalAtlasModel(config, catalogue=MINIMAL_CATALOGUE).double()
    generator = LearnedCrystalGenerator(model, allow_untrained=True)
    scaffold = CrystalScaffold.from_mapping(
        nu=3,
        space_group="P1",
        reduced_composition={6: 1},
        orbits=tuple(
            OrbitAssignment("1a", 6, tuple(Fraction(value, 100) for value in position))
            for position in ((0, 0, 0), (17, 23, 31), (41, 43, 47))
        ),
    )
    layout = model.layout(ContinuousCover.from_exact_scaffold(scaffold))
    mean = torch.as_tensor(layout.encode_exact_parameters(scaffold), dtype=torch.float64)
    metric = layout.chart.decode_latent(np.zeros(layout.lattice_dimension))
    monkeypatch.setattr(generator, "_sample_residual_permutation", lambda *a, **k: (1, 0, 2))
    real_sampler = learned_module.sample_dequantized_endpoint
    real_path = learned_module.conditional_flow_path
    captured = []

    def capture(z, eta, period, **kwargs):
        expected_mean = mean.expand_as(eta)
        torch.testing.assert_close(eta, expected_mean, rtol=0, atol=0)
        assert kwargs.get("torus_action") is None  # apply once to the complete path below
        return real_sampler(z, eta, period, **kwargs)

    def capture_path(z0, eta0, z1, eta1, time, period, **kwargs):
        action = kwargs["torus_action"]
        plain = real_path(z0, eta0, z1, eta1, time, period)
        path = real_path(z0, eta0, z1, eta1, time, period, **kwargs)
        expected = canonicalize_torus(
            torch.matmul(plain.state.eta.unsqueeze(-2), action).squeeze(-2), period
        )
        torch.testing.assert_close(path.state.eta, expected)
        torch.testing.assert_close(
            path.target_velocity.eta,
            torch.matmul(plain.target_velocity.eta.unsqueeze(-2), action).squeeze(-2),
        )
        captured.append(path)
        return path

    monkeypatch.setattr(learned_module, "sample_dequantized_endpoint", capture)
    monkeypatch.setattr(learned_module, "conditional_flow_path", capture_path)
    kwargs = dict(
        time=torch.tensor(0.5, dtype=torch.float64),
        sigma_z=0.05,
        sigma_eta=0.1,
        generator=torch.Generator().manual_seed(631),
    )
    if batched:
        result = generator.geometry_cfm_objective_batch(
            (scaffold, scaffold), (metric, metric), **kwargs
        )
    else:
        result = generator.geometry_cfm_objective(scaffold, metric, **kwargs)
    assert len(captured) == 1
    torch.testing.assert_close(result.path.state.eta, captured[0].state.eta)
    assert torch.isfinite(result.loss.total)
    result.loss.total.backward()
    assert any(p.grad is not None for p in model.geometry_field.parameters())


@pytest.mark.parametrize("budget_kind", ["pairs", "workspace"])
def test_generation_chunks_candidate_pool_without_changing_attempts(budget_kind, monkeypatch):
    catalogue = CertifiedCatalogue(
        {"P1": MINIMAL_CATALOGUE.get("P1")},
        {"P1": FIXTURE_EXPECTED_ROWS["P1"]},
        certification_scope="P1 generation batching fixture",
    )
    config = _tiny_config()
    config = replace(
        config,
        domain=replace(config.domain, max_raw_atoms=2, max_formula_units=1),
        numerics=replace(config.numerics, ode_steps=1),
    )
    torch.manual_seed(13)
    reference = CrystalAtlasModel(config, catalogue=catalogue).double()
    workspace = reference.geometry_field.transformer._estimated_workspace_bytes(
        batch_size=1, atom_count=2, dtype=torch.float64
    )
    numerics = replace(
        config.numerics,
        max_dense_pair_interactions=8 if budget_kind == "pairs" else 100,
        max_geometry_workspace_bytes=(
            2 * workspace if budget_kind == "workspace" else 2_000_000_000
        ),
    )
    bounded = CrystalAtlasModel(
        replace(config, numerics=numerics), catalogue=catalogue
    ).double()
    bounded.load_state_dict(reference.state_dict())
    assert (
        bounded.geometry_field.transformer.maximum_batch_size(atom_count=2, dtype=torch.float64)
        == 2
    )
    sizes = []
    hook = bounded.geometry_field.register_forward_pre_hook(
        lambda module, args: sizes.append(args[1].shape[0])
    )
    request = GenerationRequest(
        mode=GenerationMode.CSP,
        composition=Composition.from_counts({6: 1, 8: 1}),
        samples=5,
        seed=29,
    )
    expected = LearnedCrystalGenerator(reference, allow_untrained=True).sample(request)
    actual = LearnedCrystalGenerator(bounded, allow_untrained=True).sample(request)
    hook.remove()
    # Four RK4 evaluations for each of the chunks [2,2,1].
    assert sizes == [2] * 8 + [1] * 4
    assert len(actual) == 5
    for index, (first, second) in enumerate(zip(expected, actual, strict=True)):
        assert first.sample_index == second.sample_index == index
        assert first.seed == second.seed
        assert first.succeeded and second.succeeded
        assert first.sample.cover_sample == second.sample.cover_sample
        np.testing.assert_allclose(
            first.sample.geometry.lattice_latent,
            second.sample.geometry.lattice_latent,
            atol=1e-12,
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            first.sample.geometry.intrinsic_coordinates,
            second.sample.geometry.intrinsic_coordinates,
            atol=1e-12,
            rtol=1e-12,
        )

    decode = IntrinsicCoverLayout.decode
    decode_count = 0

    def fail_second_decode(layout, *args, **kwargs):
        nonlocal decode_count
        decode_count += 1
        if decode_count == 2:
            raise GeneratedSampleError("injected candidate conformance failure")
        return decode(layout, *args, **kwargs)

    monkeypatch.setattr(IntrinsicCoverLayout, "decode", fail_second_decode)
    traces = LearnedCrystalGenerator(bounded, allow_untrained=True).sample(request)
    assert len(traces) == decode_count == 5
    assert [trace.sample_index for trace in traces] == list(range(5))
    assert [trace.seed for trace in traces] == [trace.seed for trace in expected]
    assert [trace.succeeded for trace in traces] == [True, False, True, True, True]
    assert traces[1].failed_stage == "geometry/decode"


def test_a_decode_failure_keeps_its_slot_and_is_never_retried(monkeypatch) -> None:
    """E3 invalid-as-failure accounting (theory 4.9, mutation 18).

    One injected decode failure must surface as that attempt's failure.  The
    decoder runs exactly once per attempt, and every other attempt is unchanged.
    """

    from crystalatlas.errors import GeneratedSampleError
    from crystalatlas.geometry import IntrinsicCoverLayout

    torch.manual_seed(13)
    model = CrystalAtlasModel(_tiny_config(), catalogue=MINIMAL_CATALOGUE).double()
    generator = LearnedCrystalGenerator(model, allow_untrained=True)
    request = GenerationRequest(
        mode=GenerationMode.CSP,
        composition=Composition.from_counts({6: 1, 8: 1}),
        samples=3,
        seed=29,
    )
    original = IntrinsicCoverLayout.decode
    calls: list[int] = []

    def counting(self, geometry, **kwargs):  # type: ignore[no-untyped-def]
        calls.append(1)
        return original(self, geometry, **kwargs)

    monkeypatch.setattr(IntrinsicCoverLayout, "decode", counting)
    baseline = generator.sample(request)
    baseline_calls = len(calls)
    assert baseline_calls == 3 and all(trace.succeeded for trace in baseline)
    calls.clear()

    def failing_once(self, geometry, **kwargs):  # type: ignore[no-untyped-def]
        calls.append(1)
        if len(calls) == 1:
            raise GeneratedSampleError("injected decode failure")
        return original(self, geometry, **kwargs)

    monkeypatch.setattr(IntrinsicCoverLayout, "decode", failing_once)
    traces = generator.sample(request)
    assert len(calls) == baseline_calls
    failed = [trace for trace in traces if not trace.succeeded]
    assert len(failed) == 1
    assert failed[0].failed_stage == "geometry/decode"
    assert "injected decode failure" in (failed[0].error or "")
    for trace, reference in zip(traces, baseline, strict=True):
        if trace.succeeded:
            np.testing.assert_array_equal(
                trace.sample.geometry.lattice_latent, reference.sample.geometry.lattice_latent
            )


def test_three_modes_share_the_downstream_path_after_composition(monkeypatch) -> None:
    """D0b/S3 shared-path conformance (theory 4.9, mutations 21 and 24).

    Once c is fixed, DNG, S-DNG and CSP share one selector and one flow, with
    no task-mode or species-set input.  The same seeds must therefore give
    identical covers, log-probabilities and geometries.
    """

    torch.manual_seed(19)
    model = CrystalAtlasModel(_tiny_config(), catalogue=MINIMAL_CATALOGUE).double()
    generator = LearnedCrystalGenerator(model, allow_untrained=True)
    composition = Composition.from_counts({6: 1, 8: 1})
    monkeypatch.setattr(generator, "_composition", lambda request, stream, device: composition)
    requests = (
        GenerationRequest(mode=GenerationMode.DNG, samples=3, seed=5),
        GenerationRequest(mode=GenerationMode.SPECIES_DNG, species=(6, 8), samples=3, seed=5),
        GenerationRequest(mode=GenerationMode.CSP, composition=composition, samples=3, seed=5),
    )
    runs = [generator.sample(request) for request in requests]
    reference = runs[-1]
    assert all(trace.succeeded for trace in reference)
    for run in runs[:-1]:
        for trace, expected in zip(run, reference, strict=True):
            assert trace.succeeded
            assert trace.sample.cover_sample.cover == expected.sample.cover_sample.cover
            assert (
                trace.sample.cover_sample.log_probability
                == expected.sample.cover_sample.log_probability
            )
            np.testing.assert_array_equal(
                trace.sample.geometry.lattice_latent, expected.sample.geometry.lattice_latent
            )
            np.testing.assert_array_equal(
                trace.sample.geometry.intrinsic_coordinates,
                expected.sample.geometry.intrinsic_coordinates,
            )
