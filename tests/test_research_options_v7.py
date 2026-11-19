"""Declared opt-in mechanisms of the research runner (program R3).

``scripts/train_research_v7.py`` maps ``--options`` to fields of
:class:`ModelConfig` and :class:`CFMDequantization`.  These tests fix the
mapping: the empty set is the legacy configuration, the result does not depend
on the order of the list, and a repeated, unknown or conflicting declaration is
refused before any training starts.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


@pytest.fixture
def runner(monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    pytest.importorskip("torch")
    monkeypatch.syspath_prepend(str(SCRIPTS))
    import train_research_v7

    return train_research_v7


def test_default_is_the_legacy_configuration(runner) -> None:  # type: ignore[no-untyped-def]
    from crystalatlas.config import CrystalAtlasConfig, ModelConfig

    assert runner.resolve_options("default") == {"model": {}, "dequantization": {}}
    assert runner.research_dequantization() is None
    assert runner.research_dequantization("default") is None
    for profile in runner.PROFILES:
        for domain in runner.DOMAINS:
            base = runner.DOMAINS[domain]()
            legacy = CrystalAtlasConfig(
                domain=dataclasses.replace(
                    base, dng_element_vocabulary=base.element_vocabulary
                ),
                model=ModelConfig(**runner.PROFILES[profile]),
            )
            for config in (
                runner.model_config(profile, domain),
                runner.model_config(profile, domain, "default"),
            ):
                assert config == legacy
                assert config.fingerprint == legacy.fingerprint


def test_every_option_names_real_fields_of_its_target(runner) -> None:  # type: ignore[no-untyped-def]
    from crystalatlas.config import ModelConfig
    from crystalatlas.training_objectives import CFMDequantization

    targets = {
        "model": {item.name for item in dataclasses.fields(ModelConfig)},
        "dequantization": {item.name for item in dataclasses.fields(CFMDequantization)},
    }
    for name, option in runner.OPTIONS.items():
        assert option.fields, name
        assert set(option.fields) <= targets[option.target], name
        if option.parameter is not None:
            assert option.parameter in option.fields, name
            assert isinstance(option.fields[option.parameter], float), name
    for name, members in runner.OPTION_SETS.items():
        assert name not in runner.OPTIONS
        assert len(set(members)) == len(members)
        assert set(members) <= set(runner.OPTIONS)


def test_review_set_declares_the_review_mechanisms(runner) -> None:  # type: ignore[no-untyped-def]
    from crystalatlas.training_objectives import CFMDequantization

    resolved = runner.resolve_options("review")
    assert resolved == {
        "model": {
            "geometry_expansion_cell": "primitive-v2",
            "geometry_radial_gaussian_widths": (0.5, 1.0, 2.0),
            "selector_symmetry_descriptors": True,
            "selector_row_geometry_features": True,
            "geometry_residual_gauge": "normalizer-v3",
        },
        "dequantization": {
            "intrinsic_kernel": "cartesian-v2",
            "sigma_intrinsic": 0.05,
            "basis_augmentation": "shear-v1",
        },
    }
    config = runner.model_config("research-small", "v7", "review")
    legacy = runner.model_config("research-small", "v7")
    for name, value in resolved["model"].items():
        assert getattr(config.model, name) == value
        assert getattr(legacy.model, name) != value
    assert config.fingerprint != legacy.fingerprint
    assert config.domain == legacy.domain
    assert runner.research_dequantization("review") == CFMDequantization(
        intrinsic_kernel="cartesian-v2", sigma_intrinsic=0.05, basis_augmentation="shear-v1"
    )
    # A set plus a further option; the dropout option reaches the model only.
    extended = runner.resolve_options("review,dropout")
    assert extended["model"] == {**resolved["model"], "dropout": 0.1}
    assert extended["dequantization"] == resolved["dequantization"]


def test_resolution_does_not_depend_on_the_list_order(runner) -> None:  # type: ignore[no-untyped-def]
    forward = "radial,shear,cartesian-kernel=0.08,selector-rows"
    backward = ",".join(reversed(forward.split(",")))
    assert runner.resolve_options(forward) == runner.resolve_options(backward)
    assert list(runner.resolve_options(forward)["model"]) == list(
        runner.resolve_options(backward)["model"]
    )
    assert (
        runner.model_config("research-small", "v7", forward).fingerprint
        == runner.model_config("research-small", "v7", backward).fingerprint
    )


def test_declared_scales(runner) -> None:  # type: ignore[no-untyped-def]
    kernel = runner.research_dequantization("cartesian-kernel=0.08")
    assert kernel is not None
    assert kernel.intrinsic_kernel == "cartesian-v2"
    assert kernel.sigma_intrinsic == 0.08
    assert kernel.basis_augmentation == "none"
    assert kernel.sigma_lattice == 0.05  # the lattice scale is not an option
    assert runner.research_dequantization("cartesian-kernel").sigma_intrinsic == 0.05
    assert runner.model_config("research-small", "v7", "dropout=0.25").model.dropout == 0.25
    assert runner.model_config("research-small", "v7", "dropout").model.dropout == 0.1
    assert runner.research_dequantization("dropout") is None


def test_research_options_outside_the_review_set(runner) -> None:  # type: ignore[no-untyped-def]
    legacy = runner.model_config("research-small", "v7")
    assert legacy.model.element_embedding == "learned-v1"
    assert legacy.model.geometry_pair_carrier_smoothing_length is None
    periodic = runner.model_config("research-small", "v7", "periodic-elements")
    assert periodic.model.element_embedding == "periodic-v1"
    smoothed = runner.model_config("research-small", "v7", "carrier-smoothing")
    assert smoothed.model.geometry_pair_carrier_smoothing_length == 0.2
    wider = runner.model_config("research-small", "v7", "carrier-smoothing=0.35")
    assert wider.model.geometry_pair_carrier_smoothing_length == 0.35
    fingerprints = {item.fingerprint for item in (legacy, periodic, smoothed, wider)}
    assert len(fingerprints) == 4
    for name in ("periodic-elements", "carrier-smoothing"):
        assert name not in runner.OPTION_SETS["review"]
        assert runner.research_dequantization(name) is None
    # Each differs from the legacy model in its own field only.
    for config, field in (
        (periodic, "element_embedding"),
        (smoothed, "geometry_pair_carrier_smoothing_length"),
    ):
        assert dataclasses.replace(
            config.model, **{field: getattr(legacy.model, field)}
        ) == legacy.model


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("", "comma list"),
        ("radial,", "unknown option"),
        ("radail", "unknown option"),
        ("review=1", "unknown option"),
        ("radial,radial", "declared twice"),
        ("review,shear", "declared twice"),
        ("cartesian-kernel,cartesian-kernel=0.08", "declared twice"),
        ("shear=0.5", "takes no value"),
        ("cartesian-kernel=", "needs a number"),
        ("cartesian-kernel=wide", "needs a number"),
        ("cartesian-kernel=inf", "finite"),
        ("dropout=nan", "finite"),
    ],
)
def test_refused_declarations(runner, text: str, message: str) -> None:  # type: ignore[no-untyped-def]
    from crystalatlas.errors import DomainError

    with pytest.raises(DomainError, match=message):
        runner.resolve_options(text)


def test_out_of_range_scales_are_refused_by_the_target(runner) -> None:  # type: ignore[no-untyped-def]
    from crystalatlas.errors import DomainError

    with pytest.raises(DomainError, match="positive"):
        runner.research_dequantization("cartesian-kernel=0")
    with pytest.raises(DomainError, match="dropout"):
        runner.model_config("research-small", "v7", "dropout=1.0")
    with pytest.raises(DomainError, match="geometry_pair_carrier_smoothing_length"):
        runner.model_config("research-small", "v7", "carrier-smoothing=0.001")
