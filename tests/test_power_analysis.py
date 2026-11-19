"""Sample size for paired oracle-versus-predicted contrasts (review m7)."""

from __future__ import annotations

import pytest

from crystalatlas.diagnostics import paired_discordance_sample_size
from crystalatlas.errors import DomainError


def test_review_example_needs_about_six_hundred_queries() -> None:
    # (z_0.975 + z_0.8)^2 * 0.2 / 0.05^2 = 627.91..., rounded up.
    assert paired_discordance_sample_size(
        effect=0.05, discordant_fraction=0.2, alpha=0.05, power=0.8
    ) == 628


def test_sample_size_scales_as_discordance_over_effect_squared() -> None:
    base = paired_discordance_sample_size(effect=0.1, discordant_fraction=0.2)
    assert paired_discordance_sample_size(effect=0.05, discordant_fraction=0.2) >= 4 * base - 4
    assert paired_discordance_sample_size(effect=0.1, discordant_fraction=0.4) >= 2 * base - 2


def test_invalid_designs_are_rejected() -> None:
    for kwargs in (
        {"effect": 0.0, "discordant_fraction": 0.2},
        {"effect": 0.3, "discordant_fraction": 0.2},  # |delta| cannot exceed p_d
        {"effect": 0.05, "discordant_fraction": 1.5},
        {"effect": 0.05, "discordant_fraction": 0.2, "alpha": 0.0},
        {"effect": 0.05, "discordant_fraction": 0.2, "power": 1.0},
    ):
        with pytest.raises(DomainError):
            paired_discordance_sample_size(**kwargs)  # type: ignore[arg-type]
