"""Primitive-cell atom bound (review finding B2).

The legacy bound limits conventional atoms (40), so the primitive capacity is
40 / I_g: 10 atoms for F-centred and about 13 for R-centred settings, against
MP-20's definition of at most 20 atoms per primitive cell.  With
``max_primitive_atoms`` set, a cover (nu, G) is admitted iff
``nu |c| <= max_primitive_atoms * I_G`` (and the conventional resource cap).
Sampler masks and training-target masks share one implementation, so the
likelihood normalizers cannot diverge.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from crystalatlas.composition import Composition
from crystalatlas.config import CrystalAtlasConfig, DomainConfig, ModelConfig
from crystalatlas.constraints import SuffixReachability
from crystalatlas.cover import canonical_rows
from crystalatlas.cover_legality import AtomBound, feasible_formula_units, legal_settings
from crystalatlas.errors import DomainError
from crystalatlas.full_catalogue import build_space_group_setting

_V6_REPORT = Path("artifacts/mp20_pilot_2026-09-25/preprocessing512_v6.json")


def _oracle(composition, nu, setting):  # type: ignore[no-untyped-def]
    return SuffixReachability(composition, nu, canonical_rows(setting))


def test_legacy_domain_identity_is_unchanged() -> None:
    # loewner-v2 is the default since 2026-10-20; the legacy domain is explicit.
    domain = DomainConfig(lattice_chart="ball-v1")
    assert domain.max_primitive_atoms is None
    recorded = json.loads(_V6_REPORT.read_text())["domain"]
    payload = {
        key: list(value) if isinstance(value, tuple) else value
        for key, value in domain.identity_payload().items()
    }
    assert payload == recorded
    # Value computed with the pre-change implementation.  The legacy identity
    # also names the legacy lattice head: commutant-v2 is the model default since
    # 2026-10-29 (review C3), and truncated-v1 keeps its historical payload.
    legacy = CrystalAtlasConfig(
        domain=domain, model=ModelConfig(geometry_lattice_head="truncated-v1")
    )
    assert legacy.fingerprint == (
        "04cf6e2cbdd88a2483815fb232d8bc0ef4e8bd60c357d4176271a5b2a60abd5f"
    )
    assert CrystalAtlasConfig(domain=domain).fingerprint != legacy.fingerprint
    assert CrystalAtlasConfig().fingerprint != legacy.fingerprint


def test_primitive_bound_is_validated_and_fingerprinted() -> None:
    for invalid in (0, -3, True, 2.5):
        with pytest.raises(DomainError):
            DomainConfig(max_primitive_atoms=invalid, max_raw_atoms=80)  # type: ignore[arg-type]
    with pytest.raises(DomainError, match="conventional"):
        DomainConfig(max_primitive_atoms=20, max_raw_atoms=40)
    domain = DomainConfig(max_primitive_atoms=20, max_raw_atoms=80)
    assert domain.identity_payload()["max_primitive_atoms"] == 20
    assert domain.composition_atom_bound == 20
    assert DomainConfig().composition_atom_bound == 40
    assert asdict(domain)["max_primitive_atoms"] == 20


def test_admission_rule() -> None:
    legacy = AtomBound(conventional=40, primitive=None)
    primitive = AtomBound(conventional=80, primitive=20)
    # 11 atoms per formula, four formula units in an F-centred cell:
    # 44 conventional atoms but 11 primitive atoms.
    assert not legacy.admits(4, 11, 4)
    assert primitive.admits(4, 11, 4)
    # Primitive settings are limited to 20 atoms.
    assert not primitive.admits(2, 11, 1)
    assert primitive.admits(1, 11, 1)
    # The conventional resource cap still applies.
    assert not AtomBound(conventional=40, primitive=20).admits(4, 11, 4)


def test_masks_admit_an_f_centred_cover_only_under_the_primitive_bound() -> None:
    composition = Composition.from_counts({11: 1, 17: 2, 8: 8})
    fm3m = build_space_group_setting(225)
    p1 = build_space_group_setting(1)
    settings = (p1, fm3m)
    assert fm3m.centering_index == 4
    assert _oracle(composition, 4, fm3m).is_reachable()

    legacy = AtomBound(conventional=40, primitive=None)
    primitive = AtomBound(conventional=80, primitive=20)
    legacy_nu = feasible_formula_units(
        composition, settings, max_formula_units=16, bound=legacy, oracle=_oracle
    )
    primitive_nu = feasible_formula_units(
        composition, settings, max_formula_units=16, bound=primitive, oracle=_oracle
    )
    assert 4 not in legacy_nu
    assert 4 in primitive_nu
    assert legal_settings(composition, 4, settings, bound=primitive, oracle=_oracle) == (
        fm3m,
    )
    # P1 can host only one formula unit of 11 atoms under the primitive bound.
    assert legal_settings(composition, 1, settings, bound=primitive, oracle=_oracle)[0] == p1
    assert p1 not in legal_settings(composition, 2, settings, bound=primitive, oracle=_oracle)


def test_conventional_multiplicities_fix_the_rocksalt_formula_units() -> None:
    """C3/D1: one Diophantine equation in conventional counts (theory 4.9, mutation 19).

    Every Fm-3m row has a conventional multiplicity of at least 4, so rock salt
    needs nu = 4.  Mixing in primitive counts would admit nu = 1.
    """

    rocksalt = Composition.from_counts({11: 1, 17: 1})
    fm3m = build_space_group_setting(225)
    assert _oracle(rocksalt, 4, fm3m).is_reachable()
    for nu in (1, 2, 3):
        assert not _oracle(rocksalt, nu, fm3m).is_reachable()
