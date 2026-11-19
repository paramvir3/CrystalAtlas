from __future__ import annotations

from dataclasses import replace

import numpy as np

from crystalatlas.dataset_builder import canonical_training_target_fingerprint
from crystalatlas.decoder import CrystalScaffold, OrbitAssignment
from crystalatlas.full_catalogue import build_full_catalogue
from crystalatlas.training_targets import (
    CanonicalCatalogueTrainingTarget,
    lift_canonical_catalogue_target,
)


def test_complete_catalogue_target_fingerprint_rebuilds_exactly() -> None:
    catalogue = build_full_catalogue()
    scaffold = CrystalScaffold.from_mapping(
        nu=1,
        space_group=catalogue.get(221).key,
        reduced_composition={8: 1},
        orbits=(OrbitAssignment("1a", 8, ()),),
    )
    metric = np.diag([16.0, 16.0, 16.0])
    original = lift_canonical_catalogue_target(scaffold, metric, catalogue=catalogue)
    rebuilt = lift_canonical_catalogue_target(
        original.canonical_scaffold, original.metric, catalogue=catalogue
    )
    assert isinstance(rebuilt, CanonicalCatalogueTrainingTarget)
    assert canonical_training_target_fingerprint(rebuilt) == (
        canonical_training_target_fingerprint(original)
    )
    # Fingerprinting is exact even when both arrays pass their shape/finite gate.
    latent = rebuilt.lattice_latent.copy()
    latent[0] = np.nextafter(latent[0], np.inf)
    substituted = replace(rebuilt, lattice_latent=latent)
    assert canonical_training_target_fingerprint(substituted) != (
        canonical_training_target_fingerprint(original)
    )
