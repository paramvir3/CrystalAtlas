# CrystalAtlas

CrystalAtlas is a proof-driven research implementation for crystal structure
prediction (CSP), exact-species de novo generation (S-DNG), and unconditional
de novo generation (DNG).

This repository is now an **implementation-stage research alpha**. It contains
an exact arithmetic/crystallographic core, a dense periodic Transformer with
\(SO(3)\)-invariant learned scalar channels and an \(SO(3)\)-covariant
Cartesian readout, conditional flow matching, training
numerics, checkpoints, and an executable P1/P-1 reference generator. It also
contains a separately gated production catalogue covering all 230 space-group
types, strict raw-structure/CIF standardization, and full primitive reduction.
It does **not** contain a trained checkpoint, an energy model, or evidence that
it outperforms MatterGen or any other model.

Alpha `0.1.0a1` corrects the September peer-review findings in canonicalization,
flow endpoint symmetrization, training-data authentication, batch weighting,
bounded inference, and evaluation accounting. See the
[remediation and migration record](docs/PEER_REVIEW_REMEDIATION.md) and
[research decisions](docs/RESEARCH_UPDATE_2026-09-07.md). Earlier GPU results
remain evidence for their original `0.1.0a0` wheels, not this revised package.

The governing distinction remains:

\[
\text{mathematical correctness}
\ne
\text{implementation correctness}
\ne
\text{predictive superiority}.
\]

## Current capability status

| Capability | Status |
|---|---|
| Exact reduced-composition grammar | Implemented and exhaustively tested on finite fixtures |
| Exact species-set conditioning | Implemented by support-first grammar clamping |
| Exact formula-compatible Wyckoff completion | Implemented with a suffix reachability DP |
| Exact Hall/Wyckoff algebra | Implemented for one pinned Hall setting for all 230 types (1,731 rows) |
| Invariant SPD lattice chart | Implemented with an atom-count-scaled physical support |
| Dense scalar-invariant periodic Transformer | Implemented as the provisional default candidate: all-pairs Fourier geometry, \(SO(3)\)-invariant scalar hidden channels, reflection-odd scalar features, an \(SO(3)\)-covariant readout with no imposed \(O(3)\) parity, and no sparse neighbour graph; the current mirror test is a witness, not an all-enantiomer injectivity result, and large-run adoption still requires the registered predictive-quality and accelerator gates |
| Local \(SE(3)\)-equivariant graph Transformer | Retained as an ablation candidate for the preregistered parameter- and compute-matched study; not the default |
| Intrinsic conditional flow matching | Implemented for fixed arithmetic covers |
| Full primitive reduction and origin handling | Implemented: exact rational oracle plus pinned spglib 2.7.0 numerical boundary |
| Raw structure/CIF → canonical target | Implemented for a strict, fully occupied, ordered CIF subset; tolerance-unstable or otherwise ambiguous preprocessing fails closed |
| Canonical supervised-target lift | Implemented for the complete catalogue with exact continuous-origin quotient coordinates |
| Dataset provenance and preprocessing | Implemented as a fail-closed, reviewer-authorized source/snapshot compiler with per-record scope evidence, policy-defined deduplication, four independently generated group-disjoint split views, and authenticated tensor shards; no database dump is bundled and the licence templates are not legal advice |
| Reproducible scalable training/checkpoints | Implemented: topology/atom-count batching, factor-first phases, exact unequal-microbatch accumulation, factor-wise DDP, AMP, float64 crystallographic audits, EMA, and rank-local exact resume are bound to the scientific manifest, explicit objective/population contract, and sampler policy |
| Diagnostic experiments | Implemented as fixed-budget, query-bootstrap protocols for composition NLL/calibration, legal-scaffold Recall@\(K\), oracle-versus-predicted-scaffold geometry, and a persistent tiny-data overfit gate; no real-corpus result is bundled |
| Full-package strict static typing | Not complete; CI checks the certified typed boundary and runtime-tests the rest |
| End-to-end reference DNG/S-DNG/CSP | Implemented, but deliberately untrained and non-predictive |
| Complete 230-space-group production catalogue | Implemented as a frozen, checksummed exact compiler; training requires its proof receipt |
| Packed geometry training/inference | Implemented: dense all-pairs field calls are packed by cover topology, and RK4 batches attempts with the same exact layout; no custom fused CUDA kernel or production-throughput claim is included |
| Single-device geometry profiling | `benchmark-geometry` remains the compatible local-edge profiler; `benchmark-backends` now performs a strict constant-density, synchronized forward/backward/AdamW comparison with separate interaction and memory semantics; accelerator measurements and predictive comparisons remain gates |
| Single-device full-objective profiling | Implemented through the staged trainer and float64 crystallographic audit on an exact synthetic fixture; authenticated-corpus, CUDA, and DDP measurements remain separate gates |
| Learned production checkpoint | Not included |
| Stability/synthesizability prediction | Not included |
| Superiority to published generators | Not established |

Run `crystalatlas status` for the machine-readable package capability contract.
Its `training_pipeline`, `diagnostics`, and `profiling` objects distinguish
implemented machinery from the separately reported absence of a bundled
trained checkpoint, real-corpus accelerator measurements, and empirical
results. Dataset-specific trust and identity details are documented below.

## Probability model

For a supplied reduced composition \(c\), the model is an overlapping
containment-cover pushforward:

\[
P_\Theta(A\mid c)
=
\sum_{\alpha\in\mathcal A_{\rm arith}(c)}
\pi_\phi(\alpha\mid c)
\int
\mathbf 1_A\!\left(\bar D_\alpha(q)\right)
\rho_{\theta,\alpha}(dq\mid c).
\]

Here \(\alpha=(\nu,g,h)\) contains the formula-unit multiplier, selected
space-group containment constraint, and element-decorated Wyckoff row counts.
The decoder \(\bar D_\alpha\) is deterministic. Raw cover weights are latent
proposal probabilities, not probabilities of the decoded maximal space group.

Unconditional generation adds a normalized composition law:

\[
P_\Theta^{\rm DNG}(A)
=
\sum_c p_\omega(c)P_\Theta(A\mid c).
\]

The unique support-first serialization

\[
[K,Z_1,\ldots,Z_K,n_1,\ldots,n_K,\mathrm{EOS}]
\]

gives the exact factorization \(p_\omega(c)=p_\omega(S)p_\omega(n\mid S)\).
Consequently:

- DNG samples the element set, stoichiometry, cover, and geometry;
- S-DNG clamps the exact all-and-only element set and samples stoichiometry,
  cover, and geometry; and
- CSP supplies the reduced formula and samples only cover and geometry.

All three modes share the same downstream conditional model and decoder.

## Architecture

1. A causal support-first Transformer scores composition grammar tokens.
2. A causal cover Transformer scores \(\nu\), the group, and Wyckoff row
   occupations.
3. Exact live-prefix and suffix-DP masks remove actions without a complete legal
   continuation; no learned model may override these masks.
4. A bounded invariant log-SPD chart represents the lattice. Its reference
   volume scales with the advertised atom count, and its support has a proved
   positive lower bound on every lattice singular value.
5. Exact Wyckoff operations expand intrinsic orbit parameters to the full
   advertised periodic motif.
6. A noncausal dense periodic Transformer consumes scalar atom features and
   exact fractional-torus Fourier pair features over all atom pairs. Learned
   hidden channels are \(SO(3)\)-invariant scalars; deterministic direct,
   reciprocal, direct--reciprocal cross-product, and pinned lattice-frame
   carriers give an \(SO(3)\)-covariant Cartesian readout. Reflection-odd scalar
   features and axial carriers permit mirror sensitivity, but the output has no
   imposed \(O(3)\) parity and no claim of injective enantiomer recognition. It
   has no neighbour list, cutoff
   graph, sparse edge index, or scatter aggregation. Deterministic geometry is
   evaluated in FP32/FP64 even when learned layers use mixed precision.
7. An exact translation-quotient adjoint maps Cartesian vectors back to
   intrinsic Wyckoff velocities without a ridge.
8. Conditional flow matching trains the continuous velocity field; RK4
   integrates its ODE at inference.
9. Deterministic decoding, exact-rational reconstruction of the complete
   floating translation subgroup, and full primitive reduction preserve both
   the sampled cover motif and its physical primitive quotient. Failures remain
   first-class outcomes; there is no hidden retry, repair, relaxation, or
   filtering step.

Repeated identical orbit instances use an exchangeable base and a
permutation-equivariant expanded-atom field. Training uniformly augments the
certified residual orbit relabellings, including P1 anchor exchange. Numerical
field covariance is presently checked on P1 exchange and P-1 representative-
inversion witnesses; an all-catalogue residual-action sweep remains a hard gate
before a large run. This avoids factorial enumeration while retaining the
finite-gauge symmetry of the objective.

The Transformer is the learned geometric field. Flow matching is the stochastic
generative mechanism that transports base noise to a multimodal continuous
geometry law; a deterministic Transformer alone would produce one point per
condition rather than a normalized polymorph distribution.

Learned construction never silently substitutes the P1/P-1 fixture catalogue.
Use `CrystalAtlasModel.fixture(...)` only for equation-level tests and
`CrystalAtlasModel.production(...)` for the frozen all-230 catalogue, or pass an
explicit certified catalogue to the constructor.

## Installation

Core exact functionality requires Python 3.11+ and NumPy:

```bash
python -m pip install -e .
```

The learned stack additionally requires PyTorch:

```bash
python -m pip install -e '.[ml]'
```

All-230 standardization and raw CIF/structure conversion require the pinned
crystallography backend:

```bash
python -m pip install -e '.[crystallography]'
crystalatlas verify-crystallography --receipt crystallography-receipt.json
```

The verification command checks the frozen 230/1,731/4,425 manifest, exact
finite-group closure, multiplicities, orbit symmetry, invariant metrics,
continuous-origin charts, exact decoder round trips, the independent spglib C
operation database, and every pinned finite-origin gauge. It is intentionally
an offline pre-training gate rather than a cheap per-sample check.
The saved JSON can be reloaded with
`CrystallographyCertification.read_json(...)`; training still calls
`receipt.require_for(live_catalogue)` so a stale or mismatched receipt fails
closed.

For development checks:

```bash
python -m pip install -e '.[dev,ml]'
pytest -q
```

## Command-line use

Inspect the honest capability contract:

```bash
crystalatlas status
```

Parse and reduce a flat formula:

```bash
crystalatlas formula Si2O4
```

Exercise the untrained integration-reference law:

```bash
# Unconditional DNG: no formula and no element set
crystalatlas reference-generate \
  --mode dng --samples 10 --seed 7 --output generated/dng

# Exact-species DNG: all generated formulas contain Li, Fe, P, and O only
crystalatlas reference-generate \
  --mode species-dng --species Li,Fe,P,O \
  --samples 10 --seed 8 --output generated/li-fe-p-o

# Reduced-formula CSP
crystalatlas reference-generate \
  --mode csp --formula LiFePO4 \
  --samples 10 --seed 9 --output generated/lifepo4
```

Every reference output and manifest states:

```text
law = untrained-uniform-reference
physical_prediction = false
```

These samples validate factorization, constraints, decoding, serialization, and
failure accounting. CIF files contain the full motif, are therefore declared as
P1 to prevent accidental double symmetry expansion, and retain the selected
containment group in audit comments. Manifests pin the root seed, configuration,
reference law, exact catalogue fingerprint, runtime versions, and output hashes.
These samples are not candidates ranked by learned chemistry.

## Implemented modules

- `composition.py`, `tokenization.py`: exact canonical formula grammar and
  neural token bridge.
- `constraints.py`, `cover.py`: exact arithmetic Wyckoff suffix DP, unranking,
  and one shared canonical row order.
- `algebra.py`, `catalogue.py`, `wyckoff.py`, `full_catalogue.py`: rational
  affine algebra, the lightweight P1/P-1 fixtures, and the frozen exact
  all-230 compiler.
- `standardization.py`, `ingestion.py`, `certification.py`: complete hidden-
  translation reduction, pinned Hall/origin standardization, strict raw
  CIF/structure conversion, and the mandatory pre-training proof receipt.
- `lattice.py`, `geometry.py`, `decoder.py`: invariant SPD charts, continuous
  cover layout, exact expansion, and audits.
- `nn/`: composition/selector Transformers, the dense \(SO(3)\)-invariant-scalar
  periodic candidate with an \(SO(3)\)-covariant readout, the sparse local
  \(SE(3)\)-equivariant backend retained as an ablation, the learned intrinsic field,
  and exact neural samplers.
- `flow.py`: product base, torus-aware CFM paths/loss, and RK4.
- `training.py`: legal-action NLLs, guarded optimization, scientific manifests,
  exact model-state and catalogue-bound atomic checkpoints, and RNG restoration.
- `training_data.py`, `training_objectives.py`, `trainer.py`: authenticated-target
  re-auditing, deterministic topology/atom-count buckets, exact empirical
  measures, packed CFM objectives, factor-first DDP/AMP optimization, float64
  crystallographic audits, EMA, and rank-local exact resume. See
  [`docs/TRAINING_AND_DIAGNOSTICS.md`](docs/TRAINING_AND_DIAGNOSTICS.md) and the
  [`production pilot gate`](docs/PRODUCTION_PILOT.md).
- `diagnostics.py`, `canonical_matching.py`: persistent tiny-data overfit gates,
  strict provenance-bound canonical-partition equality, composition proper
  scores/calibration, legal-scaffold Recall@\(K\), polymorph coverage, and paired
  oracle-versus-predicted-scaffold geometry diagnostics, with the full
  equivalence-matcher algorithm/cell-policy/parameter contract, evaluated population,
  checkpoint, generation budget/protocol, cases, and match decisions
  fingerprinted in every retrieval manifest. See
  [`docs/TRAINING_AND_DIAGNOSTICS.md`](docs/TRAINING_AND_DIAGNOSTICS.md) and the
  [canonical-partition boundary](docs/CANONICAL_PARTITION_MATCHING.md). The
  strict partition matcher is not a tolerance-based physical-similarity score.
- `training_targets.py`: deterministic complete-catalogue targets, including
  exact selector masks/teacher-forcing traces and origin-quotiented continuous
  \(z,\eta\) coordinates.
- `dataset_provenance.py`, `dataset_identity.py`, `dataset_splits.py`,
  `dataset_builder.py`, `dataset_storage.py`, `dataset_orchestration.py`:
  archived licence/snapshot receipts
  and per-record scope evidence, policy-defined canonical identity partitions,
  conservative full-metric duplicate removal that retains same-composition
  structural variants, four exact group-disjoint split modes, the fail-closed
  dataset compiler, and deterministic authenticated NumPy shards with a lazy
  read-only reader. See
  [`docs/DATASET_PIPELINE.md`](docs/DATASET_PIPELINE.md).
- `benchmarking.py`, `backend_comparison.py`, `training_benchmarking.py`:
  synchronized, content-addressed single-device local, same-workload backend, and
  complete-objective capacity measurements with explicit exclusions and honest
  interaction/memory semantics.
  See [`docs/BENCHMARKING.md`](docs/BENCHMARKING.md).
- `reference.py`, `pipeline.py`: failure-preserving end-to-end reference
  generation for DNG, S-DNG, and CSP.
- `evaluation.py`, `controller.py`: auditable candidate evaluation and fixed-
  budget external allocation interfaces.

## Exact guarantees and scientific limits

Within the declared finite support and certified catalogue, the code
can guarantee exact reduced-formula arithmetic, capacity-safe Wyckoff row
occupations, SPD invariant metrics, periodic Wyckoff expansion, selected-group
containment, and normalized ancestral discrete decisions.

It cannot infer thermodynamic stability from those guarantees. A
decoder-legal crystal need not satisfy stationarity \(\nabla E=0\), phonon
stability, free-energy hull stability, kinetic persistence, or synthesizability.
Those require independent
energetic/thermodynamic calculations and, for synthesis, process and barrier
information. The primary future comparison must count distinct novel structures
that remain near the DFT hull after a prospectively frozen, fixed-budget,
identical validation pipeline.

The current domain excludes partial occupancy and disorder, magnetic order,
incommensurate phases, quasicrystals, finite-temperature displacement
distributions, and defects not represented by a bounded ordered periodic cell.
The present learned stack is an executable research model, not yet a production
materials-discovery system. Scalable training, packed geometry evaluation, and
the diagnostic protocols are implemented, but this repository contains no
authorized training corpus, completed multi-GPU run, trained checkpoint,
authenticated real-data GPU/DDP throughput study, or prospective DFT
validation. Those empirical steps are required before a comparison with
MatterGen or another generator is scientifically meaningful. The code defines
how reviewer-authorized,
scope-evidenced raw snapshots become audited targets, but it does not bundle or
silently download those snapshots.

## Theory and provenance

The formal specification, proofs, obligations, and novelty boundary remain in
[`theory/`](theory/). The implementation is an independent clean-room
translation of the package equations; external generators are benchmark and
prior-art references, not imported core modules. This is not a declaration
that every ingredient is unprecedented. Exact constrained decoding, invariant
lattice coordinates, scalar/covariant geometric attention, and flow matching
have related literature. Any eventual novelty claim must be narrow, cited, and
supported by matched empirical results.

The exact crystallographic tables are redistributed from the pinned spglib
2.7.0 source corpus under its bundled BSD-3-Clause licence and provenance file.
Those authoritative tables are not claimed as our invention; the exact
compiler, certification, canonicalization, and model integration are package
code.

## Licence

CrystalAtlas is source-available under the
[PolyForm Noncommercial License 1.0.0](LICENSE).

- **Free** for any noncommercial purpose: academic research and teaching, personal study,
  and use by educational institutions, public research organizations, charities and
  government institutions, whatever their funding.
- **Commercial use needs a separate paid licence.** That includes use by or for a company,
  in a commercial product or service, or in commercial research. To ask for one, write to
  paramvir.chem@gmail.com.

Bundled spglib 2.7.0 tables in `src/crystalatlas/_data/spglib_v2_7_0/` keep their own
BSD-3-Clause licence (`COPYING` in that directory). No dataset is bundled; the MP-20 data
the research pipeline reads is not covered by this licence and keeps its own terms.
