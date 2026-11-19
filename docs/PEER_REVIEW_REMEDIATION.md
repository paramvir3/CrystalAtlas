# September 2026 peer-review remediation

Implementation: `0.1.0a1`. This record supplements, and does not rewrite,
the [original review](../artifacts/peer_review_2026-09-06/REVIEW.md) and its
pre-fix reproductions. A passing regression means the specified failure is
addressed, not that every real-valued theorem or every source line is verified.
Historical scripts retain their old API contracts. In particular, the old
training-measure witness constructs diagnostic batches without the new sampler
normalizer; use the production-plumbing regression in `tests/test_trainer.py`
to test the corrected training measure.

## Corrected implementation boundaries

| Review finding | Implemented correction | Regression evidence |
|---|---|---|
| R1: Equivalent proper bases yielded different identities | Finite primitive-basis minimax search, complete metric/motif ranking, explicit ambiguity and resource rejection, repinning with composed source-frame certificates | 144 signed/sheared/rotated/origin-shifted presentations; cross-Gram representative; mirror retention; unresolved-key rejection; production catalogue target-to-dedup test |
| R2: Residual actions transformed only the endpoint mean | Push forward the complete noisy path and tangent in both production CFM paths; endpoint-only symmetrization is insufficient in the interior | Analytic endpoint and interior Fourier coefficients for all six S3 actions; scalar/packed FP32/FP64 paths, unwrapped tangents, derivative checks, and RNG/autograd checks |
| R3: Internally consistent substituted targets could pass local checks | Recompute the semantic target digest after rebuilding and compare it with the digest bound to the trusted build summary | Changed metric plus matching latent and refreshed local checksums rejected; external storage-pin positive control; full-catalogue fingerprint sensitivity |
| R4: Restored/mutable histories bypassed budget accounting | Replay initial actions against the budget; protect the budget, action sequence, and accumulated costs; validate finite integral counts | Constructor overspend, source-list alias mutation, reassignment, and invalid costs rejected |
| R5: Unequal topology batches changed the declared record measure | Global epoch-population normalizer and fixed nominal accumulation denominator, including short final windows and DDP partitioning | Frozen-gradient example recovers 2.8 rather than 5.5; rank-global normalizer; partial windows; mixed-contract rejection; denominator-overflow rejection before commit |
| R6: Same-topology inference formed one unbounded ODE batch | Preflight deterministic chunks under dense-pair and estimated-workspace limits | Five attempts chunked 2/2/1; ordered outputs and per-attempt RNG preserved; failed decoding retains its slot |
| R7: Composition forecasts were not bound to the evaluation contract | Mandatory externally pinned ordered forecast digest for composition metrics | Changed IDs, scores, token counts, labels, order, subsets, and missing pins rejected |
| R8: Empty candidate pools crashed kernel validation | Accept the valid 0×0 kernel before spectral indexing; validate requested count | Empty pools with zero/nonzero request, malformed kernels, and invalid request counts |
| R9: CIF operation closure did not imply cell-metric compatibility | Test each declared operation's dimensionless Cartesian isometry residual before expansion | Contradictory fourfold/orthorhombic cell rejected; compatible cells accepted |

## Mathematical scope

For a primitive row lattice \(L\), \(G=LL^T\). The basis-search objective
starts with \(\max_i(BGB^T)_{ii}\), with \(B\in SL(3,\mathbb Z)\).
The supplied basis gives an upper bound; every improving basis vector belongs
to the corresponding finite lattice ellipsoid. Exhaustive enumeration, not
an arbitrary finite list of setting changes, underlies this search. Ranking
is numerical and quantized: floating ambiguity and resource ceilings can
reject a record. It is not an interval-arithmetic proof of all real inputs.
The existing exact rational Hall/Wyckoff catalogue is a separate boundary.

For a certified finite torus-action group \(H\), endpoint augmentation is the
mixture \(|H|^{-1}\sum_{T\in H}T_\#K_\sigma(\cdot\mid\eta)\).
Transforming only \(\eta\) and then adding isotropic noise is generally
different when \(T\) is nonorthogonal. The fix addresses this ordering defect;
it also acts on the entire sampled interpolation and tangent. Otherwise the
coordinatewise half-open interpolation can break symmetry at intermediate
times even with an invariant endpoint. An analytic three-orbit midpoint
witness gives Fourier coefficients differing by `0.0196878463` for the
endpoint-only correction. Full-path averaging restores agreement by group
closure and preserves the continuity equation. It does not expand certification
to every residual gauge in all 230 groups.

Production gradient normalization is
\(\sum_{b\in W}n_b\nabla\mathcal L_b/[RA(N/B)]\), where \(N,B\) count
global retained records and microbatches, \(R\) is world size and \(A\) the
nominal accumulation length. This fixes the per-record coefficient at fixed
parameters. It is not an assertion that Adam trajectories are invariant to
batch regrouping. Explicit diagnostic batches retain window-mean semantics.

The CIF check uses \(Q=L^{-1}WL\) and \(QQ^T\approx I\), equivalent to
\(WGW^T\approx G\) in the package's row convention. Its declared
dimensionless tolerance is \(10^{-8}\); affine group closure alone cannot
establish this metric property.

## Compatibility and trust

- Rebuild raw-to-target datasets, identity partitions, deduplication, and splits.
  Do not relabel old canonical-identity hashes as version 2.
- The objective contract is version 3, trainer state version 6, sampler policy
  and state version 3, epoch plan version 2, and benchmark method version 3.
  Old states are intentionally not exact-resume-compatible. Reusing weights
  would be a separately declared new experiment, not continuation.
- Composition diagnostics use report version 4 and evaluation-contract version
  2. The forecast digest varies by checkpoint; the cross-checkpoint evaluation
  policy remains separate.
- Build, source, storage, and forecast digests still need external trusted
  receipts. Self-hashing arbitrary replacement artifacts is not authentication.
- Protocol cost/status/history fields are protected; arbitrary action metadata
  is not deeply frozen and is not an independent signed provenance store.
- Memory chunking bounds the declared estimator, not every allocator/kernel
  overhead. CUDA numerical and memory tests must be repeated for this build.

## Validation record

The first full integration run recorded 476 passed, one stale version-assertion
failure, one macOS Gloo skip, and 45 passing subtests. After correcting that
assertion and adding the computed-normalization guard, the focused training
suite recorded 61 passed and one platform skip. A subsequent mathematical
audit extended R2 from endpoint-only to whole-path/tangent symmetrization;
the affected flow, learned-model, objective, benchmark, and overfit tests
recorded 52 passed. The initial staging wheel in `dist_verified_v10` predates
this last correction and is superseded; do not use it as the release candidate.
Final-code validation completed across four disjoint selections: **484 passed,
one macOS Gloo skip, and 45 passing subtests**. The final installed wheel passed
94 targeted tests, one platform skip, and ten subtests; all 61 package files
match the source. Ruff passes. The final wheel is
`dist_verified_v11/crystalatlas-0.1.0a1-py3-none-any.whl`, SHA-256
`5feef9ca6656d734ccf8f7305c8b7976aed589b58d5ae16f512460b446437d7e`.
The preceding monolithic final-code run ended with SIGTERM and is not counted
as passing. Logs and the explicit test partition are recorded in
[the validation artifact directory](../artifacts/remediation_2026-09-07/).

The 7 September local environment uses Python 3.12.9, PyTorch 2.13.0, NumPy 1.26.4
and spglib 2.7.0. That earlier validation did not include a new CUDA test or
real-corpus training result.
The historical A100 evidence and its wheel hashes are preserved separately.

### 10 September CUDA/Linux follow-up

The same a1 wheel now has [RTX 5060 Ti execution evidence](../artifacts/gpu_rtx5060ti_16gb_2026-09-10_a1/README.md):
512 tests plus 45 subtests passed across the recorded selections, including
25 explicit CUDA tests, two real two-process CPU/Gloo normalization tests and
the exhaustive all-230 catalogue suite. Seven micro/default production-objective
precision gates completed with finite measured gradients and losses. The exact
resume GPU tests use synthetic factor-model trainer fixtures, not a full
production-model training trajectory. Three same-input precision diagnostics
found BF16 composition-gradient relative-L2 deviations of 3.914–4.334% on the
small untrained fixture; no AMP quality advantage is inferred. FP32 remains the
baseline. This follow-up changes tests and evidence, not production equations
or architecture. Real-corpus training and predictive superiority remain untested.

### 10–13 September production-model follow-up

[The bounded production follow-up](../artifacts/gpu_rtx5060ti_2026-09-10_production_a1/README.md)
completed nine full-width P1 geometry gates at 8/16/40 atoms and batch sizes
1/2/4, with 64-step RK4 execution, exact same-seed replay, public single-cover
comparisons and FP64 decoder audits. The separate bridge gate retains one
failure: the 40-atom FP32 metric exceeds its original elementwise tolerance
despite small normwise error and passing derivative/CPU–CUDA checks. No
tolerance was relaxed and no physics equation was changed to conceal it.

A genuine nested immutable scientific-manifest reconstruction bug blocked the
production resume preflight. The corrected normalization preserves canonical
JSON fingerprints and rejects unsupported/cyclic/nonfinite content. The new
v12 wheel SHA-256 is
`fe86795da6c067644aa40268ba8afab904ca1d76a3a2a9ce4282b99db1314e08`;
only `crystalatlas/training.py` differs from v11. The corrected installed wheel
passed 111 CPU/Linux regression tests and full-width FP32 CUDA exact continuation
of the real model/objective/trainer on one synthetic structure's replicas.
An additional 25 CUDA and two real CPU/Gloo regressions pass on v12, giving
138 distinct selected installed-wheel regression passes, not 138 GPU tests.
That is not full-corpus or cross-device resume evidence. The v11 geometry
reports remain explicitly bound to v11; their implementation bytes are
unchanged in v12.

## Still required before scientific performance claims

1. An authenticated, licensed real corpus; measured rejection reasons and
   representation coverage, including boundary cases outside present guards.
2. Resolve the retained FP32 precision-contract finding; extend the bounded
   production-model resume and larger-structure generation checks to broader
   real topologies, stored-dataset and full-width AMP resume, ODE step-refinement
   accuracy, capacity-triggered GPU chunking, and real multi-GPU NCCL runs.
3. All-catalogue residual-action certification beyond the present witnesses.
4. Tiny-corpus overfit, held-out composition calibration, scaffold Recall@K,
   oracle/predicted-scaffold geometry, and matched-budget baseline experiments.
5. Frozen relaxation/energy protocols and independent stability evaluation.
   Exact crystallographic legality is not energetic stability or synthesizability.
