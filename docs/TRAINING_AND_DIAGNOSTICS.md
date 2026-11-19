# Training and diagnostic protocol

This document specifies the implemented Step 3 and Step 4 boundary. It is a
protocol for training and diagnosing the three CrystalAtlas factors; it is not
a report of a completed materials experiment.

The code has unit and property tests for its numerical contracts. This
repository does not contain a legally authorized crystal corpus, a trained
checkpoint, a completed multi-GPU scaling run, prospective DFT results, or
evidence of predictive superiority. “Implemented” below means that the
algorithmic path exists and is tested on controlled fixtures. It does not mean
that an empirical claim has been established.

`crystalatlas status` exposes this distinction in separate
`training_pipeline` and `diagnostics` objects. It reports the mechanisms as
available while keeping `trained_checkpoint_bundled` and
`empirical_results_bundled` false.

## Factorization and empirical measures

The learned law is trained in the same order in which it factorizes:

\[
p_\omega(c),\qquad
\pi_\phi(\alpha\mid c),\qquad
\rho_{\theta,\alpha}(q\mid c),
\]

where \(c\) is a reduced composition, \(\alpha\) is an exact arithmetic cover,
and \(q=(z,\eta)\) contains the invariant lattice and intrinsic Wyckoff
coordinates. `StagedTrainingSchedule` makes the policy executable: composition,
cover, and geometry phases precede any nonzero joint phase. Only the active
factor is trainable in a pretraining phase.

Two finite-population measures must not be conflated. Let the retained
structure corpus have \(N_s\) structures, let it contain \(N_c\) distinct
reduced compositions, and let \(n(c)\) be the number of retained structures
with composition \(c\).

- `TrainingMeasure.UNIQUE_COMPOSITION` selects one deterministic canonical
  representative for each \(c\). It is the composition-pretraining measure.
- `TrainingMeasure.STRUCTURE` visits every retained polymorph once. It is the
  cover, geometry, and joint-fine-tuning measure.

During joint training, the composition term attached to structure \(s\) is
weighted by

\[
w_c(s)=\frac{N_s}{N_c}\frac{1}{n(c_s)}.
\]

Consequently,

\[
\mathbb E_{s\sim U(\mathcal D_s)}
\left[w_c(s)\,\ell_{\rm comp}(c_s)\right]
=
\frac{1}{N_c}\sum_{c\in\mathcal D_c}\ell_{\rm comp}(c).
\]

Database polymorph frequency therefore cannot silently redefine the learned
composition prior. `ObjectiveDataMeasure.from_dataset(...)` freezes \(N_s\)
and \(N_c\), and `CrystalAtlasTrainingObjective` applies this correction.

`TrainingMeasure.SYSTEM_FORMULA_POLYMORPH` is the discovery-oriented S-DNG
search measure (3.15a): \(S\sim\mathrm{Unif}\), \(c\sim\mathrm{Unif}(\mathcal
C_{\rm obs}(S))\), \(C\sim\mathrm{Unif}(\mathcal P_{\rm obs}(c))\), with \(S\)
the identity element-set key, \(c\) the composition key and \(C\) the identity
polymorph key.
- **Visits:** every record once. Record \(i\) carries the structure weight
  \(N_s p_i\) and the composition weight \(N_s P(c_i)/n(c_i)\), computed
  exactly in rationals (`system_formula_polymorph_weights`).
- **Expectations:** the epoch mean of the weighted losses is exactly
  \(\sum_i p_i\ell_i\) for cover and geometry, and \(\sum_c P(c)\ell_c\) for
  composition. Both composition factors use the same measure, so the sum is one
  objective under the search law. It is a search-prior objective, not the
  database law.
- **Polymorph grouping:** records sharing a polymorph key share that
  polymorph's mass. The grouping is reversible and discards nothing.
- **Opt-in:** an objective accepts these batches only when built with
  `ObjectiveDataMeasure.from_dataset(dataset, search_measure=True)`, and it
  requires every batch weight to equal the declared weight exactly. The
  objective contract then carries a digest of the weights. Legacy contracts
  are unchanged.

For every discrete decision with target \(y\), logits \(a_j\), and the exact
legal continuation set \(\mathcal L\), the loss is

\[
\ell_{\rm disc}
=-\log\frac{\exp a_y}{\sum_{j\in\mathcal L}\exp a_j}.
\]

The target must itself lie in \(\mathcal L\). Composition and cover masks are
rebuilt from the pinned grammar and catalogue before training; illegal actions
do not enter the normalization.

### Intrinsic conditional flow matching

For one fixed cover, let \(B\) be the row-basis matrix of the intrinsic period
lattice. The base law is Gaussian in \(z\) and Haar-uniform on the torus in
\(\eta\). The dequantized endpoint is

\[
z_1=z_{\rm data}+\sigma_z\epsilon_z,
\qquad
\eta_1=\operatorname{canon}_B
(\eta_{\rm data}+\sigma_\eta\epsilon_\eta),
\]

with strictly positive `CFMDequantization` scales. A uniformly sampled certified
residual relabelling \(T\) acts on the **complete sampled path and tangent**
after the coordinate lift below is chosen:
\(\eta_t\leftarrow\operatorname{canon}_B(\eta_tT)\) and
\(u_\eta\leftarrow u_\eta T\). Thus the complete noisy endpoint, including its
noise, is transformed too.
An isotropic coordinate Gaussian need not be invariant under a nonorthogonal
torus automorphism. Transforming only its mean therefore gives a different
law. Even symmetrizing the full endpoint and then recomputing a coordinatewise
half-open interpolation is insufficient: that interpolation is not covariant
under nonorthogonal actions. Transforming the whole path averages every
intermediate marginal and its probability current over the certified finite
actions. The transformed tangent is not wrapped again. This does not certify
every residual action in the full catalogue.

The deterministic half-open coordinate lift is

\[
\Delta_B(\eta_0,\eta_1)
=\operatorname{wrap}_{[-1/2,1/2)}
\left((\eta_1-\eta_0)B^{-1}\right)B,
\]

and the conditional path is

\[
z_t=(1-t)z_0+t z_1,
\qquad
\eta_t=\operatorname{canon}_B(\eta_0+t\Delta_B).
\]

The packed geometric field regresses

\[
\mathcal L_{\rm CFM}
=
\|v_z-(z_1-z_0)\|^2_{G_z(z_t)}
+\frac{1}{N}
\|v_\eta-\Delta_B\|^2_{\widehat J^T P_N\widehat J},
\]

using the metric at the current state, not an endpoint-dependent metric. \(N\)
is the expanded atom count, \(G_z\) is the lattice-chart pullback metric, and
\(\widehat J^T P_N\widehat J\) is the translation-quotiented intrinsic
coordinate metric. `CFMDequantization` defaults to \(\sigma_z=0.05\) and
\(\sigma_\eta=0.03\); these are
declared experimental hyperparameters, not fitted physical uncertainties.

The joint scalar objective is a declared weighted sum of composition NLL,
cover NLL, and CFM regression. The three weights are stored in `TrainerConfig`
through `LossWeights`; changing them changes the training identity.

## Authenticated batching

`StoredTrainingDataset` opens an authenticated `DatasetStore` only when its
build manifest belongs to the live catalogue. Every selected record is
re-audited by rebuilding its composition, cover trace, scaffold, and continuous
target from the pinned grammar, domain, and catalogue. Stored crystallographic
arrays must be read-only float64 values and must agree with the rebuilt target.
The rebuilt target's canonical fingerprint must also equal the target digest
already bound to the trusted build summary. Recomputing local shard checksums
cannot authorize a substituted, internally self-consistent target.
See [the dataset protocol](DATASET_PIPELINE.md) for provenance, deduplication,
polymorph retention, and split construction.

`DeterministicBucketBatchSampler` groups records by both expanded atom count and
an exact cover-topology key. The key pins the Hall setting, Wyckoff row order,
selected occupation actions, orbit topology, chart dimensions, and intrinsic
period-basis fingerprint. A geometry batch therefore has one tensor topology,
although samples may contain different element identities.

Epoch order is a SHA-256 rank of the seed, epoch, scope, and stable record IDs,
not a library-specific random permutation. In distributed mode:

- batches are rank-disjoint;
- a retained record is neither duplicated nor dropped under the default
  `drop_last=False` policy;
- existing batches may be split without crossing a topology bucket so all
  ranks execute the same number of steps; and
- an impossible nonempty equal-step partition fails closed instead of padding
  the epoch with duplicate records.

Dataset, rank-independent sampler-policy, rank-specific partition, epoch-plan,
epoch, and cursor fingerprints form the sampler resume state. Every sampler
records its explicit partition rank and world size. Before an epoch or
checkpoint, all ranks prove that the shared policy fingerprint is identical
while their partition ranks match the owning processes. `advance_epoch()` is
legal only after exhaustion.

**Mixed-topology batching (opt-in, review M5).** Exact topology fibres
fragment real data: the 193-record MP-20 pilot has 132 fibres, 115 of them
singletons. `MixedTopologyBatchSampler` (schema
`crystalatlas.mixed-topology-batch-sampler.v1`) ranks every eligible record
with the same SHA-256 rule and fills batches from consecutive ranks. A batch is
closed early only when the padded dense call would exceed
`max_dense_pair_interactions`, that is, when `len(batch) · max_atoms²` is over
budget. Each yielded `MixedTrainingBatch` is a tuple of ordinary per-fibre
`TrainingBatch` parts, so every exact per-topology invariant and audit still
holds:

- the composition and cover losses are record-weighted means over the parts;
- geometry is one padded dense call (`geometry_cfm_objective_padded` and
  `IntrinsicGeometryField.forward_padded`).

Padding is exact:

- in the dense backbone, padded atoms are masked out of attention keys,
  partner means, vector sources and pooling;
- each crystal's tangent solve uses its real atoms and its own coordinates,
  with identity padding and its own rank certificate;
- padded velocities are zero, and padded metrics are the identity.

A padded batch equals independent per-crystal evaluations to rounding (tests
in `tests/test_padded_batching.py`). The legacy bucket sampler and its
fingerprint are unchanged.

The primary `invariant_global` candidate packs a topology-homogeneous batch as
an explicit \([B,N,N]\) dense attention tensor.  It has no cutoff graph,
periodic-image edge enumeration, `edge_index`, or scatter aggregation, and it
fails before allocation when \(BN^2\) exceeds the declared dense-interaction
budget.  The retained `equivariant_local` ablation instead packs exact
block-diagonal periodic-image edges and enforces its separate candidate and
edge budgets.  CFM training performs one packed field evaluation, and inference
groups identical exact layouts into batched RK4 solves, chunked by both the
declared dense-pair limit and estimated workspace limit, while preserving
per-attempt random streams, failed-attempt slots, and output ordering. The local
ablation uses one candidate per chunk. The workspace estimate is not a guarantee
against allocator or kernel overhead on every accelerator. This is vectorized PyTorch
code; no custom fused CUDA kernel or production-throughput claim is made.  The
[single-device benchmark](BENCHMARKING.md) states explicitly which backend each
command measures and keeps CPU/GPU memory evidence and multi-device scaling
claims separate.

**Per-record work is batched (2026-10-25).**  When every record in a packed
batch shares one cover, the CFM objective expands it in a single
`TorchCoverGeometry.forward_batch` call.  The `loewner-v2` pullback metric
uses its batched closed form, `lattice_pullback_metric_batch`.  The field's
translation-quotient solve (`reduce_cartesian_to_intrinsic_packed`) and its
time features are batched as well.  The solve keeps the per-crystal Float64
rank and condition-number certificate, with the unbatched error messages, and
needs one host synchronization per batch.  Encoded targets come from an exact,
read-only cache.  All of these equal the per-record path up to rounding, which
the tests check.  Batches that mix covers keep per-record expansion.  On an RTX
5060 Ti (float32), one training step on the F10 development toy takes 0.0385 s
at batch 64 and 0.080 s at batch 256, against 0.267 s and 0.87 s before.

**Running on CUDA.**  Set `CUBLAS_WORKSPACE_CONFIG=:4096:8` before CUDA
initializes; the deterministic CUDA regression tests refuse to run without it.
Consumer GPUs run FP64 at a small fraction of their FP32 rate, so learned-field
experiments use float32 master precision.  Chart algebra stays in canonical
Float64 regardless, as does the tangent solve.  On the trained F10 development
field, the float32/float64 gap in endpoint L¹ is 1.2e-6.

## Scalable trainer contract

For production topology batches, let \(N\) be the number of retained records
in the global epoch plan, \(B\) its global number of microbatches, \(R\) the
world size, and \(A\) the configured accumulation length. The sampler attaches
\(\mu=N/B\) to every batch, before partitioning by rank. For a window \(W\)
over all ranks, `ScalableTrainer` differentiates

\[
\mathcal L_W
=\frac{\sum_{b\in W} n_b\mathcal L_b}
{R A\mu}.
\]

The denominator is fixed even for the final partial window. Every retained
record then has the same coefficient across an epoch at fixed model parameters;
small topology buckets are not upweighted as separate full batches. Reported
loss metrics remain observed record means and expose the separate gradient
denominator. This is a declared learning-rate convention, not a claim that
Adam, momentum, clipping, or parameter updates commute with regrouping records.
With explicit `drop_last=True`, the measure is on the retained plan, not the
discarded records or the original full corpus; the default is `False`.

Explicitly materialized diagnostic batches without sampler metadata retain
the window-record-mean denominator \(\sum_{b\in W}n_b\). One window cannot mix
these policies or different \(\mu\) values. After DDP gradient averaging the
trainer multiplies by \(R\) divided by the chosen denominator. All ranks must
present the same number of microbatches. Nonfinite denominators or scaling
factors fail before an optimizer commit.

The implemented systems policy includes:

- one process per device with the three nonoverlapping factors wrapped
  independently by DDP; unused-parameter discovery is on by default because
  exact fixed-site and zero-edge cover topologies legitimately deactivate
  different geometric branches on different ranks;
- `FLOAT32`, `BFLOAT16`, and `FLOAT16` neural precision modes, with appropriate
  autocast/scaling behavior; every model uses one FP32 or FP64 master dtype,
  and AMP requires FP32 masters;
- optional gradient-norm clipping and finite-gradient/state checks;
- EMA updates only for parameters active in the current phase;
- optimizer-step or epoch scheduler intervals, with one shared scheduler
  reserved for the joint phase; factor-pretraining learning rates remain fixed;
- atomic, scientific-manifest-bound checkpoints; and
- exact rank-local resume for RNG streams, sampler cursor, AMP scaler,
  scheduler, EMA, optimizer, model, and optional `TrainingContext`.

Mixed neural precision never weakens crystallographic checks.
`ExactBatchCrystallographicAudit` runs outside autocast in float64, decodes every
scaffold, requires primitive hidden-translation index one, round-trips the
lattice latent in a dimensionless reference-length normalization, and compares
intrinsic coordinates modulo the exact period lattice. A failed audit aborts
the step.

An FP16 loss-scale overflow does not discard a material. The trainer reduces
the scale and retries the same already-materialized accumulation window after
restoring its rank-local pre-window RNG state, reproducing the CFM coupling.
The retry budget is frozen in `TrainerConfig`; persistent overflow faults the
trainer instead of advancing either trainer or sampler cursors.

Exact resume additionally requires the same trainer configuration, staged
schedule, model-state schema, optimizer parameter names and order, scheduler
class and initial semantic configuration, catalogue/scientific identity, DDP
world size, backend, and device type. A changed world size is a new stochastic
run, not exact continuation. A trained checkpoint requires an explicit,
canonical objective contract declaring `checkpointable=True`; a name- or
bytecode-derived callback identity is
deliberately non-checkpointable because it cannot bind arbitrary mutable global
state. The production objective contract pins the authenticated dataset
fingerprint, \(N_s\), \(N_c\), and the exact hexadecimal CFM dequantization
scales, complete-noisy-path-and-tangent action policy, and sampler normalization policy.
The epoch fingerprint also binds \(\mu\). These policies changed in alpha
`0.1.0a1`; earlier trainer and sampler states cannot be resumed exactly.
Every unfinished checkpoint—including one exactly at an epoch or phase
boundary—requires the aligned sampler fingerprint and cursor; otherwise a
changed seed, batch size, rank partition, or measure could silently change the
future trajectory. Checkpoints are forbidden while an accumulation graph is
pending.

`transactional_optimizer_step=False` is the scalable default because cloning a
large model, optimizer, and EMA on every update is prohibitive. With this
setting, a failed optimizer transaction faults the trainer and requires restore
from the last atomic checkpoint. The opt-in transactional mode is intended for
small diagnostic runs.

### Minimal phase loop

The following is the core workflow after an authenticated `store`, a certified
`catalogue`, a configured `model`, and an initialized DDP process group (when
used) already exist:

```python
import torch

from crystalatlas.trainer import (
    DistributedConfig,
    PrecisionMode,
    ScalableTrainer,
    StagedTrainingSchedule,
    TrainerConfig,
    TrainingPhase,
)
from crystalatlas.training_data import (
    DeterministicBucketBatchSampler,
    StoredTrainingDataset,
    TrainingMeasure,
)
from crystalatlas.training_objectives import (
    CrystalAtlasTrainingObjective,
    ExactBatchCrystallographicAudit,
    ObjectiveDataMeasure,
)

dataset = StoredTrainingDataset(
    store,
    model.composition_codec,
    catalogue,
    domain=model.scientific_config.domain,
    record_ids=train_record_ids,
)

optimizer = torch.optim.AdamW(
    [
        {"params": model.composition_model.parameters(), "lr": 2.0e-4},
        {"params": model.cover_selector.parameters(), "lr": 2.0e-4},
        {"params": model.geometry_field.parameters(), "lr": 1.0e-4},
    ],
    weight_decay=1.0e-2,
)
schedule = StagedTrainingSchedule(
    composition_epochs=20,
    cover_epochs=20,
    geometry_epochs=40,
    joint_epochs=10,
)
trainer = ScalableTrainer(
    model,
    optimizer,
    schedule,
    config=TrainerConfig(
        accumulation_steps=4,
        precision=PrecisionMode.BFLOAT16,
        max_gradient_norm=1.0,
        distributed=DistributedConfig(enabled=use_ddp, local_rank=local_rank),
    ),
)
objective = CrystalAtlasTrainingObjective(
    model,
    ObjectiveDataMeasure.from_dataset(dataset),
)
audit = ExactBatchCrystallographicAudit(model)

phase_seeds = {
    TrainingPhase.COMPOSITION: 101,
    TrainingPhase.COVER: 102,
    TrainingPhase.GEOMETRY: 103,
    TrainingPhase.JOINT: 104,
}
active_phase = None
sampler = None
while trainer.cursor.epoch < schedule.total_epochs:
    phase = trainer.current_phase
    if phase is not active_phase:
        measure = (
            TrainingMeasure.UNIQUE_COMPOSITION
            if phase is TrainingPhase.COMPOSITION
            else TrainingMeasure.STRUCTURE
        )
        sampler = DeterministicBucketBatchSampler(
            dataset,
            batch_size=32,
            seed=phase_seeds[phase],
            measure=measure,
            rank=trainer.rank,
            world_size=trainer.world_size,
        )
        active_phase = phase
    assert sampler is not None
    report = trainer.train_epoch(
        sampler,
        objective,
        crystallographic_audit=audit,
    )
```

A sampler advances automatically only when the next epoch remains in the same
phase. At a phase boundary, construct the new phase sampler explicitly; its
phase-local epoch starts at zero. For a mid-epoch save or load, pass that exact
sampler to `trainer.save_checkpoint(...)` or
`trainer.load_checkpoint(...)`. The same requirement holds for an unfinished
epoch boundary. Every DDP rank must participate in both calls.
The checkpoint publication is performed atomically by rank zero after
collective validation.

## Diagnostic experiments

The diagnostics are model-agnostic and return canonical JSON manifests. They
require stable query and attempt IDs, a frozen generation budget, and a
caller-supplied equivalence matcher plus an `EquivalenceMatcherContract`. The
contract freezes the algorithm/version, implementation SHA-256, primitive or
conventional cell policy, and strict-JSON parameters including every numerical
tolerance. `DiagnosticEvaluationContract` separately freezes the dataset,
split, checkpoint, target corpus, generated-attempt corpus, generation
protocol, seed, attempt budget, score policy, and validity policy. Both
contracts, the exact evaluated cases, and the precomputed match-decision matrix
are fingerprinted in retrieval manifests. The callable must match its declared
implementation digest and must be deterministic, symmetric, and Boolean. The
diagnostics do not provide a hidden structure matcher or filter.

Every matcher payload is content-addressed. Immutable strict-JSON scalars and
byte strings are hashed internally and rechecked before and after matching;
structured crystals must therefore be serialized to canonical immutable bytes
before evaluation. Mutable Python containers and opaque objects are rejected.
The evaluator verifies the resulting target and attempt corpus receipts against
the evaluation contract. Matcher execution fingerprints recursively include
immutable global constants and pure Python helpers and are checked again after
matching. Module globals, classes with runtime state, and function-local imports
are rejected; a wrapper around an external crystallography library requires a
separately verified wheel/container artifact and must reference the exact
certified function rather than claiming that wrapper bytecode certifies the
dependency.

For CrystalAtlas strict canonical partitions, do not call the generic
evaluators directly. Use `evaluate_canonical_scaffold_retrieval(...)` and
`evaluate_canonical_geometry_decomposition(...)`; they validate every inner
payload against the live certified catalogue and identity policy, and bind the
evaluation dataset manifest to the externally trusted canonical provenance
before matching. Invalid canonicalization attempts use `payload=None` while
retaining their ranked slot. This exact-byte partition is not a tolerance-based
physical-similarity criterion. Its metric-uncertainty and
representation-invariance claims remain explicitly false; see
[`CANONICAL_PARTITION_MATCHING.md`](CANONICAL_PARTITION_MATCHING.md).

### 1. Tiny-dataset overfit gate

`TinyOverfitGateProtocol` freezes the dataset-manifest SHA-256, invariant
evaluation-policy fingerprint, distinct scaffold- and full-structure-matcher
contract fingerprints, exact example IDs,
update/evaluation schedule, random seed, \(K\) values, and inclusive pass
thresholds—including separate total-loss and complete-composition-NLL relative
reductions—and the required number of consecutive passing scheduled
evaluations before optimization. Checkpoint and generated-attempt receipts are
allowed to change by step, while the policy cannot. The evaluation callback
returns a `TinyOverfitEvaluation` containing actual composition, scaffold, and
geometry diagnostic objects. The gate verifies their policy, checkpoint,
respective matcher, query set, and requested \(K\), then derives every scalar snapshot
internally and fingerprints all three source manifests. A \(K\) larger than the
frozen attempt budget is rejected before evaluation.

The two matcher contracts must not be collapsed in a real experiment:
scaffold retrieval compares the declared discrete scaffold representation,
whereas geometry Match@\(K\) compares decoded periodic structures under a
separately declared cell/tolerance policy.
`LearnedCrystalGenerator.sample_geometry_given_cover(...)`
provides the checkpoint-bound oracle-scaffold arm as one no-retry attempt.

`run_tiny_dataset_overfit_gate(...)` evaluates step zero, executes exactly one
callback per update, retains the complete scheduled history, and requires a
passing state at or after `minimum_steps`.
With `stop_on_pass=False`, an early crossing that later regresses does not pass.

The gate jointly checks:

- training loss and complete-composition NLL;
- scaffold Recall@\(K\);
- geometry Match@\(K\) with oracle and predicted scaffolds;
- invalid-attempt rate; and
- relative total-loss and complete-composition-NLL reduction from step zero.

Thresholds must be chosen before seeing the run. Passing requires the complete
threshold set at a predeclared consecutive streak of scheduled evaluations;
one favorable crossing cannot pass. Passing demonstrates that the
code can memorize the declared tiny corpus under the frozen protocol; it does
not demonstrate generalization.

### 2. Composition probability and calibration

`CompositionForecast.target_log_probability` is
\(\log p(c_{\text{true}})\) for the complete reduced composition. It is not the
probability of one favorable grammar token. `evaluate_composition_forecasts`
reports

\[
\operatorname{NLL}_{\rm composition}
=-\frac{1}{Q}\sum_{i=1}^Q\log p(c_i),
\qquad
\operatorname{NLL}_{\rm token}
=-\frac{\sum_i\log p(c_i)}{\sum_i T_i}.
\]

When every query supplies top-label confidence/correctness, it also reports
accuracy, mean confidence, and equal-width ECE. When every query supplies the
complete categorical law, it additionally reports the multiclass Brier score

\[
\frac{1}{Q}\sum_i\sum_j(p_{ij}-y_{ij})^2.
\]

Calibration or complete distributions must cover all queries or none; favorable
subsets are rejected. Direct probability vectors resolve ties by the frozen
supplied class order; `CompositionForecast.from_distribution(...)` first sorts
labels in Unicode-ascending order and therefore has the corresponding tie rule.

Composition evaluation requires an externally pinned
`DiagnosticEvaluationContract.composition_forecasts_sha256`. Use
`composition_forecasts_sha256(...)` when constructing the trusted evaluation
artifact, then supply that independently resolved digest to the evaluator.
The digest includes ordered query IDs, scores, token counts, and calibration
labels. Missing pins or changed/reordered/subset forecasts are rejected; the
evaluator never creates its own authorization pin. Digest agreement does not
itself prove dataset membership: the experiment harness must authenticate that
the forecast artifact was made for the frozen target corpus and checkpoint.

### 3. Legal-scaffold retrieval

Each `RankedRetrievalCase` contains every acceptable target polymorph and every
generated attempt, including invalid attempts. Attempts are sorted by
descending score and then Unicode-ascending `attempt_id`. For query \(i\),

\[
\operatorname{Recall@K}
=\frac{1}{Q}\sum_i
\mathbf 1\!\left[\exists\text{ valid match in the first }K\text{ slots}\right].
\]

`evaluate_scaffold_retrieval` also reports unique acceptable-polymorph coverage
and valid-attempt rate. An invalid attempt occupies its ranked slot. A missing
attempt still occupies the denominator because the budget is \(QK\). Filtering
or regenerating failures before evaluation is forbidden. A case containing
more attempts than the frozen `attempt_budget`, or a requested \(K\) above that
budget, is rejected; this prevents an unreported best-of-many advantage.

The matcher must encode the frozen scientific equivalence criterion. Matching
only a raw selected cover label is generally insufficient because covers
overlap and the decoded maximal symmetry can differ from the selected
containment group.

### 4. Oracle-versus-predicted-scaffold geometry

`GeometryDecompositionCase` pairs two fixed-budget arms for the same query and
acceptable-polymorph set:

- geometry generated with the true/oracle scaffold supplied; and
- geometry generated after the model's predicted scaffold.

`evaluate_geometry_decomposition` reports both Match@\(K\) values and

\[
\Delta_K=
\operatorname{Match@K}_{\rm oracle}
-\operatorname{Match@K}_{\rm predicted}.
\]

Bootstrap resampling is paired by query, so the confidence interval for
\(\Delta_K\) preserves within-query correlation. A low oracle result indicates
that geometry remains a bottleneck even with scaffold uncertainty removed. A
large positive \(\Delta_K\) indicates additional degradation introduced by
scaffold selection. These are diagnostic attributions, not causal proofs.

### Uncertainty and persistence

All reported confidence intervals use a seeded percentile bootstrap over
queries, the experimental unit. The default is 10,000 samples at 95%
confidence. The implementation and manifest explicitly freeze NumPy `PCG64`
rather than relying on a library-default generator. Query IDs, complete cases,
match decisions, population/generation contracts, tie policy, and metric
definitions are persisted, and JSON serialization rejects NaN and infinity.

## Required experiment order

Do not start a large joint run merely because the code executes. The intended
order is:

1. Freeze authorized source snapshots, preprocessing policy, catalogue receipt,
   target-store manifest, split mode, seeds, budgets, matcher, and diagnostic
   thresholds.
2. Run the persistent tiny-data overfit gate. Investigate any failure before
   scaling.
3. Train \(p_\omega(c)\) on the unique-composition measure and report complete
   composition NLL and calibration on the frozen validation split.
4. Train \(\pi_\phi(\alpha\mid c)\) on the structure measure and report
   legal-scaffold Recall@\(K\), polymorph coverage, and invalid-attempt rate.
5. Train \(\rho_{\theta,\alpha}(q\mid c)\) on the structure measure and report
   geometry Match@\(K\) with the oracle scaffold supplied.
6. Compare oracle-scaffold and predicted-scaffold geometry under identical
   paired budgets. This locates the dominant chemistry, scaffold, or geometry
   bottleneck before joint fine-tuning.
7. Jointly fine-tune only after the three factor diagnostics pass their
   predeclared gates. Re-run every diagnostic on untouched, manifest-bound
   composition-, prototype-, scaffold-, and element-set-disjoint views.
8. Treat relaxed energy/hull stability, phonons, synthesizability, novelty, and
   comparison with external generators as separate prospective evaluations
   under an identical fixed budget.

Until those runs exist, the scientifically correct status is: the training and
diagnostic machinery is implemented; predictive performance is unknown.

## What the cover selector actually learns (review m2)

The theory defines the selector weights as latent cover probabilities over an overlapping family of covers (`theory/00_SCOPE.md`, Theorem 2.5). Training sees only part of that family:

- Every training target is the record's canonical cover, whose group is the maximal group found by standardization.
- The cover likelihood is therefore fitted to the frequency of maximal canonical covers under the training measure.
- Non-maximal covers that contain the same crystal (proper subgroups whose Wyckoff tables also fit) never appear as targets. Their probability comes only from generalization and from the exact legality masks; no data constrains it.

So the overlapping-cover machinery guarantees normalization and exact support. It does not make the learned weights estimates of subgroup-cover frequencies, and results should not be described as if it did.
