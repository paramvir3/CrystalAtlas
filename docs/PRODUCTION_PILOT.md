# Production pilot gate

This document is the executable handoff from the implemented package to the
first real-data and accelerator experiments. It is deliberately a gate, not a
promise of predictive quality. A large run is forbidden until the data,
crystallography, tiny-overfit, and hardware checks below all pass under frozen
fingerprints.

## 1. Freeze an authorized source snapshot

The repository does not download or redistribute a training database. For each
source, archive and externally trust-pin:

- the licence or terms artifact reviewed for the intended use;
- the immutable data artifacts;
- the selection specification and ordered record index;
- the ordered payload-digest index;
- the per-record scope-evidence index; and
- the reviewer authorization receipt.

The source manifest fingerprint must be supplied from outside the acquisition
tree. A self-hash inside the same tree is integrity metadata, not an
authorization signature. Build the dataset only through the authenticated
entry point described in [`DATASET_PIPELINE.md`](DATASET_PIPELINE.md):

```python
from crystalatlas import compile_local_cif_dataset
from crystalatlas.dataset_identity import DeduplicationLevel

if compiler.deduplication_level is not DeduplicationLevel.STRUCTURE:
    raise RuntimeError("production compilation requires STRUCTURE deduplication")

receipt = compile_local_cif_dataset(
    compiler,
    expected_source_manifest_fingerprint=trusted_source_manifest_fingerprint,
    expected_crystallography_receipt_fingerprint=(
        trusted_crystallography_receipt_fingerprint
    ),
    artifact_roots=artifact_roots,
    destination="compiled-dataset",
)
print(receipt.to_json(), end="")
```

Use `STRUCTURE` deduplication for the irreversible training corpus. Keep the
scale-free `POLYMORPH` identity only as a grouping/evaluation key unless
thermodynamic condition and calculation-method metadata justify collapsing
volume states. Freeze all four split views—composition, prototype, scaffold,
and element set—before any model selection.

Every accepted row must pass raw-payload binding, full primitive reduction,
hidden-translation removal, pinned-setting/origin standardization, exact
catalogue assignment, decoder round trip, and the symmetry-tolerance stability
grid. Ambiguous preprocessing is rejected and reported. A difficult but valid
model output is never removed from an evaluation denominator.

## 2. Reproduce the crystallographic proof receipt

Install the pinned crystallography extra and run:

```bash
crystalatlas verify-crystallography \
  --receipt crystallography-receipt.json
```

The currently verified catalogue has 230 pinned types, 1,731 Wyckoff rows,
4,425 finite operations, and catalogue fingerprint
`707220a51f46101ed6b7d4e2401f8d0dd2e0416dd51864f9abce3913cd84f029`.
Training must call `receipt.require_for(live_catalogue)`; matching a filename is
not sufficient.

## 3. Size the candidate and ablation on one accelerator

Run a constant-density same-workload implementation comparison at 8, 16, and 40
atoms on one CUDA device for both the provisional `invariant_global` candidate
and the retained `equivariant_local` ablation.  This is not a parameter-, FLOP-,
or measured-compute-matched quality experiment. Start at batch size one and
float32 master precision; then test BF16 autocast only after float32 succeeds.
Do not use `--device auto` in recorded hardware evidence. Pin the wheel/source
revision, Python/PyTorch/CUDA/cuDNN versions, driver, container, and all
dependency artifacts alongside every report.

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 crystalatlas benchmark-backends \
  --device cuda:0 --precision float32 \
  --batch-size 1 --atoms-per-structure 8 --volume-per-atom 16 \
  --input-scalar-dim 219 --scalar-dim 256 --pair-dim 96 --vector-dim 32 \
  --num-heads 8 --num-layers 10 --radial-basis-dim 32 \
  --reciprocal-mode-radius 2 --direct-harmonics 2 \
  --cutoff 7 --max-candidates-per-pair 32768 \
  --max-edges-per-structure 500000 --max-total-edges 500000 \
  --max-dense-pair-interactions 500000 \
  --warmup-steps 10 --measurement-steps 50 --seed 1729 \
  --learning-rate 0.001 --cpu-threads 1 \
  --output backend-comparison-cuda-n8.json
```

Repeat this command with `--atoms-per-structure 16` and `40`, preserving the
constant volume per atom, exact commands, and report fingerprints. Increase
batch size only after the
40-atom batch-one case fits with explicit memory headroom. The CUDA allocator
telemetry from this command is only a both-models-resident process observation:
it is not backend-isolated and cannot rank backend memory.  The logical tensor
payload is a lower bound, not a capacity estimate.  A backend-specific capacity
datum requires a fresh subprocess containing only that backend and the same
workload contract.  Even then, this synthetic mildly triclinic workload is not
a support-wide upper bound: legal skewed
or compact cells can enumerate many more periodic images. Before fixing batch
its batch size, profile authenticated batches stratified by dense-pair count and
exact local edge count, and include the highest-edge legal stress cases found
under the
frozen metric support.
Report their distance to the configured per-structure and total edge budgets;
do not assume a budget itself is physically reachable.

The `CUBLAS_WORKSPACE_CONFIG` prefix is mandatory for this deterministic
PyTorch/CUDA contract and is fingerprinted in the report.  The command rejects
CUDA execution if it is absent or has a different value; do not set it inside
an already running Python process.

This command compares the two Transformer backend modules. It excludes composition,
cover selection, exact cover expansion/reduction, CFM path construction, and
the full staged objective, and the two backends are not parameter- or
compute-matched. Therefore a separate pilot must execute the real
`CrystalAtlasTrainingObjective` on authenticated batches and record peak
memory, examples/s, atoms/s, backend-labelled dense-pairs/s or periodic-edges/s,
and optimizer steps/s.  Those interaction counts must never share one
unlabelled "edge" field. CFM
training samples a path state and does not integrate an ODE; checkpoint-bound
RK4 generation is a separate inference pilot that must report NFE, attempts/s,
valid outputs/s, and peak memory under a predeclared fixed no-retry attempt
budget. The training measurement must execute through `ScalableTrainer` with
`ExactBatchCrystallographicAudit`; calling the objective alone omits mandatory
float64 crystallographic checks and trainer overhead.

The executable synthetic seam for that second measurement is:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 crystalatlas benchmark-objective \
  --source exact-fixture --model-profile default-v1 \
  --device cuda:0 --precision float32 --batch-size 1 \
  --warmup-steps 10 --measurement-steps 50 \
  --seed 1729 --learning-rate 0.0001 --ema-decay 0.999 --cpu-threads 1 \
  --output objective-cuda-exact-fixture.json
```

With `default-v1`, this command verifies the dense candidate's complete
objective/trainer/audit path on one exact three-atom P1 fixture.  It is not an
8/16/40-atom capacity comparison and supplies no local-backend result.
It does not close the production gate: the same report contract must
still be executed on authenticated `DatasetStore` batches spanning the frozen
topology and backend-appropriate interaction-count strata. The fixture source,
minimal P1/P-1 catalogue,
and exclusion of dataset I/O are explicit in its manifest.

## 4. Pass diagnostics before scale-out

Run the factors in this order on a tiny authenticated subset:

1. \(p_\omega(c)\): predeclare a relative complete-composition NLL-reduction
   threshold and require it at a fixed number of consecutive scheduled
   checkpoints, not just at one favorable crossing; on a held-out validation
   split, separately require a frozen complete-composition calibration
   threshold (for example Brier/ECE with its binning protocol);
2. \(\pi_\phi(\alpha\mid c)\): require legal scaffold Recall@\(K\) under a
   fixed no-retry budget;
3. \(\rho_{\theta,\alpha}(q\mid c)\): require Match@\(K\) with the true scaffold
   supplied; and
4. joint fine-tuning: compare oracle-scaffold and predicted-scaffold Match@\(K\)
   with paired query-level uncertainty.

The bundled synthetic smoke result intentionally does not pass this gate: after
four staged updates—composition, cover, geometry, then joint—loss fell by about
2.50%, while scaffold Recall@1 and both geometry Match@1 values remained zero.
That result proves the checkpoint-bound diagnostic path executes; it is not
memorization evidence. The smoke forecast carries no calibration vector and is
not a calibration result.

Increase optimization steps only on the frozen tiny corpus, preserve every
checkpoint and diagnostic manifest, and stop if the production matcher or
target corpus changes.

The scaffold and full-structure matchers must have separate, content-addressed
contracts that pin cell policy, parameters or tolerances, implementation
artifacts, and environment. CrystalAtlas provides strict canonical scaffold and
structure byte equality only through its mandatory prevalidating diagnostic
wrappers. Its current contract explicitly records
`metric_uncertainty_certified=false` and
`representation_invariant_physical_equivalence_claim=false`. It is an exact
accepted-partition diagnostic, not a conventional tolerance-based CSP score.
See [`CANONICAL_PARTITION_MATCHING.md`](CANONICAL_PARTITION_MATCHING.md). A
benchmark comparison must additionally freeze and report its compatible
physical similarity criterion. The local P1 fixture matcher used by the smoke
test is not a production all-space-group matcher.

## 5. Measure DDP rather than assuming it

After a one-device full-objective pilot passes, sweep 1, 2, 4, and then 8 GPUs
at fixed global batch and again at fixed per-GPU batch. Record both throughput
and peak memory. DDP efficiency is

\[
\eta(G)=\frac{R_G}{G R_1},
\]

where \(R_G\) is measured structures/s for \(G\) devices. No CPU or
single-device benchmark determines \(\eta(G)\). Keep sampler-policy, data-order,
objective, precision, accumulation, checkpoint, and software/container
fingerprints fixed across the sweep.

## 6. Large-run release conditions

A large training allocation is justified only when all of the following are
archived and externally trust-pinned:

- an authorized source manifest and authenticated compiled dataset;
- the all-230 crystallography receipt;
- leakage-free split manifests;
- a persistent tiny-overfit pass using production matchers;
- a successful one-device full-objective memory/throughput report;
- a thresholded held-out complete-composition calibration receipt;
- a checkpoint-bound RK4 generation memory/NFE/throughput report;
- an exact checkpoint-resume replay in which an interrupted and uninterrupted
  run perform the same next optimizer step and produce identical model,
  optimizer, EMA, scheduler, sampler-cursor, and RNG states under the same
  world size, backend, and device type;
- a measured DDP scaling curve;
- a green test receipt from the pinned production wheel/container/driver stack;
- a frozen evaluation protocol that counts invalid and missing attempts as
  failures and applies identical post-generation validation to every baseline.

Only after those gates should the package train at scale and compare against
MatterGen or another generator. Superiority requires prospective, matched
stable-unique-novel yield with confidence intervals and identical relaxation,
energy, hull, deduplication, and compute budgets—not merely more generated
samples or lower training loss.
