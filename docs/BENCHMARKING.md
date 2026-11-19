# Single-device benchmarking

`crystalatlas benchmark-geometry` measures only the retained
`equivariant_local` periodic \(SE(3)\)-equivariant backend.  It predates the dense
`invariant_global` candidate and is kept as a local-edge capacity baseline; it
must not be reported as a benchmark of the provisional implementation default.  It does not
require CUDA.  It is a capacity-planning microbenchmark, not a predictive
experiment or model-quality result.

The benchmark fixes and fingerprints:

- the geometry factor, device request and resolved runtime;
- master/autocast precision and deterministic backend policy;
- batch size, atoms, model widths, layers, heads, cutoff and edge budgets;
- a constructive periodic workload and parameter initializer that consume no
  ambient random stream;
- AdamW configuration, warmup count and measurement count; and
- synchronization at every measured device boundary.

It records three phases separately:

1. exact periodic-image enumeration;
2. Transformer forward evaluation with the exact edges precomputed; and
3. an end-to-end **Transformer microbenchmark** step containing edge
   enumeration, forward, loss, backward and the optimizer update.

The third phase is not a full CrystalAtlas objective: it deliberately excludes
the composition and cover factors, exact cover expansion/reduction, and the CFM
path construction. The report records that exclusion explicitly, so its
throughput cannot be quoted as complete model-training throughput.

For a local-backend phase with \(I\) measured iterations, batch size \(B\), atom count
\(N\) per structure, exact edge count \(E\) per iteration and total elapsed
time \(T\), the report uses

\[
R_{\rm structures}=\frac{IB}{T},\qquad
R_{\rm atoms}=\frac{IBN}{T},\qquad
R_{\rm edges}=\frac{IE}{T}.
\]

Raw synchronized iteration durations remain in the canonical JSON report, so
means and dispersion can be recomputed rather than inferred from rounded
console output.

## Running it

The smallest useful CPU smoke profile is:

```bash
crystalatlas benchmark-geometry \
  --device cpu \
  --batch-size 1 \
  --atoms-per-structure 4 \
  --warmup-steps 2 \
  --measurement-steps 10 \
  --cpu-threads 1 \
  --output geometry-cpu.json
```

Use the same command with `--device cuda:0` on a CUDA worker. An explicitly
requested unavailable accelerator is an error; the command never silently
falls back to CPU. `--device auto` resolves CUDA, then MPS, then CPU, and the
resolved device is recorded.

Every deterministic CUDA benchmark must start the process with
`CUBLAS_WORKSPACE_CONFIG=:4096:8` (the lower-workspace `:16:8` setting is also
accepted).  The CLI fails closed if neither deterministic setting is present,
and the resolved value is recorded in the runtime manifest.  Setting it after
CUDA has initialized is not an equivalent contract.

For hardware sizing, first sweep atom count and batch size on one accelerator
while keeping the production model widths, precision, cutoff and edge budgets
fixed. A successful report proves only that exact workload fit and ran on that
runtime.

## Geometry-report memory semantics

The report distinguishes two quantities that must not be conflated:

- **logical tensor payload** exactly counts parameters, buffers, gradients,
  optimizer state, synthetic inputs/targets, exact edge tensors and retained
  outputs;
- **backend allocator peak** is recorded only when the backend exposes a
  resettable measurement (currently CUDA).

Logical payload excludes saved activations, temporary kernels, allocator
fragmentation and process/framework memory. Consequently it is a lower bound,
not a device-capacity estimate. CPU reports leave allocator peak fields null;
they do not relabel process RSS as tensor memory. MPS reports current allocator
memory when PyTorch exposes it but do not invent a phase-local peak.

## What can and cannot be projected

For the *same recorded* single-device workload, a first wall-time projection is

\[
T_{\rm target}=I_{\rm target}\,\overline t_{\rm measured}.
\]

This is useful for deciding whether to run a longer single-device pilot. It
does not establish multi-GPU throughput. If \(G\) devices are used, the actual
time would involve an unknown efficiency \(\eta(G)\):

\[
T_G=\frac{T_1}{G\,\eta(G)}.
\]

One-device measurements contain no evidence for \(\eta(G)\), because they do
not exercise DDP collectives, interconnects, distributed input contention,
stragglers or rank imbalance. Multi-device scaling therefore requires a
separate measured sweep at fixed global workload. `benchmark-objective` records
finite synthetic-fixture component losses as execution evidence. Neither
benchmark estimates held-out or corpus NLL/calibration, Recall@\(K\),
structural validity, novelty, stability, or predictive superiority.

Failed phases raise an error and do not produce a report marked `completed`.
Every successful report is strict JSON (no NaN/Infinity) with a SHA-256
fingerprint over its complete evidence payload.

The full-objective schema v2 records `geometry_interaction_kind` explicitly:
`dense_atom_pairs` for `invariant_global` and `periodic_image_edges` for
`equivariant_local`.  Their counts are not physically or computationally
interchangeable, so no generic "edges/s" comparison is permitted.

## Same-workload geometry-backend implementation comparison

`crystalatlas benchmark-backends` runs `invariant_global` and
`equivariant_local` against the same deterministic inputs, targets, batch size,
atom count, scalar/vector widths, head count, layer count, optimizer and physical
number density.  The default mildly triclinic cell has volume
(N\times16\,\text{\AA}^3), so atom-count sweeps hold density fixed rather
than silently compressing the structure into a fixed box.

“Same workload” does not mean parameter- or compute-matched.  This command
compares the two Transformer backend modules rather than the complete
expand--reduce intrinsic fields, and it is not the predictive ablation in
theory section 6.8.  Its cutoff is an explicit workload parameter.  The default
and example below use the production-domain value (7\,\text{\AA}); reports
with different cutoffs must not be pooled.

```bash
crystalatlas benchmark-backends \
  --device cpu --precision float32 \
  --batch-size 2 --atoms-per-structure 20 --volume-per-atom 16 \
  --input-scalar-dim 219 \
  --scalar-dim 256 --pair-dim 96 --vector-dim 32 \
  --num-heads 8 --num-layers 10 \
  --radial-basis-dim 32 --reciprocal-mode-radius 2 \
  --direct-harmonics 2 --cutoff 7 \
  --warmup-steps 3 --measurement-steps 12 --cpu-threads 1 \
  --output backend-comparison-cpu.json
```

For each backend the report records raw synchronized durations for end-to-end
forward and for a complete loss/backward/AdamW step.  The local end-to-end
phases include exact periodic-image enumeration.  By default a separately
labelled local forward with prebuilt interactions is also measured; pass
`--no-local-prebuilt-core` to omit it.  Parameter counts, finite outputs,
finite gradients, exact optimizer-step counts, logical tensor bytes and
backend allocator telemetry are recorded separately for each backend.  Forward
and training measurements alternate backend order (AB,BA,AB,\ldots); raw
paired samples, medians, and linearly interpolated quartiles are retained.

The workload fingerprint binds the seed; dtype, shape, and SHA-256 digest of
every represented input and target; the exact local interaction tensors; cell
determinants and determinant-derived number densities; units; optimizer; and
execution order.  The runtime fingerprint binds the CrystalAtlas version,
resolved model configs, hashes of both backend source modules and the benchmark
code, and the Python/PyTorch/device stack.  Batch members are disclosed as
global fractional translations of one periodic structure, not independent
structural samples.  Local interaction membership is evaluated in float64
topology arithmetic after conversion from the recorded master-precision
inputs.

Synthetic scalar features, targets, and benchmark model parameters use an odd
integer-affine permutation modulo \(2^{24}\), exact integer-to-Float64
conversion, and normal finite Float32-exact power-of-two scaling.  No random
generator or transcendental library participates in their construction.  This
makes their bytes independent of vector-libm dispatch, allocation alignment,
and CPU thread partitioning; frozen digest tests guard the production-width
fixture and parameter initializer.  Method and workload-generator revisions
change whenever these numerically relevant fixture bytes change.

The interaction labels are intentionally different:

- `dense_ordered_atom_pairs` has count (BN^2);
- `exact_periodic_image_edges` has the enumerated periodic-image count.

Dense pairs are never called edges.  These counts describe different
representations and are not evidence of equal compute or physical fidelity.
The models share principal widths but their pair, Fourier, radial and vector
parameterizations differ, so the report explicitly says they are not
parameter-, weight- or FLOP-matched.  Its timing ratios are descriptive for the
recorded device and stack only.  They do not establish predictive quality,
sample quality, asymptotic superiority or multi-device scaling.

Numerical fields are deliberately narrow: every measured scalar loss is
checked, while output and parameter-gradient finiteness refer to the final
measured step rather than pretending that every intermediate tensor was
audited.  Logical interaction memory includes retained exact and
activation-precision edge tensors with shared storage counted once.  Allocator
peaks are **not backend-comparable**: both models remain resident and CUDA cache
state is shared.  A memory-superiority claim would require one backend per
fresh, identically configured subprocess; the current report explicitly marks
backend isolation and allocator comparison as false.

Only the float32 run is presently a common-precision systems comparison.  In
BF16/FP16 mode the invariant backend deliberately keeps Fourier phases,
reciprocal vectors, and covariant bases in FP32, whereas the local backend uses
activation-precision displacement and distance tensors after exact edge
membership is fixed.  Such reports are labelled **backend-native mixed
precision** evidence; they are not arithmetic-precision-matched architecture
comparisons.  BF16/FP16 adoption additionally needs an identical-initial-state
FP32 deviation study for losses, gradients, parameter updates, and outputs.

## Full training-objective benchmark

`crystalatlas benchmark-objective` measures the complete joint objective through
`ScalableTrainer`, after one optimizer update for each factor-specific phase.
Every warmup and measured joint window executes:

- the exact float64 batch crystallographic audit;
- composition and legal-cover negative log-likelihoods;
- CFM path sampling and its current-state metric loss;
- exact cover expansion, the selected dense-pair or periodic-edge geometry
  backend, and intrinsic tangent reduction;
- backward, gradient checks, AdamW, and optional EMA; and
- a synchronized device boundary around the complete window.

The bundled source is intentionally one exact synthetic P1 C/N/O fixture. It
tests the numerical and trainer integration path; it does not represent an
authenticated materials distribution. A deterministic CPU smoke is:

```bash
crystalatlas benchmark-objective \
  --source exact-fixture --model-profile micro-v1 \
  --device cpu --precision float32 --batch-size 1 \
  --warmup-steps 2 --measurement-steps 10 \
  --seed 1729 --learning-rate 0.0001 --cpu-threads 6 \
  --output full-objective-cpu-fixture.json
```

Use `--model-profile default-v1 --device cuda:0` for an architecture-width CUDA
systems check. The report still excludes authenticated dataset loading and
collation, checkpoint I/O, an RK4 generation trajectory, DDP communication,
optimization convergence, and predictive quality. A production pilot must
therefore repeat the objective measurement on authenticated topology-stratified
batches and separately measure checkpoint-bound generation and DDP scaling.
