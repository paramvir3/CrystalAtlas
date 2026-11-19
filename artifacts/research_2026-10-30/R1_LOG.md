# R1 — CFM target-variance reduction by gauge-aligned minibatch OT coupling

Date: 2026-10-30, amended 2026-11-10. Host: tcm-pc38 (RTX 5060 Ti). Status: **closed,
prediction falsified** (aligned E1 seed 0, 2026-11-10; see Results). The lever is dropped
as a fix for the E1 z-marginal. The coupling stays in the code as an opt-in.

## Hypothesis
The learned endpoint accuracy of the geometry flow is limited by the conditional
variance of the CFM target velocity (F10 loss decomposition: > 99 % of the loss is
irreducible target variance; z-marginal TV scales as N^-0.54 with the number of path
samples; E1 fails A-F10L at 0.093–0.095 against 0.08). A coupling that reduces that
variance without changing either endpoint marginal should lower the TV at a fixed
budget.

## Construction (Proposition 2.9d)
`coupling="gauge-ot-v1"` in `geometry_cfm_objective_batch`: for each record draw K = 8
uniform residual actions; cost of (base i, record j, action k) =
d_z(z0_i, z1_j) in the chart-origin pullback metric + d_eta(gamma_k^-1 eta0_i, eta1_j)
with the half-open lift in the reference translation-quotient metric divided by N;
minibatch assignment (Hungarian) on min_k cost; the path is built from the unpushed base
and pushed by the chosen action, so it starts at the drawn base and ends at gamma eta1.
Both marginals are preserved exactly (Proposition 2.9d); legacy draws are bitwise
unchanged under `coupling="independent"` (`tests/test_gauge_ot_coupling.py`).

## Prediction and falsifier
Toy E1 (Pa-3 8c, sigma_z = 0.2), the A-F10L model options and budget (30 000 x 256,
seed 0): the aligned coupling gives z-marginal TV < 0.08 while the same-code
independent baseline stays near 0.093–0.095. If the aligned TV is not lower than the
baseline by more than the Monte Carlo SE (0.0055), the hypothesis that coupling variance
limits E1 is rejected for this toy and the lever is dropped.

## Runs
- `~/crystalatlas-gpu-20261025/r1/E1_gauge-ot-v1_seed0.{json,log}`
- `~/crystalatlas-gpu-20261025/r1/E1_independent_seed0.{json,log}` (same code baseline)

## Baselines measured (independent coupling, same code, 30 000 x 256, float32, CUDA)
| seed | z-marginal TV | verdict against 0.08 |
|---|---|---|
| 0 | 0.0947 | FAIL |
| 1 | 0.0930 | FAIL |

Oracle control z-TV 0.0048; Monte Carlo SE 0.0055. Files:
`~/crystalatlas-gpu-20261025/r1/E1_independent_seed{0,1}.{json,log}`.

The aligned runs started on 2026-10-30 (`E1_gauge-ot-v1_seed{0,1}.log`,
`E2_gauge-ot-v1_seed0.log`) were ended from an interactive session on the account before
their first seed finished. Those logs contain the header only. **No aligned-coupling
result has been observed at the time of this amendment.**

## Amendment 2026-11-10 (made before any aligned result was observed)
Three changes to the construction; the prediction and the falsifier above are unchanged.

1. **Assignment within one cover.** The geometry field is conditioned on the cover, so
   the base marginal has to be preserved for each cover separately. The first
   implementation assigned over the whole batch. A cross-cover assignment keeps the batch
   base marginal but biases each cover's base marginal (counterexample on the 1-D torus:
   record 1 with trivial group, record 2 with group {e, eta -> -eta}; the full-batch rule
   gives P(gamma* = s) = 3/4 for record 2). The batch is now partitioned into blocks of
   equal cover and the Hungarian assignment runs inside each block. E1 and E2 have a
   single cover, so their runs are one block and are unaffected by this change.
2. **Exact quotient when the group table is small.** When the residual-action table of
   the cover has at most K rows, the K i.i.d. candidates are replaced by the whole table
   (every group element exactly once, Proposition 2.9d rule (ii)). The cost is then
   min over the whole residual group, with no candidate sampling noise and no random
   draw. E1 has a residual group of order 4 under `normalizer-v3`, so the E1 run uses 4
   candidates per record (the exact quotient cost), not 8 i.i.d. draws. Larger tables
   keep K i.i.d. uniform candidates (rule (i)).
3. **Batched candidate sampling.** Residual actions are coded by digits
   (`ResidualActionTable`: Fisher-Yates digits per identical-orbit group, one bit per
   free P-1 row under `relabel-v1`, one index per orbit gauge and one normalizer index
   under `normalizer-v2/v3`). Uniform digits give the same law as the legacy sampler.
   Profile before the change, E1 batch 256: aligned 0.756 s/step against independent
   0.207 s/step, with 2048 Python-level action draws per step.

Proposition 2.9d was restated accordingly (fixed cover, rules (i) and (ii), any
measurable cost, null tie sets for rule (ii), marginals at a uniformly chosen batch
position). `coupling="independent"` stays bitwise equal to the pre-edit source (losses,
path states, velocities and generator state; relabel-v1, mixed-cover, normalizer-v2 and
normalizer-v3 cases).

Consequence for real data, recorded in advance: with one record per cover in a batch
every block has size 1, so only the gauge alignment acts and the assignment gain is
absent. Any MP-20 benefit of this lever comes from the residual-group alignment alone.

## Runs after the amendment
- `~/crystalatlas-gpu-20261025/r1/E1_gauge-ot-v1_exact_seed{0,1,2}.{json,log}`

## Results (2026-11-10)
E1, 30 000 steps x batch 256, float32, CUDA, 32 768 endpoint draws. Monte Carlo SE 0.0055.
The aligned run used the exact quotient (residual table of 4 rows, `exact_quotient: true`).

| coupling | seed | mean loss, last 1000 steps | z-marginal TV | rel. error E z | rel. error E z^2 | worst Fourier error | verdict |
|---|---|---|---|---|---|---|---|
| independent | 0 | 9.490 | 0.0947 | 0.0219 | 0.0036 | 0.0103 | FAIL (TV) |
| independent | 1 | 9.471 | 0.0930 | 0.0420 | 0.0183 | 0.0070 | FAIL (TV, mean) |
| gauge-ot-v1 | 0 | 0.0485 | 0.0997 | 0.0524 | 0.0158 | 0.0069 | FAIL (TV, mean) |
| exact oracle control | n/a | n/a | 0.0048 | 0.0110 | 0.0001 | 0.0069 | PASS |

Thresholds: z-TV 0.08, moments 0.03, Fourier 0.025. File:
`~/crystalatlas-gpu-20261025/r1/E1_gauge-ot-v1_exact_seed0.{json,log}` (training 3678.8 s).

**Verdict against the registered prediction.** The prediction (aligned z-TV < 0.08) failed.
The falsifier fired: the aligned z-TV is not lower than the baseline at all (0.0997 against
0.0947 and 0.0930, a difference of about one Monte Carlo SE in the wrong direction). By
the rule fixed before the run, the hypothesis that the coupling variance limits the E1
z-marginal at this budget is rejected for this toy and the lever is dropped. Seeds 1 and 2
and the aligned E2 run are not run: the seed-to-seed spread of the baseline is 0.0017, and
reaching 0.08 would need a seed effect of 0.02. E1 stays FAILED under A-F10L.

**What the run does establish.**
1. The coupling does what Proposition 2.9d says. The per-sample CFM loss fell from 9.49
   to 0.0485, a factor of 196, and the torus marginal stayed at the oracle-control level
   (worst Fourier error 0.0069, equal to the control), so the endpoint marginal in eta is
   preserved in practice as well as in the proof.
2. Removing almost all conditional target variance did not improve the z endpoint. The
   attribution of the E1 z under-fit to the gradient signal-to-noise of the target
   variance (F10 loss decomposition of 2026-10-25, quoted as an explanation in A-F10L) is
   therefore not supported by this intervention. The approved amendment's protocol and
   thresholds do not depend on that explanation and are untouched.

**What the run cannot decide.** The two couplings regress on different fields. The E1 data
law in z is a mixture of three components (centres -0.5, 0.3, 1.0, sigma_z = 0.2). Under
the independent coupling the exact lattice velocity is smooth at early times and steep
between components only as t -> 1; under an assignment coupling the field is steep
between the base regions sent to different components at every t. A loss value under one
coupling is not a fit quality under the other. The result says that noise removal at a
fixed budget did not help; it does not say whether noise or the steepness of the
late-time lattice velocity limits the independent-coupling fit. The earlier measurement
(lattice rho_v about 0.11 at t = 0.9, held-out E1, 2026-10-26) is consistent with a
late-time lattice error under either reading.

**Open, not run.** Levers that address a late-time lattice error directly: the time
sampling density (more mass near t = 1), an endpoint (x1) parametrization of the lattice
head, and an exponential moving average of the weights in `train_on_toy`. Each would need
its own registered prediction on a fresh toy, because E1 and E2 are no longer held out.
Relevance for real data is limited: with composition and cover fixed, the lattice law of
MP-20 is almost always one component, and the certified matcher is volume-normalized.

---

# R2 — Exact model likelihood for candidate ranking (no energy model)

Date: 2026-10-30. Status: implemented and certified on the exact oracle.

`flow.log_density_transport` integrates the learned ODE backward from t = 1 with the
divergence of the field (exact trace or Hutchinson) on R^{d_z} x the flat torus;
`LearnedCrystalGenerator.geometry_log_density` returns log rho_{theta, alpha}(q_1) =
log rho_0(q_0) - int div v (Proposition 2.9e). Certificate: run on the exact Fourier
oracle's own velocity field, the transport reproduces `oracle.log_density(1, q_1)` to
2e-3 on both F10 toys (192 RK4 steps), and the Hutchinson estimator is unbiased
(`tests/test_flow_log_density.py`). `rank_samples` combines the exact selector
log-probability of the cover with the geometry log density; cross-dimension
comparisons are flagged, not hidden. Next: measure on MP-20 (R3) whether likelihood
ranking raises CSP match@1 over random candidate choice at a fixed candidate budget.

---

# R3 — First real-data training under the v7 policy (toward MP-20 numbers)

Date: 2026-10-30. Status: compile running on tcm-pc10; training runner smoke-tested.

Facts that reshaped the plan: the v6 research pilot store (2026-10-12) accepted 193/512
records (37.7 %): 143 ranking-bin-boundary rejections (a numerical guard, not physics),
88 symmetry-window instabilities at symprec 1e-5, 65 lattice-fidelity rejections. The
B1/B2 remediation's v7 policy (tau = 1e-4 A for classification and idealization,
scale-relative ranking quantum 1e-9, quantization-equivalent orbit merging, primitive
bound 20/80) accepted 431/512 in the S2 study, and the Loewner chart has since addressed
most of the remaining chart-domain rejections.

Steps:
1. `scripts/compile_mp20_research_v7.py`: same reviewed source declaration, bundle and
   catalogue certification as the pilot (512 records, research training use approved
   2026-10-12/17), v7 policy, STRUCTURE deduplication, four split views; destination
   `data/research-pilots/mp20-512-v7-20261030` on the shared home. Builder fingerprint
   recorded, not pinned.
2. `scripts/train_research_v7.py`: factor-first training (composition, cover, geometry)
   on every training record of the composition-disjoint split, checkpoints and
   train/validation losses after each phase, then composition-conditioned CSP on the
   frozen test partition with the certified physical matcher: match@1, match@K and
   match@1 under the exact likelihood ranking (R2). Smoke-tested end to end on the v6
   store (6 records, 1 epoch per phase, 2 candidates, CPU).
3. Full-corpus (45,229 rows) training needs the source-rights declaration generalized
   beyond the 512 pilot (owner decision) and a bundle for all rows.

Caveat recorded in advance: a few-hundred-record corpus gives a learning signal and a
small-sample CSP rate, not a benchmark number.

## R3 compile result (2026-10-30, tcm-pc10, 1824 s)
v7 policy with the default Loewner chart: **504 of 512 accepted**, 8 rejected
(4 symmetry-window instabilities, 2 primitive-ranking resource limits, 1 identity-cell
guard, 1 torus-cell guard), 0 duplicate aliases. Splits (all four views): 404 train /
50 validation / 50 test. Receipt: `mp20-512-v7-compilation.json` (builder fingerprint
f1f9d1ee…6e5c). Training run 1 launched on pc10: `research-medium` profile, epochs
30/30/120, batch 16, composition-disjoint split, CSP test with 20 candidates per
composition (`~/crystalatlas-gpu-20261025/r3/v7_medium_run1/`).

## R3 training run 1 outcome (partial)
Run 1 was ended on 2026-10-30 together with the R1 jobs (interactive session on the
account; no crash or out-of-memory evidence in the logs). It completed the composition
phase (train NLL 23.78 to 7.74, validation NLL 23.0 to 22.4: the composition factor
memorises 404 training compositions and does not generalise to disjoint ones at this
corpus size, as expected for a composition-disjoint split) and one cover epoch. No
geometry training and no CSP number exist from run 1. Throughput on the shared pc10 GPU
was about 5.5 minutes per composition epoch, which is the first thing to profile before
run 2. From 2026-11-10 the owner restricts GPU use to tcm-pc38.

## R3 run 2 outcome (2026-11-10, tcm-pc38, default options)
Baseline with no opt-in mechanism, from the frozen code snapshot
`r3/v7_medium_run2_code` (sha256 of the sorted per-file sha256 list
`3747bf1d758224d52e79cfd9c05709d560002dca418ceead4a600d9afd8ca0e4`). Settings:
`research-medium`, domain v7, composition-disjoint split (404 / 50 / 50), mixed
batches of 16, epochs 20 / 150 / 1000, AdamW at 2e-4 for every factor (constant, no
schedule), EMA 0.999, validation every 5 / 5 / 25 epochs, selection of the evaluated
candidate (raw or EMA) with the lowest validation loss, seed 20261030. Wall time
9428 s, 30 400 optimizer steps. Output `r3/v7_medium_run2/`.

Phase-end evaluations (`result.json`; the CFM regression there uses one fixed draw):

| state | train comp. NLL | val. comp. NLL | train cover NLL | val. cover NLL | train CFM | val. CFM |
|---|---|---|---|---|---|---|
| baseline | 23.777 | 23.006 | 11.165 | 9.952 | 15.756 | 18.186 |
| after composition | 15.200 | 16.691 | 11.165 | 9.952 | 15.756 | 18.186 |
| after cover | 15.200 | 16.691 | 6.341 | 7.526 | 15.756 | 18.186 |
| after geometry | 15.200 | 16.691 | 6.341 | 7.526 | 8.314 | 12.028 |

Selected candidates: composition phase epoch 15 of 20 (raw); cover phase epoch 15 of
150 (raw, 7.526; the EMA minimum is 7.582 at epoch 50); geometry phase epoch 150 of
1000 (EMA, 9.523 as the mean of the four fixed validation draws). Checkpoint
`after-geometry.pt`, sha256
`119e249ac6cb54d7cd20ae861c364088c99980e97298765fee4fc8dbd84ba756`.

Every factor overfits the 404-structure corpus. Cover: validation NLL rises from its
minimum 7.526 to 13.217 (raw) at epoch 150 while the training NLL falls to 3.297.
Geometry: the four-draw validation CFM regression falls to 9.523 at phase epoch 150 and
then rises monotonically to 12.399 (EMA) and 12.877 (raw) at epoch 1000, while the
training loss falls from about 9.5 to about 7.5. Early stopping on the validation
partition is therefore doing real work in all three phases, and 850 of the 1000
geometry epochs were spent past the optimum.

Estimator note. The phase-end validation CFM (12.028) and the selection value (9.523)
are estimates for the same restored weights with different draws (one draw against
the mean of four), so they are not comparable with each other. The check that the
restored weights reproduce both numbers is `scripts/diagnose_cfm_draws_v7.py`.

Result (`r3/v7_medium_run2_cfm_draws_after-geometry.json`, 12 draws on the validation
view, runner's own `evaluate_losses`): the phase-end call returns
12.027618191801011 / 16.690729522705077 / 7.525781865119934 (CFM, composition, cover),
bit-identical to `after-geometry-evaluation.json`, and the four-draw geometry call
returns 9.523490031719776, bit-identical to the selected value. The checkpoint
therefore holds the selected weights (phase epoch 150, EMA). Per-draw values:
12.028, 8.355, 8.480, 9.232, 8.836, 8.892, 9.232, 12.158, 8.111, 9.966, 9.115, 8.512;
mean 9.410, standard deviation between draws 1.346, standard error of the 12-draw
mean 0.389 (time and base draws on the 50 fixed records; not the sampling error of
the partition). A one-draw validation CFM thus carries an error of about 1.3 and
must not be compared across estimators; comparisons between epochs of one run use
common draws and are far tighter than this. Follow-up for the runner: report the
phase-end CFM with the selection's draws and state the draw count in the JSON.

## R3 cover-selector factor decomposition (run 2, `after-cover`, 2026-11-10)
`scripts/diagnose_cover_factors_v7.py` splits the teacher-forced cover NLL of the chain
p(nu | c) p(G | c, nu) prod_r p(a_r | c, nu, G, a_<r) into its three terms and compares
each with the policy that is uniform on the exact legal action set of the same
decision. Raw NLLs of different partitions are not comparable, because their legal
action sets differ (uniform total 11.020 on train, 9.845 on validation, 11.066 on
test); the comparable quantity is the gain over that uniform policy, in nats per
structure:

| factor | train NLL | val. NLL | test NLL | train gain | val. gain | test gain |
|---|---|---|---|---|---|---|
| nu | 0.990 | 2.116 | 1.636 | 1.086 | -0.014 | 0.564 |
| space group | 2.325 | 2.381 | 2.627 | 1.868 | 1.711 | 1.679 |
| rows | 3.026 | 3.029 | 3.282 | 1.724 | 0.622 | 1.277 |
| total | 6.341 | 7.526 | 7.546 | 4.679 | 2.319 | 3.520 |

Reading. The space-group factor transfers to unseen compositions (gain 1.87 on train,
1.71 and 1.68 held out). The nu factor and the row factor do not: nu keeps none of its
gain on validation and about half on test; rows keep about a third on validation and
three quarters on test. The two held-out partitions hold 50 records each and differ by
1.2 nats in total gain, so partition noise is of the same size as the effects of
interest; comparisons between model variants must be paired per record. Space-group
top-1 / top-5 / top-10 / top-20: train 0.36 / 0.67 / 0.85 / 0.97, validation
0.40 / 0.64 / 0.84 / 0.92, test 0.40 / 0.58 / 0.74 / 0.86. The untrained selector
(`after-composition`) sits at the uniform level (validation 9.952 against 9.845), and
the diagnostic reproduces the runner's cover NLL to 4e-9.

## Periodic-table element prior (opt-in, 2026-11-10)
`ModelConfig.element_embedding = "periodic-v1"` (runner option `periodic-elements`)
replaces the free per-element table e(Z) at four sites (the selector's composition
encoder and action encoder, the field's composition encoder and species embedding) by

    e(Z) = P[period(Z)] + G[column(Z)] + B[block(Z)],   e(0) = 0,

with exact integer coordinates of the 32-column long form: noble-gas boundaries
N = (2, 10, 18, 36, 54, 86, 118), period p with N_(p-1) < Z <= N_p, period length
L = N_p - N_(p-1), k = Z - N_(p-1), column = k if k = 1 or (k = 2 and L > 2), else
32 - L + k; block s for columns 1-2 and He, f for 3-16, d for 17-26, p for 27-32
(Lu and Lr are placed under Y; La-Yb and Ac-No are the f block). The 118 cells are
distinct, so e can still separate every pair of elements, but a statistic learned for
one element moves its whole period, column and block: 7 + 32 + 4 = 43 trainable rows
per site instead of 118. Each table is initialised N(0, 1) scaled by 1/sqrt(3), so e(Z) has unit
variance per entry like the free table. The default `learned-v1` keeps the identical
constructor call, state-dict keys, initial values and configuration fingerprint
(`tests/test_periodic_elements.py`, 11 tests). The composition model's token embedding
is not changed.

Motivation: the nu and row factors above fail on unseen compositions, and with 404
training structures most elements occur in a handful of compositions, so a free
embedding gives an unseen composition nothing to interpolate from.

## Registered A/B: periodic prior on the cover selector (registered 2026-11-10, before any treatment run)
Arms: control `--options default`, treatment `--options periodic-elements`. Everything
else identical: `research-medium`, domain v7, composition split, mixed batches of 16,
`--epochs 1,60,1`, `--eval-every 1`, `--validation-draws 1`, `--select-best`,
`--ema-decay 0.999`, learning rates 2e-4, `--skip-csp`; seeds 20261030, 20261031,
20261032 (model initialisation and batch order; the split does not depend on the
seed). The cover phase is the measurement; one composition epoch and one geometry
epoch only satisfy the runner. Sixty cover epochs contain both minima of run 2 (raw at
epoch 15, EMA at epoch 50). Code: frozen snapshot `r3/v7_eval_20261110_code`
(sha256 of the sorted per-file sha256 list
`d8e24f700290f70a55fa13768c1899d841cefc1cc8cb7b92791d78f1f78ef885`). After each run the
factor diagnostic is run on its `after-cover` checkpoint for validation and test.

- P1 (primary, as planned): the best validation cover NLL (`selection-cover.json`,
  `selected.cover_nll`), averaged over the three seeds, is at least 0.5 nats lower in
  the treatment than in the control.
- P2 (not selection-biased): test cover NLL of the validation-selected checkpoint.
  For each of the 50 test records the difference treatment minus control is averaged
  over the three seeds; the prediction is a negative mean with magnitude above two
  standard errors over the 50 records.
- P3 (mechanism): the gain appears in the nu and row factors, not in the space-group
  factor.

Falsifier: P1 below 0.5 nats falsifies the registered size of the effect; a P2 interval
that contains zero means no evidence of a benefit. The option enters run 3 only if P1
and P2 both hold. Consistency check of the protocol: the control at seed 20261030 has
the same initial weights and cover batch order as run 2, and a frozen factor is not
changed by another factor's phase (run 2's cover NLL is bit-identical before and after
its composition phase), so its validation cover NLL at epochs 5, 10, 15, ... is
expected to reproduce run 2 (7.733, 7.608, 7.526, ...); a mismatch would reveal a
dependence of the cover phase on the length of the composition phase. Limits: three
seeds, 50 records per held-out partition, selector only (the field's use of the prior
is measured in a full run).

## A/B verdict: periodic prior on the cover selector (2026-11-10)
Six runs (two arms, three seeds) from the queue `r3/ab_periodic/`; analysis
`r3/ab_periodic_analysis.py`, output `r3/ab_periodic_analysis.txt`. Every run made 120
validation evaluations (60 cover epochs, raw and EMA), and every selected candidate is
the raw weights. The factor diagnostic's validation total reproduces the selected value
to 3e-8 in all six runs.

Consistency check: passed. The control at seed 20261030 gives validation cover NLL
7.732968893051147, 7.608347821235657, 7.525781865119934 at epochs 5, 10, 15,
bit-identical to run 2, so the cover phase does not depend on the length of the
composition phase. The same run's minimum is 7.408974 at epoch 9: run 2's five-epoch
evaluation grid missed a value 0.117 nats below its selection (7.526 at epoch 15). The
cover phase of a full run should evaluate every epoch.

| seed | control selected (epoch) | periodic selected (epoch) | difference |
|---|---|---|---|
| 20261030 | 7.408974 (9) | 7.102504 (17) | -0.306470 |
| 20261031 | 7.252371 (10) | 7.132487 (12) | -0.119884 |
| 20261032 | 7.322977 (12) | 6.945113 (20) | -0.377864 |
| mean | 7.328107 | 7.060035 | -0.268073 |

- P1: **fails.** The mean selected validation cover NLL is 0.268 nats lower in the
  treatment; the registered size was at least 0.5 nats.
- P2: **holds.** Test cover NLL of the validation-selected checkpoints, per-record
  difference treatment minus control averaged over seeds, 50 records: control 7.5666,
  periodic 7.3214, difference -0.2453, standard error 0.1019 (ratio -2.41); per seed
  -0.2579, -0.2146, -0.2634.
- P3: the direction holds; only the row factor is resolved. Test differences: space
  group +0.0324 (standard error 0.0434), nu -0.1293 (0.0746, ratio -1.73), rows -0.1484
  (0.0528, ratio -2.81). Validation: space group -0.0166 (0.0311), nu -0.1584 (0.0829),
  rows -0.0931 (0.0674), total -0.2681 (0.1126, ratio -2.38).
- Test ranks, mean over seeds, control then periodic: space-group top-1 0.413 / 0.407,
  top-5 0.593 / 0.607; nu top-1 0.247 / 0.293; all rows maximal 0.353 / 0.400; mean row
  errors 1.200 / 1.160.

Verdict by the registered rule: the registered effect size is falsified, so the option
does not enter run 3 automatically. What the data support: a gain of about 0.25 nats
per structure with the same sign in all three seeds on both held-out partitions, about
half the registered size, located in the nu and row factors and absent in the
space-group factor. Including it in run 3 regardless is an owner decision. Limits as
registered: three seeds, 50 records per partition, selector only.

## R3 CSP measurement: run 2 `after-geometry`, reduced-formula track (2026-11-10)
Protocol `scripts/csp_evaluation_v7.py` (schema v2 for this measurement), driver
`scripts/evaluate_csp_research_v7.py`, frozen snapshot `r3/v7_eval_20261110_code`
(code fingerprint `5e705eb3e416751a6d5b3ca43e8604ccc121631cc6dd060821ad257f5dc7f31a`),
checkpoint sha256 `119e249a...ba756`, protocol seed 20261530 (run seed + 500), 20
attempts for each of the 50 test compositions, bucket sampler, float32, CUDA,
torch 2.12.0+cu130. Every quantity is read from the same 20 attempts; no selection rule
reads the target; no energy model, relaxation or filter; every attempt stays in every
denominator. Output `r3/v7_medium_run2_csp_after-geometry_reduced-formula.json`.

| quantity | count | rate | Wilson 95 % |
|---|---|---|---|
| match@1 (first attempt) | 3 / 50 | 0.06 | 0.021 to 0.162 |
| match, MAP selection (R2 likelihood) | 8 / 50 | 0.16 | 0.083 to 0.285 |
| match, consensus selection | 8 / 50 | 0.16 | 0.083 to 0.285 |
| match@20 (any attempt) | 13 / 50 | 0.26 | 0.159 to 0.396 |
| target cover table sampled at least once | 10 / 50 | 0.20 | 0.112 to 0.330 |
| match given the target's table (attempts) | 43 / 44 | 0.977 | 0.882 to 0.996 |

Match per attempt 0.081 (standard error over cases 0.026); valid fraction 0.994 (valid
= decoded and readable by the matcher, see the next section); mean number of distinct
cover tables among 20 attempts 16.48; one target whose three atom counts (record, cover,
matcher payload) disagree. Mean RMS of the matched selections, in units of the cube root
of the volume per atom: first attempt 0.102, MAP 0.0046, consensus 0.0154, best of 20
0.0505.

Reading. (i) Selection (the R2 question). Both target-blind rules select a match in 8
of the 13 cases that contain one (7 cases in common, one each alone). The paired
comparison with a uniformly chosen attempt, per case "selected attempt matches" minus
"fraction of the 20 attempts that match", has mean +0.079 with standard error 0.032
for the MAP rule (ratio 2.46) and +0.079 with standard error 0.032 for consensus
(ratio 2.51). The likelihood ranking therefore beats random choice by about 2.5
standard errors on 50 cases; the unpaired Wilson intervals of 0.06 and 0.16 still
overlap. The consensus rule, which needs no density, does as well. (ii) Geometry is
not the limit where it can be tested: when an attempt has the target's cover table, it
matches in 43 of 44 attempts. (iii) The selector is the limit: the target's table is
sampled in 10 of 50 cases, and 20 attempts spread over 16.5 distinct tables on average.
The table statistic is a lower bound on the selector's useful mass, because 38 of the
81 matching attempts use another table (a different description of the same
structure); 13 cases match and 10 sample the target's table. (iv) The numbers are a
small-corpus research measurement (404 training structures), not a benchmark value.

Cost: wall 9204 s. Sampling 5522 s (mean 110.4 s per case, median 108.5, maximum 235.3),
MAP likelihood 3562 s (mean 71.2, median 12.7, maximum 1544.8), consensus 103 s,
matching 16.5 s. Profile of one request of 8 attempts (`r3/profile_csp_sample_after-cover.log`):
6 topology buckets, 6 RK4 solves of 64 steps, 52.3 s of 59.9 s; the warm repeat costs
54.3 s. The sampler pays one solve per distinct cover table, and the selector produces
almost one table per attempt; that is the motivation of the fused sampler below.

A smoke of the harness (2 cases, 4 attempts, `after-cover` weights, both tracks,
`r3/v7_medium_run2_csp_smoke_after-cover.json`) preceded this run; it tests the
plumbing and carries no statistical content.

## Primitive-count track: encoder failure, diagnosis, harness schema v3 (2026-11-10)
The primitive-count track of the same measurement stopped at case index 16 with
`InvariantViolation: certified translation quotient did not form complete primitive
fibres`, raised by the matcher's encoder on a generated candidate. Reproduction:
`r3/diagnose_csp_encode.py`, outputs `r3/diagnose_csp_encode_case16.{json,log}`.

Case: record `8733cd61...6ea2`, composition H2 O2 Ni, target P1 with nu = 4 (20 atoms),
request seed 1209838612. Attempt 2 is an I4mm cover with nu = 8 and 20 primitive atoms
(orbits 8c O, 8d H, 8d H, 8d O, 8d Ni). Four of its hydrogen sites lie on the same
point (0, 0, 0.956105): pair distances 2.76e-8 and 3.90e-8 angstrom. One 8d orbit has
collapsed onto the 2a axis; its free parameter is about 2.6e-9 in fractional units.
The decoder accepts the motif, because its collision tolerance is 1e-8 angstrom
(`collision_atol_angstrom`) and its symmetry tolerance 1e-10. The matcher reduces to a
primitive cell at 0.01 angstrom, finds a same-species cluster of four sites where the
translation index is one, and reports that the cluster is not a translation fibre. The
matcher is right to refuse; the defect is in the candidate. None of the 20 attempts of
this case has the target's table, and 9 of the 20 have a pair closer than 0.5 angstrom
(closest pairs 0.401, 0.200, 0.000, 0.091, 0.490, 0.500, 0.140, 0.123, 0.393 angstrom
for attempts 0, 1, 2, 3, 7, 9, 13, 15, 19).

Consequence for the reported numbers: `valid_fraction` 0.994 says "decoded and readable
by the matcher". It is not the structural validity of the literature (shortest
interatomic distance above 0.5 angstrom, the CDVAE criterion). That statistic was not
measured by schema v2.

Harness schema v3 (`scripts/csp_evaluation_v7.py`, `tests/test_csp_evaluation_v7.py`,
9 tests; the certified matcher is unchanged):
- a decoded attempt that the encoder refuses is a miss with a cause:
  `evaluation/coincident-atoms` when two sites of one species lie within the matcher's
  0.01 angstrom reduction tolerance (checked with `periodic_pairs_within` after the
  refusal), `evaluation/encode` for a declared domain failure, and
  `evaluation/encode-invariant` with the kept message for any other invariant failure,
  so an encoder defect cannot hide behind the first label;
- every attempt records its shortest distance between two distinct atoms of the
  infinite crystal, images of one site included (`shortest_interatomic_distance`,
  tested against brute force on 20 random cells), and the summary reports
  `structurally_valid_fraction` at 0.5 angstrom next to `valid_fraction`, with
  `failed_stages` aggregated over the track.
A progress file of schema v2 cannot be resumed under v3 (header equality), so both
tracks are measured again under v3.

## Numerical finding: `torch.matrix_exp` for one matrix (2026-11-10, not fixed)
Found while testing the fused sampler against the bucket sampler on the legacy `ball-v1`
chart: a bucket of one crystal and the same crystal inside a larger batch gave lattice
metrics that differ by more than rounding. Check: `r3/matrix_exp_check.py`, output
`r3/matrix_exp_check.txt`, torch 2.12.0+cu130 on tcm-pc38, CPU and CUDA. The matrix is
a fixed symmetric 3 x 3 matrix scaled to a chosen 1-norm; the reference is the Taylor
series in float64 summed to 40 terms. The figures are relative 1-norm errors of the
first matrix of the call, from the CPU block of the output.

| 1-norm | float64, one matrix | float64, batch of two | float32, one matrix | float32, batch of two |
|---|---|---|---|---|
| 1e-3 | 2.031e-20 | 2.219e-16 | 4.050e-08 | 1.001e-07 |
| 1e-2 | 2.084e-14 | 2.203e-16 | 1.248e-08 | 2.238e-07 |
| 4e-2 | 2.078e-11 | 2.151e-16 | 2.939e-08 | 8.712e-08 |
| 4.9e-2 | 5.697e-11 | 4.296e-16 | 7.915e-08 | 1.258e-07 |
| 6e-2 | 2.141e-16 | 3.171e-16 | 4.802e-08 | 1.490e-07 |
| 0.5 | 1.656e-16 | 1.656e-16 | 4.699e-06 | 9.302e-08 |

A call with the shape (3, 3) and a call with the shape (1, 3, 3) give the same figures,
and a batch of three gives the figures of a batch of two. CUDA reproduces the four
anomalous one-matrix figures (float64 at 1e-2, 4e-2 and 4.9e-2, float32 at 0.5) to the
printed digits; entries at rounding level differ in the last place. In float64 the
one-matrix error grows as the
fifth power of the norm between 1e-2 and 4.9e-2 (ratios 997 and 2.74 against 4^5 = 1024
and 1.225^5 = 2.76) and returns to rounding at 6e-2. That is the signature of a
polynomial that is exact only to fourth order in that norm range. The cause inside
torch was not examined.

Scope in this package: `nn/bridge.py` calls `torch.matrix_exp` only in the `ball-v1`
chart (lines 491, 614, 1053), and the scientific manifest pins
`"matrix_exponential": "torch.matrix_exp"`. A `ball-v1` call that carries exactly one
crystal therefore carries a relative error in the exponential factor of its metric
that reaches 5.7e-11 in float64 on this grid, which is not small against
`symmetry_atol` = 1e-10. All R3 runs use the
`loewner-v2` chart, which does not call the matrix exponential, so no R3 number is
affected. The fused-sampler tests on `ball-v1` compare batches of at least two for this
reason. Not changed: replacing the pinned function changes the manifest of every
`ball-v1` checkpoint. Owner decision.

## Finding: the matcher fingerprint depends on the install path (2026-11-10, not fixed)
`diagnostics.matcher_callable_sha256` hashes `marshal.dumps(matcher.__code__)`. A
marshalled code object contains `co_filename`, so the value depends on the directory
the package is imported from. Evidence on tcm-pc38 with byte-identical
`physical_matching.py`, `geometry.py`, `periodic.py` and `lattice.py` (equal sha256):

| import path | `implementation_sha256` | `fingerprint` |
|---|---|---|
| `r3/v7_eval_20261110_code` | 016dc743ac450cbd... | 04725dfd90d26427... |
| `r3/v7_eval_20261110b_code` | 1f5dbbd2f3cfff0c... | 407b131aa9ee1ec9... |
| `repo3` | af54c7475ae1175d... | c9c3983a2d16ced3... |

The three matcher contracts are equal in every other field. Consequences: the
fingerprint proves "this file at this path", not "this matcher", so two machines or two
snapshots with the same matcher disagree, and a stored result cannot be bound to the
matcher across installs. It does detect every code change at a fixed path. The
comparison tool `r3/compare_csp_samplers.py` therefore compares the matcher contract
without these two fields and relies on the file hashes above. Not changed: a
path-independent rule (hash `co_code`, constants and names recursively, without
`co_filename` and line tables) changes every stored matcher fingerprint. Owner
decision.

## Fused geometry sampler `mixed-v1` (2026-11-10, opt-in; differing attempt explained 2026-11-12)
`LearnedCrystalGenerator(geometry_sampler="mixed-v1")`, default `bucket-v1`, harness flag
`--geometry-sampler`. All prepared attempts of a request are padded into one state and
integrated with one field read per RK4 stage, in chunks of ascending atom count; every
member draws its base point from its own generator as in the bucket path; a member
error is isolated by bisection and decided by the bucket path. Tests:
`tests/test_mixed_sampler.py` (local 127 passed, pc38 147 passed, in the combined
selection); fixture gaps to the bucket path 8.9e-16 (float64) and 4.8e-07 (float32).

Real checkpoint: run 2 `after-geometry`, reduced-formula, 50 cases of 20 attempts,
snapshot `r3/v7_eval_20261110b_code` (sha256 of the `LC_ALL=C` sorted per-file sha256
list of `src` and `scripts` without `__pycache__`:
`df84c8b93f5ac8a735762a0c1f6f4d48dbdf2ff0b39c02111eb379c8a497669e`), output
`r3/v7_medium_run2_csp3_after-geometry_reduced-formula_mixed-v1.json`, comparison
`r3/compare_csp_samplers.py`, replay `r3/repro_case_sampler_gap.{py,log}`.

- Every summary rate equals the bucket run: match@1 3/50, MAP 8/50, consensus 8/50,
  match@20 13/50, target table sampled 10/50, match given the target table 43/44.
  Match per attempt 0.082 against 0.081; decoded attempts 995 against 994.
- Two of 1000 attempts differ.
  - Record `8f6174bd`, attempt 3 (I-4, nu 2): latent gap 3.7e-08 in z, 0 in eta; the
    matcher says no for the bucket sample and yes for the fused one, RMS 0.4739 over
    4 sites. The matcher accepts a site only up to a normalized displacement 0.5, so
    this RMS requires every one of the four displacements to be at least 0.385. A
    decision boundary of the matcher lies between two latents 3.7e-08 apart; the site
    tolerance is the likely one. Which tolerance decides was not instrumented.
  - Record `1518540c`, attempt 10 (R-3m, nu 3, orbits 3b S, 9e Te, 6c Mg, 6c Mg):
    **not rounding.** Bucket eta (0.0, 0.6924124), z (-0.8383546, 2.0984399); fused eta
    (0.9694639, 0.6904806), z (-0.8404725, 2.1010380). The bucket sample fails decode
    with "parameter specializes Wyckoff row 6c below its multiplicity"; the fused one
    decodes (shortest distance 1.2466 angstrom). Each sampler reproduces its own value
    bit for bit on repeat. The other 19 attempts of the case agree within 2.4e-07 in z
    and 6.0e-08 in eta. The bucket value is exactly 0.0, the orbit-collapse pattern of
    the primitive-count case 16. Explained on 2026-11-12 by the step-by-step
    trajectories of both samplers (next section): both paths collapse onto the special
    position by step 18 and agree there to the float32 floor; they differ only in how
    rounding lets them leave an unstable equilibrium. It is a property of the learned
    field and of float32, not a defect of the fused path.
- Cost. Evaluation: sampling 5522.1 s to 947.0 s (mean 110.44 to 18.94 per case), wall
  9204 s to 3463 s; the bucket run shared the host with the A/B queue, so these two are
  not a clean comparison. Clean replay on the idle host: 86.81 and 85.54 s against 9.91
  and 9.85 s (record `1518540c`), 81.09 and 75.31 s against 9.99 and 9.97 s (record
  `8f6174bd`).
- Structural validity (schema v3, first measurement): 956 of 1000 attempts have their
  shortest distance above 0.5 angstrom (0.956, standard error over cases 0.0082); 995
  decode; 39 decoded attempts have a pair within 0.5 angstrom, smallest 0.0878; no
  matching attempt is among them.

Not done on 2026-11-10: primitive-count track under v3, profile of the remaining 10 to
30 s of fused sampling.

## Orbit collapse: measured mechanism (2026-11-12)
Diagnostic `r3/diagnose_orbit_collapse.py` (arithmetic-equal recording integrator, both
samplers from one loaded model, field scan along one intrinsic coordinate with all
other coordinates held on the recorded trajectory), analysis `r3/analyse_traj.py`,
dumps `r3/traj_1518540c.{bucket-v1,mixed-v1}.json`. Run 2 `after-geometry`, float32,
`ode_steps` 64, `ode_terminal_time` 1.0, snapshot `r3/v7_eval_20261110b_code`. The host
was shared with another user's job, so no timing is taken from these runs.

Case: record `1518540c`, request seed 1231800437, attempt 10: R-3m, nu 3, orbits 3b S,
9e Te, 6c Mg, 6c Mg. Coordinate 0 is the z parameter of the first 6c orbit, sites
(0, 0, z) and (0, 0, -z). At z = 0 the two sites coincide (the orbit specializes to 3a),
and z = 0 is also the wrap point 0 = 1 of the torus coordinate.

Facts.
- The base draws of all 20 attempts are bit-identical between the samplers (14 solves
  in the bucket path, 1 in the fused path). The other 19 attempts agree at every step
  within 2.4e-07 in z and 6.0e-08 in eta.
- Attempt 10 starts at z = 1.426356e-03. Both paths shrink it to 5.9e-05 at step 4,
  to 1e-06 at step 13 and to about 1e-08 at step 17, and hold it between 5e-09 and
  1.5e-08 until step 27. They agree to that floor.
- Bucket path: step 28 gives exactly 1.0 (a negative value of magnitude below 3e-08
  wrapped by `c - floor(c)` rounds to 1.0 in float32), step 29 gives exactly 0.0, and
  the coordinate stays 0.0 to the end. Its field at exactly 0.0 and 1.0 is exactly 0 at
  all 16 scanned steps, because the two coincident sites receive identical velocities.
  The decoder then reports the specialization.
- Fused path: step 28 gives 6.6e-10, then the value alternates between exactly 1.0 and
  exactly 0.0 until step 44. Its field at the special position is not exactly zero
  (magnitudes 4e-20 to 1.3e-18 at 10 of the 16 scanned steps). At step 45 the value is
  0.99999988, one float32 spacing below 1, and it then leaves: 0.99994, 0.99929, ...,
  0.96946 at step 64.
- The field along this coordinate, as a function of the offset d from the special
  position, is odd in d to float32 accuracy. It is linear for |d| up to 1e-05, bends
  between 1e-05 and 1e-04 and is saturated at 1e-03. Slope dv/dd at d = 1e-05 (scan
  on the bucket trajectory; the scan on the fused trajectory agrees to three digits):

  | t | 0 | 0.125 | 0.25 | 0.375 | 0.5 | 0.625 | 0.75 | 0.875 | 0.9375 | 1 |
  |---|---|---|---|---|---|---|---|---|---|---|
  | slope | -105 | -167 | -146 | -15 | +120 | +373 | +690 | +791 | +822 | +568 |

  Speed at |d| = 1e-03: 0.021, 0.027, 0.022, 0.0076 toward the special position for
  t = 0, 0.125, 0.25, 0.375; 0.0065, 0.032, 0.063, 0.067, 0.038 away from it for
  t = 0.5, 0.625, 0.75, 0.875, 1.
- At d = 1e-08 the scanned velocity has the wrong sign against the local slope at
  several steps (for example +6.3e-07 against an expected -1.7e-06 at t = 0.125): the
  float32 field does not resolve offsets below about 3e-08.

Reading.
1. Until t of about 0.4 the special position attracts: outside a layer of width about
   1e-04 the coordinate drifts toward it at 0.02 to 0.03 per unit time, so a base draw
   within about 0.01 of it arrives in finite time; inside the layer it decays at a rate
   of 100 to 170. After t of about 0.4 the same point repels, with rates up to 822.
2. The layer is the smoothing length of the vector carriers.
   `nn/invariant_transformer.py` normalizes every pair carrier as
   v / sqrt(|v|^2 + epsilon) with `normalization_epsilon` = 1e-4, and the direct
   carrier is about 2 pi r for a pair at separation r in units of `reference_length` =
   1 angstrom. Two atoms 0.01 angstrom apart therefore still exchange a carrier of
   nearly unit length; the carrier changes sign across the coincidence within about
   0.0016 angstrom. The field is close to a step function of the orbit parameter at
   the special position.
3. The integrator does not resolve this. With h = 1/64 the attracting phase has slope
   times h between -1.6 and -2.6, inside the real-axis stability limit 2.785 of
   classical RK4 (slope 178), which is why the decay is geometric and not divergent.
   In the repelling phase slope times h reaches 12.8, where the RK4 growth factor per
   step is the quartic Taylor polynomial and far from the exponential.
4. Once the coordinate sits on the special position to float32 accuracy, its later
   path is decided by rounding, not by the field. The wrap `c - floor(c)` maps a
   negative offset of magnitude below 3e-08 to exactly 1.0 and the next step maps 1.0
   to exactly 0.0, so an offset on that side is erased at every step and the growth of
   the repelling phase cannot accumulate. Neither final state is the flow of the
   learned field from the drawn base point. In exact arithmetic the offset would stay
   nonzero and leave after the sign change.
5. Second case, same mechanism: primitive-count case 16, record `8733cd61`, attempt 2
   (I4mm, nu 8), dumps `r3/traj_8733cd61_pc.{bucket-v1,mixed-v1}.json`. Coordinate 1
   is x of an 8d hydrogen orbit (x, 0, z); x = 0 is the 2a axis and the wrap point. The
   base draw is 0.998530, an offset of -1.47e-03. Slopes at d = 1e-05: -71, -83, -71,
   -36, -3.7 at t = 0, 0.125, 0.25, 0.375, 0.5; +78, +157, +203, +238, +216, +105 at
   t = 0.625, 0.6875, 0.75, 0.8125, 0.875, 0.9375; -27 at t = 1. The coordinate
   reaches exactly 0.0 at step 15, sits near 1.4e-08 for steps 17 to 39, alternates
   between exactly 1.0 and 0.0 from step 40, and ends at 2.6e-09 in the bucket path
   (decoded, closest pair 2.76e-08 angstrom) and at exactly 0.0 in the fused path
   (decode failure). Here the samplers' roles are exchanged, which confirms that
   rounding decides, not the sampler. The other 19 attempts of this case agree between
   the samplers; all other coordinates of attempt 2 agree within 1.5e-08.
6. The run's `geometry/decode` failures (6 of 1000 attempts under the bucket sampler,
   5 under the fused one) include the first case. The messages of the others are not
   stored in the result files, so their cause is not asserted here.

Reference calculation (not a measurement). For a uniform base on the circle, geodesic
conditional paths and a target symmetric under d -> -d with modes at +-d1, the exact
marginal field is v(d, t) = (m - d) / (1 - t), where m is the mean of the target
inside the window of half-width (1 - t) / 2 around d (given d1, the point d_t is
uniform on that window, because wrap(d1 - d0) is uniform and independent of d1). Near
the midpoint both modes lie in the window until t = 1 - 2 d1, so m = 0 and
v = -d / (1 - t): a base offset d0 contracts to d0 (1 - t), reaches about 2 d1 d0 and
never zero, and then departs to its own mode. The symmetry assumption holds for the
learned field by construction: exchanging d and -d permutes the two sites, and the
field is permutation equivariant, so it is odd in d. Attraction to the special position
at early times is therefore a property of the flow-matching target on a symmetric orbit
coordinate, with rate 1 / (1 - t), which is 1 at t = 0 and 1 / (2 d1) when the modes
separate. What the learned field adds is the kink: a rate of 70 to 170 inside a layer
of 1e-04 and a nearly constant drift of 0.01 to 0.03 outside it, where the exact field
is proportional to d. That turns a bounded contraction into a collapse to the rounding
floor.

Consequences, none acted on yet.
- Sampler comparison: the two samplers agree to rounding wherever the trajectory is
  well conditioned. `mixed-v1` stays opt-in until the owner decides otherwise.
- Likelihood selection: the fused run selected this attempt as the MAP candidate of
  its cover class with `log_geometry` 20.21 against 2.99 and 2.78 for the other two
  members. A density integrated through a collapse is not a density of the sampled
  point. A target-blind reliability check of the backward integration is needed before
  the MAP rule can be trusted on such attempts.
- Model: `normalization_epsilon` sets a 0.0016 angstrom smoothing length. A length of
  the order of 0.1 angstrom would bound the field's slope near a coincidence by the
  integrator's limit. This changes the model and needs a registered A/B.

## R3 CSP measurement: run 2 `after-geometry`, primitive-count track (2026-11-12)
Same protocol and checkpoint as the reduced-formula measurement, harness schema v3,
fused sampler `mixed-v1`, snapshot `r3/v7_eval_20261110b_code` (code fingerprint
`a03cc984...590fb0`), 50 test compositions, 20 attempts, the request carries the
target's primitive atom count. Output
`r3/v7_medium_run2_csp3_after-geometry_primitive-count_mixed-v1.json`. The host was
shared with another user's job; no timing is taken from this run.

| quantity | count | rate | Wilson 95 % |
|---|---|---|---|
| match@1 (first attempt) | 6 / 50 | 0.12 | 0.056 to 0.238 |
| match, MAP selection | 10 / 50 | 0.20 | 0.112 to 0.330 |
| match, consensus selection | 10 / 50 | 0.20 | 0.112 to 0.330 |
| match@20 (any attempt) | 17 / 50 | 0.34 | 0.224 to 0.478 |
| target cover table sampled at least once | 16 / 50 | 0.32 | 0.208 to 0.458 |
| match given the target's table (attempts) | 60 / 67 | 0.896 | 0.800 to 0.948 |

Match per attempt 0.139 (standard error over cases 0.040). 993 of 1000 attempts
decode (7 `geometry/decode` failures, no encoder refusal); 959 have their shortest
distance above 0.5 angstrom (structural validity 0.959, standard error over cases
0.012); 34 decoded attempts have a pair within 0.5 angstrom, smallest 0.0914. No
matching attempt and no selected attempt is structurally invalid. Mean number of
distinct tables 14.0. 139 attempts match, 60 of them on the target's table. Mean RMS of
the matched selections: first 0.115, MAP 0.055, consensus 0.077, best of 20 0.090.

Reading. The atom-count condition raises every rate over the reduced-formula track
(match@20 13 to 17 of 50, target table sampled 10 to 16 of 50), as expected from a
smaller legal action set; the intervals of the two tracks overlap. Selection: MAP and
consensus each pick a match in 10 of the 17 cases that contain one (9 in common). The
paired gain over a uniformly chosen attempt is +0.061 for MAP (standard error 0.028,
ratio 2.21) and +0.061 for consensus (0.030, ratio 2.01); the first attempt gives
-0.019 (0.020). Geometry given the target's table is lower here (60 of 67) than on the
reduced-formula track (43 of 44); the two intervals overlap. The selector remains the
limit: the target's table is absent from 34 of 50 cases. The I4mm attempt that stopped
the schema-v2 run is a `geometry/decode` failure under the fused sampler (exact
collapse) and a decoded coincident structure under the bucket sampler; see "Orbit
collapse".

## Pair-carrier smoothing length (opt-in, 2026-11-12)
`ModelConfig.geometry_pair_carrier_smoothing_length = s` (runner option
`carrier-smoothing`, default value 0.2 angstrom, `carrier-smoothing=VALUE` for another)
multiplies every direct and reciprocal pair carrier of the dense backbone by

    sqrt((|m_1|^2 + epsilon) / (|m_1|^2 + (2 pi s)^2)),   m_1 = sin(2 pi delta) L / l_0,

where |m_1| = 2 pi rho and rho is the smooth periodic separation of the pair (rho = r
for r far below the cell, rho <= r always). The first direct carrier becomes exactly
m_1 / sqrt(|m_1|^2 + (2 pi s)^2), that is rho / sqrt(rho^2 + s^2) times its direction;
the cross-product carriers carry the factor squared; the lattice-frame carriers are
not pair carriers and are unchanged. The factor is a function of |m_1|^2 only, has no
parameter, is smooth at a coincidence (its argument is at least epsilon / (2 pi s)^2),
and keeps SO(3) and permutation equivariance. `None` (the default) keeps the legacy
carriers bit for bit, the state-dict keys and the configuration fingerprint.
`(2 pi s)^2 > epsilon` and s > 0 are required. Tests: `tests/test_pair_carrier_smoothing.py`
(19 tests), `tests/test_research_options_v7.py` (19 tests). Local regression selection
after the change (fused sampler, padded expansion and batching, member fallback, CSP
harness, runner options, configuration, backbone, radial features, periodic elements):
155 passed in 186 s.

Why this form. The exact marginal field is continuous at a pair coincidence: for two
identical atoms the relative velocity is odd in the separation and linear at zero (the
reference calculation above), and for two different atoms the direction of an
infinitesimal separation carries no information. A unit pair carrier with a nonzero
coefficient is a step at the coincidence. The legacy floor smooths that step over
0.0016 angstrom; s smooths it over a length below every physical distance and far
above the rounding and integrator scales. With a drift of 0.02 to 0.2 per unit time in
fractional units, as measured, the slope bound drift / (s / 2c) is 2 to 16 for
c = 8 angstrom, against the RK4 limit 178 at 64 steps.

Known property, not a defect: rho is at most |L| / 2 pi along an axis, so in a cell
shorter than about 2 pi s = 1.26 angstrom every pair carrier is damped. Such cells
occur only early in the flow from the lattice base, where every pair is closer than
any physical distance.

## Registered A/B: pair-carrier smoothing on the geometry field (registered 2026-11-12, before any treatment run)
Arms: control `--options default`, treatment `--options carrier-smoothing` (s = 0.2).
Everything else identical: `research-medium`, domain v7, composition split, mixed
batches of 16, `--epochs 1,1,250`, `--eval-every 1,1,25`, `--validation-draws 4`,
`--select-best`, `--ema-decay 0.999`, both learning rates 2e-4, `--skip-csp`,
`--cpu-threads 2`; seeds 20261030, 20261031, 20261032. The geometry phase is the
measurement; one composition epoch and one cover epoch only satisfy the runner. 250
geometry epochs contain run 2's minimum (EMA 9.523 at epoch 150; flat from 100 to 200).
Code: a new frozen snapshot, recorded with its hash when it is taken.

Evaluation: `scripts/diagnose_field_replay_v7.py`. The covers are drawn by the selector
of run 2 `after-geometry` under the stored protocol (seed 20261530, 50 test records, 20
attempts, reduced-formula track); each arm's selected `after-geometry` field integrates
the same covers from the same base draws (fused sampler, float32). The likelihood rule
is not evaluated. A second measurement integrates 20 draws of each record's own cover
(oracle cover). The hybrid of two checkpoints is a paired diagnostic, not a model.

Counts per arm and seed over the 1000 replayed attempts: N (no structure: ODE or
decode failure), S01 (decoded, shortest distance at most 0.01 angstrom), S5 (decoded,
at most 0.5). Collapse count C = N + S01; invalid count I = N + S5. Reference, run 2
under the fused sampler: N = 5, S01 = 0, S5 = 39, so C = 5 and I = 44.

- P1 (mechanism, primary): C summed over the three seeds is at most one quarter of the
  control's sum.
- P2 (validity): I is lower in the treatment; the per-record difference treatment
  minus control, averaged over the seeds, has a negative mean with magnitude above two
  standard errors over the 50 records.
- P3 (no harm): the per-record difference of the oracle-cover match fraction (20
  draws), averaged over the seeds, is not below minus two standard errors; the same
  for the replayed match fraction per attempt.
- P4 (field slope): for the two diagnosed coordinates (record `1518540c` attempt 10
  coordinate 0; primitive-count record `8733cd61` attempt 2 coordinate 1), scanned as
  above with the treatment field of seed 20261030 on its own trajectory, the largest
  |slope| at d = 1e-05 over the scanned steps is below 50. Control references: 822
  and 238.

Falsifiers and rule. P1 fails if the treatment keeps more than a quarter of the
collapses; P2 fails if the interval of the paired difference contains zero. The option
enters the research configuration of run 3 only if P1 and P2 hold and P3 shows no
harm. Futility stop: if at seed 20261030 the treatment has neither a lower C nor a
lower I than the control, the other seeds are not run and the verdict is "not
supported". Consistency check: the control at seed 20261030 is expected to reproduce
run 2's validation CFM at geometry epochs 25, 50, ... (10.349 raw at 25; 9.523 EMA at
150) if the geometry phase does not depend on the length of the earlier phases; then
its replay reproduces the stored fused CSP result attempt by attempt. Limits: one
value of s, 404 training structures, three seeds, covers of one selector, 50 records.

## Registered A/B: selector inputs and regularisation (registered 2026-11-12, before any treatment run)
Three single-lever treatments of the cover selector, each against the control arms of
the periodic A/B, which are reused and not rerun (`r3/ab_periodic/control_s20261030`,
`..31`, `..32`; mean selected validation cover NLL 7.328107):

- `selector-symmetry`: operation-derived inputs. Per space group the rotation-type
  histogram of the point group, log(1 + |PG|) / log 49, a centring one-hot, a polar
  flag and a chiral flag; per Wyckoff row the rotation-type histogram of its site
  stabilizer.
- `selector-rows`: per Wyckoff row the exact chart of its normalizer-canonical
  representative (offset, coordinate matrix) and two scaled row indices.
- `dropout`: dropout 0.1 in the composition and selector Transformers (off in every
  evaluation).

Protocol identical to the periodic A/B except `--options`: `research-medium`, domain
v7, composition split, mixed batches of 16, `--epochs 1,60,1`, `--eval-every 1`,
`--validation-draws 1`, `--select-best`, `--ema-decay 0.999`, learning rates 2e-4,
`--skip-csp`; seeds 20261030, 20261031, 20261032; frozen snapshot
`r3/v7_eval_20261110_code` (hash `d8e24f70...f885`, computed over the `*.py` files
with the locale-dependent `sort` of en_US.UTF-8), which already declares the three
options. After each run the factor diagnostic on `after-cover`, validation and test.
Queue `r3/ab_selector/queue.sh`, analysis `r3/ab_selector/analysis.py`.

For each lever separately:
- P1: the best validation cover NLL, averaged over the three seeds, is at least 0.2
  nats lower than the control's.
- P2: test cover NLL of the validation-selected checkpoint; the per-record difference
  treatment minus control, averaged over the seeds, has a negative mean with
  magnitude above two standard errors over the 50 test records.
- P3 (mechanism): `selector-symmetry` gains in the space-group and row factors;
  `selector-rows` gains in the row factor and not in the nu or space-group factor;
  `dropout` selects a later epoch than the control (9, 10, 12) and has no predicted
  factor.

Rule: a lever enters the research configuration of run 3 only if its P1 and P2 both
hold. The size 0.2 is just below the periodic prior's measured 0.268, the smallest
effect resolved at two standard errors so far (standard error about 0.10 on 50
records). Three levers are tested: under no effect, the chance that at least one P2
holds by accident is about 7 %. A combined arm is registered separately after these
verdicts; additivity with the periodic prior is not assumed. Limits as before: three
seeds, 50 records per held-out partition, selector only.

## Carrier A/B: launch record (2026-11-12, before any treatment result)
Snapshot `r3/v7_eval_20261112_code`, 100 files, sha256 of the `LC_ALL=C` sorted
per-file sha256 list of `src` and `scripts` without `__pycache__` and `.DS_Store`:
`266fca5ad86ea362707eea170c06eee5475746181e46ff1870ff0e90f1853740` (equal on the Mac
and on the host). It differs from `v7_eval_20261110b_code` in six files (`config.py`,
`learned.py`, `nn/field.py`, `nn/invariant_transformer.py`,
`scripts/evaluate_csp_research_v7.py`, `scripts/train_research_v7.py`) and adds
`scripts/diagnose_field_replay_v7.py`. Queue `r3/ab_carrier/queue.sh`, started
05:42:43 UTC, sequential, one GPU job at a time (the host is shared).

Plumbing check, not a treatment run (`r3/ab_carrier/smoke_run2.json`): run 2 replayed
through its own field for the first three test records reproduces the stored fused
result in every compared field, including the 60 shortest distances to all digits
(`check_smoke.py`: 0 mismatching fields). The default path of the new snapshot is
therefore unchanged in sampling on the host.

Consistency check, first point: the control at seed 20261030 gives training CFM
14.19310159966497, 12.356331060428431, 12.355257336455997 in its first three geometry
epochs and validation CFM 10.348635123497807 (raw) and 11.593114121493416 (EMA) at
geometry epoch 25, identical to run 2 in every printed digit. The geometry phase does
not depend on the length of the earlier phases, and the default path of the new
snapshot is unchanged in training.

The selector queue `r3/ab_selector/queue.sh` was started at 06:08:08 UTC. It launches a
run only with 5 GB of GPU memory free, and 9 GB while the carrier queue is active,
because another user's job holds about 9 GB of the 16 GB card and a geometry
evaluation holds 2.7 GB.

## Interruption of both A/B queues (2026-11-12)
The owner stopped the jobs on tcm-pc38 because the GPU cannot be used for now.
- 07:06:26 UTC: the replay of `control_s20261030` was terminated; the carrier queue
  exited. No treatment run of the carrier A/B was started.
- Defect of the queue design: the selector queue treated the carrier queue's
  "FAILED" line as "carrier finished" and started on its own. It trained and
  diagnosed `selector-symmetry_s20261030` (07:58 and 08:04 UTC) and trained
  `selector-symmetry_s20261031` (08:56 UTC); its diagnostic was terminated at 09:01 UTC
  and the queue exited. A queue must not start work after another job was stopped by
  hand; any relaunch needs an explicit go-ahead and a stop file check.
- State on disk under `r3/`: `ab_carrier/control_s20261030` complete (20 validation
  points equal to run 2, largest difference 0.0; selected EMA epoch 150, 9.523490);
  `ab_selector/selector-symmetry_s20261030` trained and diagnosed;
  `ab_selector/selector-symmetry_s20261031` trained, not diagnosed. Nothing was
  analysed: both registered analyses need three seeds, so no verdict and no interim
  reading is recorded. At 09:11 UTC no process of this project was running on the host.

## Move to tcm-pc10 and cross-host check (2026-11-14)
tcm-pc38 stayed occupied by another user's jobs (GPU at 100 % on 2026-11-13 and
2026-11-14), and its driver changed to 615.71.09 after a reboot. The owner moved the
work to tcm-pc10: same GPU model (RTX 5060 Ti 16 GB), driver 580.196.07 (the version
pc38 had for every earlier run), torch 2.12.0+cu130 from the shared venv, 12 CPUs, the
same NFS home, so the snapshots, the store and all earlier results are the same files.
pc10 is shared too; every queue waits for 6 GB of free GPU memory before a launch and
exits when its `STOP` file exists. Scripts: `diagnostics/pc10/` (on the host
`r3/pc10/`), outputs `r3/hostcheck_pc10/`.

Cross-host check, started 02:29:30 UTC, compared with `compare_hosts.py`:
- a) sampling: run 2 replayed through its own field for three test records on pc10
  equals the pc38 replay (`r3/ab_carrier/smoke_run2.json`) and the stored fused CSP
  result in every compared item, including all shortest distances and the oracle
  draws: 0 differing items.
- d) evaluation: the cover-factor diagnostic of the pc38-trained
  `selector-symmetry_s20261030` computed on pc10 equals the pc38 JSON in every item
  (records, summary, ranks, checkpoint sha256): 0 differing items.
- b) geometry training (carrier control, seed 20261030) and c) cover training
  (selector control, seed 20261030): reruns on pc10, to be compared with the pc38 runs
  including every checkpoint tensor. Recorded below when they finish.

Both queues were restarted on pc10 at 02:34 UTC (`r3/ab_carrier/queue_pc10.log`,
`r3/ab_selector/queue_pc10.log`). They skip finished runs; a "trained" or "diagnosed"
line stamped 02:34:01 or 02:34:02 marks a result that already existed from pc38, not a
run on pc10. Evaluation-type jobs (replay, diagnostic) are covered by checks a and d.
Treatment runs trained on pc10 are compared with controls trained on pc38 only if
checks b and c show identical training; otherwise the controls are rerun on pc10.

## Independent review and amendment of the carrier A/B (2026-11-15, before any treatment result was read)
A multi-agent review (2026-11-14; three reviewers, each top finding checked by an
adversarial verifier) read the carrier option, the A/B scripts and the quantitative
claims of this log. Journal: session workflow `wf_a35ac87d-42c`. No treatment result of
the carrier A/B or of the selector A/B had been read when this was written.

Confirmed, affects the interpretation of the registered carrier A/B:
1. Scope of the switch. `m_1 = sin(2 pi delta) L / l_0` vanishes at all eight `delta` in
   `{0, 1/2}^3`, so rho is the distance to the nearest half-lattice translation, not the
   pair separation. The switch also makes every pair carrier vanish linearly within
   about s of the seven non-zero half-lattice separations, in a cell of any size. In a
   4 angstrom cubic cell a pair at `delta = (0.48, 0, 0)` (r = 1.92 angstrom) keeps a
   first direct carrier of 0.371 instead of 1.0. In the 512 local MP-20 pilot structures
   (spglib conventional cells, s = 0.2), 10.8 % of all atom pairs have r above 1
   angstrom and a switch below 0.9, 8.8 % a switch below 0.5, and 92 % of structures
   contain such a pair (verifier's local measurement, not repeated here). The legacy
   carriers already vanish exactly at those points and change sign over
   sqrt(epsilon) / 2 pi there, so the switch widens an existing zero; it creates none.
   The "Known property" paragraph above, which names only cells shorter than 1.26
   angstrom, is incomplete.
   Amendment. The registered option and the registered predictions are unchanged, and
   the A/B measures the option as implemented. P1 (collapse counts) and P4 (slopes at the
   two diagnosed coincidences) still test the coincidence mechanism. P2 and P3 measure
   the whole option: a P2 gain or a P3 harm is not attributable to the coincidence alone.
   Registered contingency: if P1 holds and P3 shows harm, the next test is a
   coincidence-only switch with `q = |m_1|^2 + sum_c G_cc (1 - cos 2 pi delta_c)^2` in
   place of `|m_1|^2` (G = L L^T / l_0^2). q vanishes only at integer delta, is smooth,
   periodic and even, and equals `|m_1|^2 + O(|delta|^4)` near a coincidence. It is not
   implemented.
2. Futility gate. The queue treated every nonzero exit of `futility.py` as the
   registered stop, so a crash (truncated replay JSON, missing file) would have been
   logged as "futility stop" with exit 0. Fixed in the repository copies
   (`diagnostics/ab_carrier/futility.py`, `queue.sh`, `diagnostics/pc10/carrier_queue.sh`):
   exit 3 is the decision, any other nonzero status stops the queue as a failure. Not
   yet deployed on pc10 (see below). Whether the running pc10 queue reached this gate is
   not known.

Corrections of statements in this log (no measured number and no verdict changes):
- "rho <= r always" holds in orthogonal cells only. In a sheared basis rho can exceed
  r: rows (4,0,0), (12,4,0), (0,0,4), delta = (0.3, -0.1, 0) gives r = 0.400, rho =
  0.638 angstrom. In every basis rho = r (1 + O(|delta|^2)) near a coincidence.
- Reference calculation, the parenthesis: given the target point x, d_t is uniform on
  the window of half-width (1 - t) / 2 around x. The window around d is the posterior
  support of the target given d_t = d, by Bayes with the constant likelihood
  1 / (1 - t). The formula and every number derived from it stand. The exact minimum of
  the contracted offset is 2 d1 d0 / (1 - 2 |d0|), "about 2 d1 d0" for small d0.
- 0.99999988 is 1 - 2^-23, two float32 spacings below 1 (the spacing below 1 is
  2^-24), not one.
- The slope bound "2 to 16" is 1.6 to 16.
- Primitive-count table: the Wilson upper endpoints of 17/50 and 60/67 are 0.478 and
  0.948 (double rounding); corrected in the table above.
- "The atom-count condition raises every rate": true for the per-case rates and match
  per attempt, not for the conditional rate "match given the target's table" (0.977 to
  0.896), as the same paragraph says.
- The node chirality is computed from the switched carriers; s is in units of
  `reference_length_angstrom` (1 angstrom in both registered domains).

Fixed in code, behaviour unchanged: the docstrings of the option
(`nn/invariant_transformer.py`, `config.py`) state the half-lattice scope, the basis
caveat and the units. Tests added to `tests/test_pair_carrier_smoothing.py`: the switch
at and near the half-lattice separations, and the forwarding of the option from
`ModelConfig` through the field to the transformer (21 tests, 49 with the option and
configuration tests, all pass). `diagnostics/ab_carrier/analysis.py` now refuses a
verdict on fewer than the three registered seeds (`--partial` prints a reading marked
"not the registered verdict"), asserts each arm's options and run directory, and warns
on request-level failures, which count toward N for all 20 attempts of a case; checked
on synthetic inputs. These edits change the local source hash; the frozen snapshot
`r3/v7_eval_20261112_code` used by the running experiment is not modified.

Recorded, not changed (no effect on the pairing of the two arms): the replay hybrid
samples covers with the cover run's model left in training mode, while run 2's own
generator samples in eval mode (run 2 has dropout 0; the three-record replays on pc38
and pc10 equal the stored fused result exactly); the oracle labels every encoder
invariant failure `evaluation/encode`; `hostcheck.sh` launches four jobs without the
GPU-room check; `gpu_room` is a point-in-time check shared by two queues.

State of the hosts. Since 2026-11-14 02:34 UTC the pc10 queues have run unobserved.
On 2026-11-15 the SSH connection to the TCM gateway asked for interactive
authentication, which this assistant does not perform, so neither the queues nor the
cross-host training checks b and c (`r3/hostcheck_pc10/`) have been read.

## Results of the carrier and selector A/Bs (read 2026-11-15)
Read on 2026-11-15 through the TCM gateway, which mounts the same NFS home; no login to
tcm-pc10 was needed or made. All pc10 jobs had finished on 2026-11-14: host-check jobs
03:22:33 UTC, selector queue 05:58:12 UTC, carrier queue 08:14:58 UTC ("queue done" in
both queue logs, no `STOP` file). The result files were copied to the Mac and analysed
there (JSON and six checkpoint files, CPU, seconds); the printed analysis outputs are
`diagnostics/ab_carrier/verdict_2026-11-15.txt` and
`diagnostics/ab_selector/verdict_2026-11-15.txt`. The verdicts below were re-derived
independently from the raw JSON and checked by adversarial reviewers (session workflow
`wf_1e01f52a-767`, eight agents); every number reproduced and no verdict was refuted.

### Validity of the comparisons
- Cross-host checks b and c (registered condition above). b) Carrier control, seed
  20261030, pc10 rerun against the pc38 run: 20 of 20 validation points identical, all
  four phase evaluations identical, `after-geometry.pt` 1115 of 1115 tensors equal.
  c) Selector control, seed 20261030: 120 of 120 validation points identical,
  evaluations identical, `after-geometry.pt` 1115 of 1115 and `after-cover.pt` 809 of 809
  tensors equal. The checkpoint files differ only in `provenance/created_utc` and, for c,
  in `provenance/platform` (kernel 7.0.0-42 against 7.0.0-38); the training contracts
  differ only in the output path. Runs trained on the two hosts are therefore compared
  as registered. Host of each run: carrier `control_s20261030` pc38, the other five
  carrier runs pc10; selector controls pc38 (2026-11-10), `selector-symmetry` seeds
  20261030 and 20261031 trained on pc38, the other seven selector runs pc10. The checks
  cover the default path only; the lever-specific paths were not compared across hosts.
  The carrier outcome is the same on the pc10-only seeds 20261031 and 20261032 (C 7 to 2;
  I 83 to 97, per-record difference +0.140, standard error 0.069). Every selector P1
  misses its threshold by at least 0.17 nats, far above any rounding effect.
- Code. The replay code fingerprint `102a0b14...f553` (73 files) equals the fingerprint
  recomputed over the frozen snapshot `r3/v7_eval_20261112_code`, whose registered hash
  `266fca5a...3740` (100 files) reproduces. The selector snapshot hash `d8e24f70...f885`
  reproduces with its original rule (`find src scripts -type f -name "*.py" | sort |
  xargs sha256sum | sha256sum`) under an English UTF-8 collation; the gateway has
  `en_GB.utf8` and no `en_US`, and `C` gives a different hash. The scripts in `r3/pc10/`
  equal the repository copies except `carrier_queue.sh`, whose futility handling was
  fixed in the repository after launch.
- Futility gate. The pc10 queue ran the older `futility.py` (stop exit 1). Its decision
  expression equals the fixed copy's. It logged `control C, I: 5 44 | treatment C, I: 2
  47` and continued; that is the registered decision under either version, and the
  crash-as-stop defect did not occur.
- Consistency check. `control_s20261030` equals run 2 at all 20 shared validation points
  (largest difference 0.0; 10.348635 raw at 25, 9.523490 EMA at 150), and its replay equals
  run 2's stored fused CSP result in all 50 cases and all 1000 attempts, in every
  compared field including the shortest distances.

### Carrier A/B verdict: not supported
| seed | arm | N | S01 | S5 | C | I | replay match per attempt | oracle match per draw |
|---|---|---|---|---|---|---|---|---|
| 20261030 | control | 5 | 0 | 39 | 5 | 44 | 0.082 | 0.443 |
| 20261030 | smooth | 1 | 1 | 46 | 2 | 47 | 0.082 | 0.444 |
| 20261031 | control | 3 | 0 | 46 | 3 | 49 | 0.084 | 0.432 |
| 20261031 | smooth | 0 | 2 | 53 | 2 | 53 | 0.086 | 0.437 |
| 20261032 | control | 4 | 0 | 30 | 4 | 34 | 0.078 | 0.434 |
| 20261032 | smooth | 0 | 0 | 44 | 0 | 44 | 0.080 | 0.435 |

- P1 fails. C summed: control 12, treatment 4; a quarter of the control is 3. The
  treatment keeps one third of the collapses.
- P2 fails in the harmful direction. I summed: 127 against 144, higher in the treatment
  at every seed. Per-record difference +0.1133 invalid attempts of 20, standard error
  0.0535, ratio +2.12; the two-standard-error interval [+0.006, +0.220] excludes zero.
- P3 holds (no harm). Oracle match fraction +0.0023 (standard error 0.0033); replayed
  match fraction +0.0013 (0.0009). match@20 13, 14, 13 and match@1 3 in both arms.
- P4 holds. Largest |slope| at d = 1e-05 on the treatment's own trajectory: 11.16
  (`1518540c`, t 0.8125) and 3.41 (`8733cd61`, t 0.8125), against 822.1 and 238.3 in the
  control.
- Rule: the option does not enter the research configuration of run 3. The registered
  contingency (P1 holds and P3 shows harm) did not occur, so the coincidence-only switch
  is not triggered.

Attempt-level description. The two arms integrate the same covers from the same base
draws, so attempt k of record r is paired exactly across arms (verified from the stored
seeds; within a seed the correlation of log d between arms is 0.91 to 0.93, against
0.23 to 0.26 with the attempt index shifted by one). The counts are descriptive; the same
attempt recurs across seeds, so pooled counts are not independent.
- None of the 12 control collapses is a collapse in the treatment: 3 become valid, 4
  end between 0.01 and 0.1 angstrom, 5 between 0.1 and 0.5. Four new collapses occur in
  attempts whose control distance was 0.055 to 0.238 angstrom.
- 27 attempts go from valid to invalid and 10 the other way. The change of I by control
  category: -3 from N, -7 from (0.1, 0.5] (rising above 0.5), +26 from (0.5, 1], +1
  from above 1 angstrom. The treatment distances of the 27 have median 0.365 angstrom.
  No flip gains or loses a match, except one oracle draw that becomes valid and matches.
- Short contacts get shorter, long distances do not change. For attempts with
  sqrt(d_control d_treatment) in (0.1, 0.3] angstrom the median of
  log(d_treatment / d_control) is -0.361 (35 of 46 shorter; per seed -0.370, -0.271,
  -0.584). Above 0.75 angstrom the pooled medians are within 0.0006 of zero. The same
  band compared between control seeds gives -0.011. The oracle draws give -0.249 in the
  band, but between seeds that band varies by 0.22, so the oracle does not separate the
  effect.
- One record, `9be20999` (P4_2nm), gives 6 of the 27 valid-to-invalid flips, in every
  seed. Without it the per-record difference is +0.0748 (0.0380).
- Seed 20261032 contributes +10 of the +17. Its arms selected different geometry epochs
  (treatment phase epoch 100, control 125; EMA validation 10.315523 against 10.312496).
  The epoch is an outcome of the registered selection, not a pre-treatment factor, and
  the shortening of short contacts appears at every seed.
- Oracle track (not registered): C 11 to 1, I 70 to 85.
- Arithmetic bound: had every control collapse become valid with nothing else changed,
  the P2 difference would have been -0.080 (standard error 0.036, ratio -2.20). With 50
  records, removing collapses alone could at best just pass P2.
- Scans (bucket sampler, seed 20261030). `1518540c` attempt 10 escapes in the treatment
  (final z 0.0422, shortest distance 1.673 angstrom) where the control collapses exactly.
  `8733cd61` attempt 2 ends at an offset of 2.46e-3 and decodes at 0.0258 angstrom, still
  invalid. Field at an offset of 1e-3, t = 0.8125: 0.0654 to 0.0139 and 0.0409 to 0.0042.
  Neither scanned attempt is among the replay's counted collapses (the first collapses
  only under the bucket sampler, the second is a primitive-count attempt), so no measured
  link connects P4 to the changes of C and I.

Reading (inference, not demonstrated). The switch removed the step at a coincidence as
designed: P4, and no control collapse survives. With s = 0.2 angstrom it also smooths the
field over the range of distances that decides validity, and the shortened distances lie
in exactly that range (below about 0.5 angstrom, unchanged above 0.75). One consistent
reading: once the repelling phase starts, the legacy unit carrier pushes a near pair
apart at a speed independent of its separation, while the switched carrier's push falls
linearly below s. A reduced one-coordinate model built from the scans supports this on
`8733cd61` and gives the opposite sign on `1518540c`, so it is not established. Two
other readings are open: the half-lattice part of the switch (along one axis of an
orthogonal cell, a pair whose carriers are damped below 0.9 near a half-lattice
separation is at least 0.649 angstrom apart when the axis is at least 2.595 angstrom, so
it can act only through the field on other pairs), and training divergence between
separately trained arms. The stored data hold one scalar per attempt, the shortest
distance. They do not say which pair is closest, whether it is a pair of symmetry mates
of one orbit, or whether it sits near a half-lattice separation.

Next measurements (proposed, not run).
- M1, evaluation only, on the six existing fields: re-run the replay and oracle for the
  74 replay and 48 oracle (record, attempt) pairs that are N or at most 0.5 angstrom in
  any file, and record per attempt the closest pair (atoms, species, orbits, whether they
  are mates of one orbit, delta, rho to the nearest lattice vector and to the nearest
  non-zero half-lattice translation) and, for every free coordinate, its smallest
  distance from a special value along the path and the time of that minimum. This
  decides between the readings above, before any new treatment is registered.
- M0, no model: the rate of contacts below 0.5 angstrom in the training target itself
  under the dequantization kernel, from the test records' own covers. A model that
  reproduced the dequantized target exactly would still produce these contacts. The
  kernel and its widths must first be read from the training contract (inferred so far:
  `fractional-v1`, intrinsic sigma 0.03).
- Not proposed: a length that shrinks with t. On `1518540c` s = 0.2 already matches the
  exact reference growth (log growth 4.66 against 5.01), and the drafted schedule would
  raise the late slopes to 172 to 188, at the RK4 limit 178 the option was meant to stay
  below.

### Selector A/B verdict: no lever enters run 3
Control mean selected validation cover NLL 7.328107 (selected epochs 9, 10, 12).

| lever | P1: mean difference (per seed) | P2: test difference, SE, ratio | P3 |
|---|---|---|---|
| `selector-symmetry` | -0.0212 (-0.143, -0.028, +0.107) | -0.1632, 0.0922, -1.77 | not supported |
| `selector-rows` | +0.0068 (-0.047, -0.039, +0.107) | -0.1063, 0.0865, -1.23 | falsified |
| `dropout` | +0.0030 (+0.005, +0.017, -0.012) | +0.0305, 0.0180, +1.69 | not supported |

- P1 fails for all three levers (required: at most -0.2). P2 fails for all three.
- P3. `selector-symmetry`: no partition gains in both the space-group and the row
  factor (test: group +0.0113, row -0.1447 with ratio -1.71; validation: group -0.0405
  with ratio -2.67, row +0.0235). `selector-rows`: no row gain (test -0.0511, ratio -0.59;
  validation +0.0630) but a gain in nu, which the prediction excluded (test -0.0617,
  ratio -3.47; validation -0.0695, ratio -3.39). `dropout`: selected epochs 9, 8, 15
  against 9, 10, 12, later at one seed of three; its test space-group factor is worse
  (+0.0130, ratio +2.42).
- Test ranks are unchanged in every arm: space-group top-1 0.413; top-5 0.587 to 0.600
  against 0.593.
- Every selected checkpoint is the unique minimum of its 120-point history, and each
  diagnostic's validation total equals the selected NLL within 6.7e-08.
- Post hoc, not registered: the `selector-rows` nu gain has ratios -3.47 (test) and
  -3.39 (validation); over the 24 factor tests the Bonferroni-adjusted normal p values
  are 0.012 and 0.017. The standard error is over records only. Between seeds the ratios
  are -1.65 and -1.81, the gain is carried by seed 20261032 on both partitions (test per
  seed -0.006, -0.047, -0.133), and validation is the selection partition. It is recorded
  as an observation that a future registration could test, not as an effect.
- The registration announced a combined arm "after these verdicts". With no lever
  passing, it has no registered basis; whether to run one is an owner decision.

The "State of the hosts" paragraph of the previous section is superseded by this one.

## Measurement M1: contact pairs of the carrier A/B (registered 2026-11-15, before it was run)
Purpose: find where along the flow the treatment's short contacts form, and choose between
the readings of the previous section. The owner approved the run on 2026-11-15.

Procedure. Script `diagnostics/m1_contacts.py` (new, outside the frozen snapshot), queue
`diagnostics/pc10/m1_queue.sh`, tcm-pc10, PYTHONPATH on the unchanged snapshot
`r3/v7_eval_20261112_code`. One job per carrier field (two arms, three seeds), all 50
test records, replay and oracle, fused sampler, float32, the same protocol seed. The
integrator is replaced by one that makes the same calls in the same order and keeps the
65 RK4 states. A case counts only if every compared field of its replay and oracle
results equals the stored result of that field; any other case is reported and left
out. For every attempt the script reads the cell and atoms the field itself sees
(`PaddedCoverPlan.expand`) and computes exact image distances up to 4 angstrom (checked
locally against `periodic_pairs_within` and `minimum_periodic_pair_distance` on 210
random cells: largest difference 1.8e-15 angstrom over 7836 pairs). Trajectories are
kept for the attempts in `diagnostics/ab_carrier/m1_targets.json`: 131 replay and 111
oracle attempts with no structure or a shortest distance at most 0.8 angstrom in any of
the six replay files.

Definitions. The contact class of an atom pair is the unordered pair of its orbits
("mates" when it is one orbit). For class c at RK4 step k (t = k / 64): r_c(k), the
smallest image distance over its atom pairs; rho_c(k), the smooth separation
|sin(2 pi delta) L| / 2 pi of that minimizing pair, the argument of the switch. Pairs
within 1e-6 relative of the minimum count as tied, and rho_c is the smallest rho among
them (on 3- and 6-fold axes tied pairs can differ in rho; the largest is reported). At
s = 0.2 angstrom the switch is at most 0.9 exactly when rho <= 0.41293 angstrom, at
least 0.95 exactly when rho >= 0.60847 angstrom, and at most 1/sqrt(2) when
rho <= 0.19999 angstrom.

Registered statistic. Set V: the 27 replay attempts that are valid in the control and
invalid in the treatment of the same seed. For each, c* is the treatment's closest class
at t = 1, Lambda(k) = log(r_T,c*(k) / r_C,c*(k)), Lambda_end = Lambda(64) (negative for
every member of V, because every control class ends above 0.5 angstrom), and k_half is
the smallest step with Lambda(k') <= Lambda_end / 2 for every k' >= k; a step where
Lambda is undefined (a distance beyond 4 angstrom or unresolved in either arm) fails the
condition. The statistic is the median over V of rho_T,c*(k_half): the treatment pair's
smooth separation when half of its final log gap has formed.

Decision.
- (a) Median at most 0.41293 angstrom: the gap forms where the pair's own carriers are
  switched (switch at most 0.90). The next treatment is a smaller constant s,
  registered separately; the q-switch is not run.
- (b) Median at least 0.60847 angstrom (switch at least 0.95): the gap forms while the
  pair's own carriers are within 5 % of the legacy ones, so the coincidence part of the
  switch is not the main cause. The next test is the registered q-switch, which keeps
  the coincidence part and removes the half-lattice part.
- (c) Otherwise inconclusive; no treatment follows from M1.

Corrected before any run, after an adversarial review of the script (session workflow
`wf_56bb415a-afd`): the bound of (b) was written as 0.60 angstrom, where the switch is
0.9487, not 0.95; it is now the exact 0.95 point. The tie rule above and the undefined
steps of Lambda were not stated. Pairs exactly on {0, 1/2}^3 (to 1e-6 in delta), whose
carriers are zero in both arms, are left out of the half-lattice counts and counted
separately. The result files now carry the sha256 of the script, the replay module and
the targets file, every class is kept for the traced attempts, and the queue refuses to
skip an existing result that does not parse with 50 cases.

Reported with the decision, not part of it: the same statistic on the band set
(attempts decoded in both arms with sqrt(d_C d_T) in (0.1, 0.5] angstrom and
Lambda_end <= -0.1), the share of mates among the c*, the half-lattice counts along the
paths, the oracle draws, and any attempt with an unresolved step at or after k_half.

M1 launch record (2026-11-15). Files on the host equal the repository copies:
`m1_contacts.py` sha256 `1f8332a8...a432`, `m1_targets.json` `82eab2d7...de4d`,
`pc10/m1_queue.sh` `d6c1039b...05ca`. Queue started on tcm-pc10 at 05:58:59 UTC
(`r3/ab_carrier/m1/queue.log`); it first runs the three-record smoke check and stops
unless all three cases equal the stored replay. At launch the GPU was held by two of the
owner's other training jobs (10.8 GB used, 5.1 GB free), so the queue waits for 6 GB free
before each job. A first smoke attempt at 05:28 UTC was started without that wait and
stopped by hand within seconds; it wrote nothing but an empty log.

M1 completion record (2026-11-15). The owner's jobs ended before 06:56 UTC and the queue
ran without waiting further: smoke check 3 of 3 cases reproduced at 06:56:50, last job
09:16:07, "queue done". Every one of the six files holds 50 of 50 cases reproduced, no
mismatches, 1000 of 1000 replay attempts recorded and script sha256 `1f8332a8...a432`;
the copies read here equal the host files by sha256. The registered analysis
`diagnostics/ab_carrier/m1_analysis.py` has sha256 `fce4ebfb...27a5` (`fce4ebfb8367ce6a
828104b210608fddf885270ffbe9df4a1fcbddc0d1c627a5`) and was last modified at 05:58:08 UTC,
before the first M1 result existed; it was run unchanged on tcm-pc10.

## Result of measurement M1 (read 2026-11-15)

Registered decision: (a). The median over V of rho_T,c*(k_half) is 0.03056 angstrom
against the bound 0.41293; all 27 members are at or below it and none reaches 0.60847.
Printed output: `diagnostics/ab_carrier/m1_verdict_2026-11-15.txt` (sha256
`d63687f1...2991`).

| set | members | median rho_T,c*(k_half), angstrom | at most 0.41293 | mates among c* | median k_half |
|---|---|---|---|---|---|
| V (registered) | 27 | 0.0306 | 27 | 8 | 32 |
| band set (reported) | 68 | 0.0603 | 68 | 11 | 40 |
| oracle draws (reported) | 23 | 0.0302 | 22 | 9 | 37 |

No member of any set has an unresolved step at or after k_half or a c* pair on
{0, 1/2}^3, and the largest tie spread rho_max - rho at k_half is 2e-7 angstrom.

### Independent checks
Session workflow `wf_98af5b6c-398`, read-only, run on tcm-pc10 with the standard library;
scripts and outputs in `diagnostics/ab_carrier/m1_checks/`.
- Two reimplementations written from the registered text, without reading
  `m1_analysis.py`, give the same 27 members with the same c*, k_half and rho, and the
  same median 0.03056363.
- Pairing. Over 393 traced replay attempt pairs and 8277 classes the two arms are equal at
  k = 0 (largest difference 0.0), so Lambda(0) = 0. Indices, request seeds, covers,
  solved and traced flags agree over all 6000 paired attempts. Every kept track has 65
  states. For every member c* is the unique argmin of the treatment trajectory at k = 64,
  and every control class ends above 0.5 angstrom (smallest 0.5005).
- Switch argument. In the frozen snapshot, `nn/invariant_transformer.py` lines 692-713
  form m_1 = sin(2 pi delta) L / l0 and multiply the direct and reciprocal units by
  sqrt((|m_1|^2 + eps) / (|m_1|^2 + (2 pi s)^2)). The inputs are the padded cover-plan
  coordinates and the conventional-v1 Cholesky cell from `PaddedCoverPlan.expand`
  (`learned.py` 1844-1854, `nn/bridge.py` 1088-1097). This is the rho M1 measures. The
  code fingerprint recomputed from the snapshot, `102a0b14...f553`, equals the one in all
  six files.
- Seed 20261032, record 93ebf3b3, attempt 7 has Lambda undefined at steps 8 to 37 because
  the control pair is exactly coincident there (r = 0). The control pair then separates and
  ends at 0.8997 angstrom; the treatment pair stays between 0.0019 and 0.0037 angstrom up
  to step 44 and ends at 0.0471. Its k_half is 41 either way.

Robustness, not registered:

| variant | median, angstrom | at most 0.41293 |
|---|---|---|
| registered | 0.03056 | 27 of 27 |
| seed 20261030, 20261031, 20261032 | 0.01976, 0.03179, 0.03698 | 5 of 5, 7 of 7, 15 of 15 |
| record 9be20999 (6 members) dropped | 0.03179 | 21 of 21 |
| rho_max instead of rho | 0.03056 | 27 of 27 |
| first crossing instead of k_half | 0.03056, the same step for all 27 | 27 of 27 |
| k_half - 1, k_half + 1 | 0.02935, 0.03197 | |
| half point of the absolute gap r_C - r_T | 0.08321 | 27 of 27 |

### What the result shows and what it does not
- The carrier of the first direct harmonic is m_1 / sqrt(|m_1|^2 + eps) in the control
  (`_smooth_unit_vectors`, snapshot line 51), so with the switch it becomes
  m_1 / sqrt(|m_1|^2 + (2 pi s)^2) in the treatment. Its length is rho / sqrt(rho^2 + s^2)
  in the treatment and rho / sqrt(rho^2 + rho_0^2) in the control, with
  rho_0 = sqrt(eps) / 2 pi = 0.0016 angstrom. The smoothing moves the length below which a
  pair's carrier fades from 0.0016 to 0.2 angstrom. At the V median 0.0306 angstrom the
  control carrier has length 0.999 and the treatment carrier 0.151.
- The switch is strongly active on the pairs that turned invalid: at k_half its median on
  c* is 0.151, 22 of 27 are at most 1/sqrt(2) and all 27 at most 0.9.
- These are true near coincidences, not half-lattice ones: rho / r at k_half has median
  0.99993 and smallest 0.912.
- They start nearly coincident. In the initial state the arms share, the c* pair is at a
  median 0.0308 angstrom (q10 0.0031, q90 0.1128). For the 239 traced attempts valid in
  both arms the treatment's closest class starts at a median 0.180 angstrom. The control
  separates the V pairs to above 0.5 angstrom by t = 1; the treatment does not.
- Limit of the registered rule. Every V member ends with the treatment pair at most
  0.5 angstrom (largest rho_T,c*(64) 0.4970), so outcome (b) needed the pair to be farther
  apart at k_half than at the end, a late approach. No member's c* pair exceeds
  0.4970 angstrom at any step. On this set the rule could in practice return only (a), so
  M1 does not separate the readings in the way (b) was framed. This was not seen at
  registration. On the 130 traced attempts valid in both arms with Lambda_end < 0 the same
  statistic is 0.3355, also in region (a).
- The cross carriers are products of switched direct and reciprocal units and carry g^2,
  and node chirality averages switched bases over all partners (snapshot lines 716-749).
  The switch therefore also acts beyond the pair's own primary carriers, and M1 cannot
  exclude a non-local contribution. The phrase "within 5 % of the legacy ones" in (b)
  holds only for the primary carriers; for g^2 the 0.90 and 0.95 points are at 0.59998 and
  0.87175 angstrom.

Consequence. By the registered rule the next treatment is a smaller constant s, registered
separately, and the q-switch is not run. The physics points the same way independently of
the rule: the q-switch keeps the suppression at true coincidence, where these pairs are,
so it would not restore their carriers.

Arithmetic for that registration, g = sqrt((rho^2 + rho_0^2) / (rho^2 + s^2)), the ratio
of treatment to control first-harmonic carrier length:

| s, angstrom | rho 0 | 0.005 | 0.0306 | 0.054 | 0.10 | 0.30 | rho at g = 0.9 |
|---|---|---|---|---|---|---|---|
| 0.2 | 0.0080 | 0.026 | 0.151 | 0.261 | 0.447 | 0.832 | 0.4129 |
| 0.1 | 0.0159 | 0.052 | 0.293 | 0.475 | 0.707 | 0.949 | 0.2064 |
| 0.05 | 0.0318 | 0.104 | 0.523 | 0.734 | 0.895 | 0.986 | 0.1032 |
| 0.02 | 0.0796 | 0.255 | 0.838 | 0.938 | 0.981 | 0.998 | 0.0411 |
| 0.01 | 0.1592 | 0.469 | 0.952 | 0.984 | 0.995 | 0.999 | 0.0203 |

The code requires (2 pi s)^2 > eps, that is s > 0.0016 angstrom. Open for the owner: the 12
control collapses that s = 0.2 removed set the other side of the choice. Whether one
constant s keeps that gain and loses this harm depends on the separations at which the gain
forms, which the same six files can show (the mirror statistic over attempts invalid in the
control and valid in the treatment) with no GPU time. Proposed, not run.

## Measurement M1b: separations of the collapses the smoothing removed (registered 2026-11-15, before it was run)
Purpose: rule (a) of M1 makes a smaller constant s the next treatment. A smaller s gives
the push back to the pairs of V, but it also weakens the smoothing near an exact
coincidence, which removed all 12 control collapses. M1b measures at which separation the
treatment kept those pairs apart and decides whether one constant s can do both. The owner
approved it on 2026-11-15. It uses no GPU and reads only the six M1 result files. No path
of a member of G had been read when this was written: the M1 checks read V, the band set,
the oracle set and the traced attempts valid in both arms.

Definitions.
- Collapse, as in the carrier A/B: no structure, or a decoded shortest distance at most
  0.01 angstrom. Set G: the replay attempts that collapse in the control and do not
  collapse in the treatment of the same seed. The A/B result implies 12 members (5, 3, 4
  per seed), all with no structure in the control.
- c-dagger: the control's closest class in its plan end state at t = 1; M1 records that
  state for attempts that do not decode. Any other class also within 0.01 angstrom there is
  reported.
- k_cap: the smallest step k with r_C,c-dagger(k') <= 0.01 angstrom for every k' >= k; a
  step where r_C,c-dagger is undefined fails the condition. From k_cap on, the control pair
  stays inside the collapse threshold.
- gamma = rho_T,c-dagger(k_cap): the treatment pair's smooth separation at that step.
- Statistic m: the median of gamma over the members of G that have one. A member without
  one is reported with the reason.

Decision. h = 0.03056 angstrom is the registered M1 median. With
rho_0 = sqrt(eps) l_0 / 2 pi = 0.0015915 angstrom, the switch is
g_s(rho) = sqrt((rho^2 + rho_0^2) / (rho^2 + s^2)). s* = 0.014908 angstrom is the s with
g_s*(h) = 0.9 exactly; for it g = 1/sqrt(2) at rho = 0.014737 angstrom, the smoothing
length itself to within 1.2 %.
- (A) m <= 0.014737 angstrom: at the medians one constant s keeps the gain pair inside its
  smoothing length (g <= 1/sqrt(2)) and the harm pair at 0.9 of its legacy carrier. The next
  treatment is a carrier A/B with s = 0.015 angstrom (g(h) = 0.899), registered separately
  on the protocol of the carrier A/B, P4 included.
- (B) m > 0.014737 angstrom: no constant s does both at the medians. No constant-s A/B
  follows; the next construction is an owner decision.
- (C) fewer than half of the members of G have gamma: inconclusive.
The boundary g = 1/sqrt(2) is a convention, the edge of the smoothing length. Whether a
switch of 1/sqrt(2) or less at gamma prevents a collapse is not known; the A/B of (A) would
test it. For that A/B: the P4 slope was 11.16 at s = 0.2; if it scaled as 1/s it would be
about 150 at s = 0.015, below the RK4 limit 178 but close to it.

Validity check, before anything is read: the collapse counts per seed recomputed from the
M1 files must equal the A/B table (control 5, 3, 4; treatment 2, 2, 0). Otherwise no
result is read.

Reported, not part of the decision:
- per member of G: seed, record, attempt, c-dagger, mates, Wyckoff letters, species, the
  starting separation rho(0) that both arms share, k_cap, gamma, r_C,c-dagger(k_cap - 1),
  the treatment pair's smallest rho and its step, rho_T,c-dagger(64), the treatment's
  decoded shortest distance, g_0.2(gamma) and g_s*(gamma);
- m with the collapse threshold in k_cap set to 0.001 and to 0.1 angstrom;
- the mirror of the M1 statistic on set W, the replay attempts invalid in the control and
  valid in the treatment (expected 10): c-dagger as above, Lambda = log(r_T / r_C), which
  is positive at the end, k_half as the smallest k with Lambda(k') >= Lambda_end / 2 for
  every k' >= k (undefined steps fail), and rho_T,c-dagger(k_half). A member whose
  Lambda_end is undefined because the control pair is exactly coincident is reported as
  such;
- G on the oracle draws (the A/B's oracle C went from 11 to 1).

Script: `diagnostics/ab_carrier/m1b_analysis.py`, run on the CPU of tcm-pc10.

Corrected before the run, after a review of the script against this text on synthetic
inputs only (session workflow `wf_4021f4e2-b51`; neither reviewer read an M1 file). The
1/sqrt(2) point is 1.15 % below s*, not within 1 %. A treatment rho is used only where the
same arm's r is defined: where no pair of the class is within 4 angstrom, m1_contacts stores
r = None with the rho of an arbitrary pair, so gamma would not have been the minimizing
pair's. Such a member is reported without gamma. The decision line is printed before the
descriptive lines, so an undefined rho(0) cannot stop it, and rule (A) compares m with the
registered 0.014737, not with the unrounded 0.0147370057. Members without gamma now carry
the descriptive fields. The output adds the control's decoded distance, G per seed, the
oracle collapse counts, and a warning when G differs from 5, 3, 4 or a member decoded in the
control. The script needs Python 3.10 or newer; the host venv is 3.12.3. Final script
sha256 `2cc8fa86...bb32` (`2cc8fa8656fb54931c35b26c5eab20dc8c37e48bb66bfc40891e7e4faed4bb32`),
fixed before any M1 file was read for this measurement.

## Result of measurement M1b (read 2026-11-15)
Run unchanged on tcm-pc10 (venv Python 3.12.3, `nice`, CPU), host copy equal by sha256.
Printed output: `diagnostics/ab_carrier/m1b_verdict_2026-11-15.txt` (sha256
`b9cb3bbc...8de0`).

Validity. The collapse counts are those of the A/B table: control 5, 3, 4; treatment 2, 2,
0. G has 12 members (5, 3, 4 per seed), all `geometry/decode` failures in the control, and
all 12 have gamma.

Registered decision: (A). m = 0.0038558 angstrom against the bound 0.014737; 8 of 12
members are at or below it. By the rule the next treatment is a carrier A/B with
s = 0.015 angstrom, registered separately.

Independent checks (session workflow `wf_f690c522-16c`, run on tcm-pc10, read-only;
scripts and the returned results in `diagnostics/ab_carrier/m1b_checks/`). A blind
reimplementation from this text matches every member, every class, k_cap and gamma, the
median and the decision. It found one inconsistency in the registration text: the text
says h = 0.03056, while s* = 0.014908 and the bound 0.014737 come from the unrounded
0.03056363; from 0.03056 the bound would be 0.014735. This does not affect the decision.

The members, as distinct starting states:

| record / attempt | seeds in G | class, Wyckoff, species | gamma, angstrom | k_cap | treatment decoded distance, angstrom |
|---|---|---|---|---|---|
| d5bbc848 / 13 | all three | 1-4, 2b 4e, W O | 0.001018 | 0 | 0.026, 0.102, 0.178 |
| 782a5e4a / 14 | all three | 2-3, 2a 2a, Ni Ni | 0.002976 | 0 | 0.050, 0.053, 0.043 |
| 9595947a / 14 | 20261030 | 0-3, 4b 8i, Er Rh | 0.004736 | 0 | 0.305 |
| 9be20999 / 0 | 20261030 | 7-7 mates, 4c 4c, F F | 0.012671 | 3 | 0.112 |
| 9595947a / 17 | 20261030 | 4-5, 4a 4a, Er Er | 0.019589 | 7 | 0.312 |
| 782a5e4a / 11 | 20261031 | 1-2, 2d 4k, Pu Ni | 0.024729 | 5 | 1.590 |
| 10c6e7e8 / 8 | 20261032 | 0-6, 1a 4j, Ti Ti | 0.038142 | 7 | 0.734 |
| 8f6174bd / 3 | 20261032 | 1-2, 2d 4f, Zn Ac | 0.057459 | 9 | 0.921 |

Robustness, not registered:

| variant | median gamma, angstrom | side of the bound |
|---|---|---|
| registered, 12 members | 0.003856 | (A) |
| 8 distinct starting states | 0.016130 | (B) |
| distinct, with min, max, median or first-seed gamma per state | 0.016130 | (B) |
| distinct, collapse threshold 0.001 or 0.1 in k_cap | 0.022189, 0.016085 | (B), (B) |
| per seed 20261030, 20261031, 20261032 | 0.004736, 0.002976, 0.020559 | (A), (A), (B) |
| leave one record out, 12 members | 6 of 6 at or below | (A) |
| leave one record out, distinct states | 2 of 6 at or below | mixed |
| largest gamma among classes tied at r = 0 | 0.007824 | (A) |
| oracle draws, 11 members; 6 distinct | 0.005628; 0.013139 | (A); (A) |

### What the result shows and what it does not
- Repeated starting states. All six fields integrate the same covers from the same base
  draws: over all 242 traced attempts, 0 of 120115 values at k = 0 differ between the six
  files. The 12 members are therefore 8 starting states, and two of them, d5bbc848/13 and
  782a5e4a/14, enter in all three seeds with the same gamma. The registration did not
  anticipate this. Counted once each, the starting states give (B), under every way of
  combining the repeats that was tried. The registered (A) depends on the repeats.
- gamma is mostly the starting separation, not a separation at which the treatment held
  the pair apart. 7 of 12 members (3 of 8 states) have k_cap = 0, so gamma is the shared
  rho(0), and the registered median equals the median of rho(0) exactly. In 6 of those 7
  the treatment pair also stays within 0.01 angstrom for 46 to 55 of the 65 steps and
  separates late. In 8 of 12 members the treatment rho falls below gamma after k_cap.
- What s = 0.2 gained here is mostly a decode failure turned into a short contact. 9 of 12
  members (5 of 8 states) end between 0.026 and 0.312 angstrom, still invalid. The 3 that
  become valid have gamma 0.0247, 0.0381 and 0.0575 angstrom, all above the bound, and every
  member at or below the bound stays invalid.
- The same starting state can be a gain at one seed and a harm at another: 9be20999/0 and
  9595947a/14 are in G at seed 20261030 and in V at seed 20261032. These cases depend on the
  trained field, not only on the option.
- The 11 of 12 collapses that are not mates are pairs of different orbits colliding, for
  example two 2a orbits of Ni in Amm2. The case diagnosed on 2026-11-12, two mates of one
  orbit meeting at a special position, is the exception here.
- G and V overlap. The distinct medians are 0.0161 for G and 0.0318 for V, and 7 of the 8
  G values lie inside V's range. With the distinct values, keeping the G median inside the
  smoothing length needs s >= 0.016287, while the switch at h needs s <= 0.014908, so no
  constant s does both. At s = 0.015 the switch at the distinct G median is 0.736.
- The 7 attempts of W that were not collapses gained at a median rho_T(k_half) of
  0.262 angstrom, where g_s*(rho) is about 1, so a smaller s would probably give them back.
  This is an inference and was not measured.

Consequence. The registered decision is (A), so by the rule the next treatment is the
s = 0.015 angstrom A/B. The decision is fragile: it does not survive counting each starting
state once, and the validity gains of s = 0.2 lie above the bound. The likely gain of
s = 0.015 over the control is therefore fewer decode failures, not more valid structures.
Whether to spend the six trainings is the owner's decision.

Common origin, from M1 and M1b together. The attempts that change between the arms start
from near-coincident pairs in the shared base draw: V at a median 0.031 angstrom, G at
0.001 to 0.07 (distinct median 0.016). The traced attempts that are valid in both arms
start at 0.18. Proposed, not run: the smallest starting pair separation of all 1000 replay
attempts and the invalid rate as a function of it. The M1 files hold starting states only
for the 131 traced attempts per seed, so this needs a new run that draws covers and base
points without integrating them; its feasibility has not been checked. If invalidity is
concentrated at near-coincident starts, the lever is the base distribution or how such
starts are handled, not s.

Correction of a statement made to the owner before this verification: the count of
members with k_cap = 0 is 7, not 6.

## Measurement M2: starting pair separations and validity (registered 2026-11-15, before it was run)
Purpose: decide whether the invalid structures of the geometry flow come mainly from
near-coincident pairs in the starting state, the base draw. The owner chose this over the
s = 0.015 angstrom A/B on 2026-11-15. Evaluation only, no new model.

Procedure. Script `diagnostics/m2_starts.py` wraps `diagnostics/m1_contacts.py` (sha256
`1f8332a8...a432`, unchanged) and adds to every solved attempt a `start` block: the
geometry of the plan state at step 0, from the same `PaddedCoverPlan.expand` and the same
exact image distances up to 4 angstrom. No attempt is traced. One field is replayed, carrier
control seed 20261030, on tcm-pc10 with the frozen snapshot `r3/v7_eval_20261112_code`;
every case must reproduce its stored replay and oracle result, or it is left out. All six
fields integrate the same covers from the same base draws, so one replay gives the starting
state of every attempt in all six M1 files. This is checked: for the 242 traced attempts the
new start must equal step 0 of the M1 trajectory in all six files, or no result is read.
Outcomes are read from the six M1 files: an attempt is invalid when it has no structure or
its decoded shortest distance is at most 0.5 angstrom.

Definitions.
- d0: the smallest image distance between two distinct atoms at step 0, in angstrom (above
  4 angstrom recorded as such). A start whose image search is unresolved is reported and
  left out.
- Near-coincident start: d0 <= 0.1 angstrom. Chosen before this run because it is a fifth of
  the validity threshold, and because the starts M1 and M1b examined lie mostly below it (V
  median 0.031, q90 0.113; G at most 0.070), while traced attempts valid in both arms start
  at a median 0.18. That choice used those traced subsets; the full curve is reported.
- Unit: the starting state (record, attempt), 1000 replay states. Each has three control
  outcomes (seeds 20261030, 31, 32) and three treatment outcomes.

Registered statistics, on the control arms, pooled over the three seeds (3000 outcomes):
- F: the share of invalid outcomes whose start is near-coincident;
- RR: the invalid rate of near-coincident starts divided by that of the other starts.
Point estimates decide. Percentile 95 % intervals from 2000 bootstrap resamples of the 50
records (seed 20261115) are reported.

Decision.
- (P) F >= 0.5 and RR >= 5: invalidity is concentrated at near-coincident starts. The next
  treatment acts on the start or on how such starts are integrated, and is designed and
  registered separately; the owner chooses which.
- (N) F < 0.25: near-coincident starts are not the main source. The next measurement is
  M0, the contact rate of the dequantized training target.
- (I) otherwise: both directions stay open; owner decision.

Reported, not part of the decision: the mass of near-coincident starts (the share of the
1000 states with d0 <= 0.1); the invalid rate in d0 bins (0, 0.01], (0.01, 0.03],
(0.03, 0.1], (0.1, 0.3], (0.3, 0.5], (0.5, 1], (1, 4], above 4, for both arms; F and RR for
the thresholds 0.01, 0.03, 0.3 and 0.5 angstrom and per seed; the same with d0 divided by
the starting volume per atom to the power 1/3; whether the closest class at the start is a
pair of mates; for invalid control outcomes, how often the closest class at the end is the
closest class at the start; the treatment arms; the oracle draws.

Scripts: `diagnostics/m2_starts.py`, queue `diagnostics/pc10/m2_queue.sh` (three-record
smoke check first, the GPU-room wait, its own STOP file), analysis
`diagnostics/ab_carrier/m2_analysis.py`, run on the CPU of tcm-pc10.

Corrected before the run, after a review of the three scripts (session workflow
`wf_dddad916-953`; a stub harness ran the unchanged `m1_contacts.main` and the wrapper
without the model, and the analysis was tested on synthetic files; no real result file was
read). Stated precisely now:
- Attempts without a recorded start are left out of F and RR and counted. They are those
  not solved in a recorded padded solve and those whose start search is unresolved. Next to
  the registered line the analysis prints how many invalid control outcomes this leaves out
  and F with them counted as near and as far. The mass is over the states with a defined d0.
- A case that does not reproduce in M2 or in any M1 file is left out of every arm, and the
  count is printed. If F or RR is undefined, no decision is printed.
- Added checks, each a stop: the M1 script's sha256 (full, before the run, in the wrapper),
  a targets file that traces any attempt (before the run), the field of every M1 file
  against its name and six different field checkpoints, radius 4 angstrom and M1
  `keep_min` at least 4, every attempt of M2 equal to `m1_control_s20261030` in decoded
  distance, solved flag and end state, exactly 6 x 242 traced comparisons, and a resolved
  start against an unresolved step 0.
- The normalized variant uses the threshold with the same number of states as d0 <= 0.1.
- An unsolved traced attempt no longer stops the analysis before the decision.
- The queue accepts an existing result only if its schema and the sha256 of the deployed
  `m2_starts.py` match.

Final sha256: `m2_starts.py` `5adfad80...0bd9`, `m1_contacts.py` `1f8332a8...a432`,
`ab_carrier/m2_targets.json` `9817b402...bc5c`, `pc10/m2_queue.sh` `987a668d...9b95`,
`pc10/common.sh` `f68b096e...8dac`, `ab_carrier/m2_analysis.py` `6176970f...c74e`.

M2 launch record (2026-11-15). The files on the host equal these hashes
(`r3/ab_carrier/m2/`, `r3/pc10/m2_queue.sh`). The queue started on tcm-pc10 at 10:34:18
UTC (`r3/ab_carrier/m2/queue.log`), with 9014 MiB of GPU memory free; another job of the
account held 6920 MiB. The smoke check launched at once.

## Result of measurement M2 (read 2026-11-15)
Smoke check 3 of 3 reproduced at 10:38:44 UTC; the full replay finished at 11:12:58 ("queue
done"), 50 of 50 cases reproduced, no mismatches, output
`r3/ab_carrier/m2/m2_starts_control_s20261030.json` (sha256 `60ccf1fe...f844`). The
registered analysis ran unchanged on tcm-pc10; printed output
`diagnostics/ab_carrier/m2_verdict_2026-11-15.txt` (sha256 `c61d3b20...4aca`).

Validity. No case was left out. All 2000 attempts are solved and every start is
resolved, so no outcome was left out. Every M2 attempt equals `m1_control_s20261030` in
decoded distance, solved flag and end state. The start equals step 0 of the M1 trajectory
in all 6 x 242 = 1452 traced comparisons.

Registered decision: (P).

| quantity | value |
|---|---|
| near-coincident starts (d0 <= 0.1 angstrom) | 91 of 1000 states (0.091) |
| F, share of the 127 invalid control outcomes from near starts | 99 / 127 = 0.7795 |
| invalid rate, near starts against the others | 99 / 273 = 0.3626 against 28 / 2727 = 0.0103 |
| RR | 35.32 |
| record bootstrap 95 % | F 0.664 to 0.878; RR 20.3 to 74.0 |

Invalid rate by d0 (three seeds pooled):

| d0, angstrom | states | control | treatment |
|---|---|---|---|
| (0, 0.01] | 12 | 21 / 36 = 0.583 | 27 / 36 = 0.750 |
| (0.01, 0.03] | 19 | 24 / 57 = 0.421 | 29 / 57 = 0.509 |
| (0.03, 0.1] | 60 | 54 / 180 = 0.300 | 58 / 180 = 0.322 |
| (0.1, 0.3] | 114 | 27 / 342 = 0.079 | 29 / 342 = 0.085 |
| (0.3, 0.5] | 113 | 1 / 339 = 0.003 | 1 / 339 = 0.003 |
| above 0.5 | 682 | 0 / 2046 | 0 / 2046 |

Reported with it: per seed F 0.864, 0.735, 0.735 (RR 63.3, 27.7, 27.7); treatment F 0.792,
RR 38.0; oracle draws F 0.757, RR 48.8 (60 of 1000 near); d0 divided by the starting volume
per atom to the power 1/3, at the threshold with the same 91 states, F 0.764, RR 32.3. F at
the other thresholds: 0.165 (0.01), 0.354 (0.03), 0.992 (0.3), 1.000 (0.5). The 12
no-structure control outcomes all start near. In 90 of the 127 invalid outcomes the closest
class at the end is the closest class at the start.

### Independent checks
Session workflow `wf_0e86983a-143`, run on tcm-pc10, read-only; scripts and the returned
results in `diagnostics/ab_carrier/m2_checks/`.
- A blind reimplementation matches every number of the verdict and the decision. It also
  found that each traced attempt's whole step-0 state (19092 class values per file) is
  identical across the six files, and that M1 `shortest` equals the stored replay files for
  all 12000 attempts. Its bootstrap intervals, F 0.667 to 0.880 and RR 20.9 to 75.6,
  differ slightly because the registration fixed the seed but not the resampling routine.
  They are not part of the decision.
- A skeptic confirmed (P) and found three things that change how it is read.

### What the result shows and what it does not
- Within records the effect holds but is weaker. The Mantel-Haenszel RR stratified by record
  is 18.3 (record bootstrap 10.4 to 45.0). The 37 records that hold both kinds of start
  contain 126 of the 127 invalid outcomes; within them the near rate is higher in 26, lower in
  3 and equal in 8. Leaving one record out keeps F between 0.765 and 0.810.
- The start cannot be separated from the cover. Covers are almost unique per attempt (829
  (record, cover) strata for 1000 states; only 6 hold both kinds of start), so stratifying by
  cover cannot be estimated. Coarser cover strata give MH RR 7.2 to 14.1, with the lowest
  bootstrap bound 4.1. Covers with two or more orbits on one Wyckoff letter with a single
  free parameter, a shared occupied axis, hold 69 of the 91 near states and 120 of the 127
  invalid outcomes. Within those covers near starts are invalid in 94 / 207 = 0.454 and other
  starts in 26 / 621 = 0.042. In covers without a shared axis the rates are 5 / 66 and
  2 / 2106. Both the cover type and the start raise the rate.
- What a near start is. Mostly two orbits on one one-parameter axis whose parameters were
  drawn close: 47 of the 91 have the same Wyckoff letter with one free parameter, for example
  R-3m 6c and 6c, both on (0, 0, z). Another 22 are an axis orbit near a fixed site, and 21 are
  mates of one orbit. None of the 91 pairs is between two fixed sites. Near starts occur in
  larger cells (median 21 atoms against 16).
- No match depends on them. None of the 91 near states is on the record's own target cover.
  Among near starts that end valid, 1 of 174 matches the target, against 243 of 2699 for the
  other starts. Excluding near starts would lose 1 of 244 matched control attempts and no
  record of match@20 (13, 14 and 13 of 50 stay).
- Threshold. The registered threshold was set before the run. F first reaches 0.5 at
  d0 <= 0.061 angstrom, and at 0.03 angstrom the decision would have been (I).
- Counterfactual arithmetic, not a measurement. Had the 273 near outcomes the other starts'
  invalid rate, the control would have about 31 invalid outcomes instead of 127, or 34 with
  record-specific rates and 50 with the rate of the (0.1, 0.3] bin. That is structural
  validity of about 0.99 instead of 0.958. It would not change match rates.

Consequence. By the registered rule the next treatment acts on the start, or on how such
starts are integrated. It is designed and registered separately, and the owner chooses.
Because the start and the cover are confounded here, only an intervention on the start of
the same covers can show that the start causes the failures. That intervention is the
paired test: the same covers with the near-coincident base draws redrawn, every other draw
unchanged.

## Measurement M3: paired redraw of near-coincident starts (registered 2026-11-15, before it was run)
Purpose: test whether the near-coincident start causes the invalid outcomes that M2 found,
by changing the start of the same covers and nothing else. M2 cannot separate the start
from the cover. The owner approved this test on 2026-11-15 after the M2 decision (P).
Evaluation only: no new model and no retraining.

Procedure. Script `diagnostics/m3_redraw.py` wraps `diagnostics/m1_contacts.py` (sha256
`1f8332a8...a432`, unchanged) as M2 did, with the M2 arguments, and changes one thing.
Inside every padded solve, a start with d0 <= 0.1 angstrom is drawn again from the same
attempt's geometry generator, which continues its own stream. The draw repeats until d0
exceeds 0.1 angstrom or 1000 draws are used. d0 is read from M2's start block
(`m2_starts.start_summary`, sha256 `5adfad80...0bd9`, radius 4 angstrom) of the state the
integrator starts from. If that image search is unresolved, an exact search at 0.1 angstrom
decides. Every other start is its first draw. The covers, the field, the members and order
of each padded solve, and the integrator are unchanged. Replay attempts and oracle draws
are both treated. One field is replayed: carrier control seed 20261030, the field of M2, on
tcm-pc10 with the frozen snapshot `r3/v7_eval_20261112_code`.

What the treatment is. Each redraw is a rejection step, so the treated sampler draws its
start from the base law conditioned on d0 > 0.1 angstrom. It uses no information about the
target or the outcome. Its law is the model's law conditioned on an event of the start; it
is a change of the sampler, not of the trained field.

Unit and pairing. The unit is one of the 91 replay states that M2 found near in this field.
Each has two outcomes: its first draw (the stored replay, equal to M2) and its redraw (M3).
An outcome is invalid when the attempt has no structure or its decoded shortest distance is
at most 0.5 angstrom. In this field 38 of the 91 first draws are invalid, against 6 of the
909 other starts, and 1 of the 91 matches its target. These counts come from the M2 file
and the stored replay; they were read before this registration and fix no threshold below.

Checks, each a stop in the analysis:
- the attempts redrawn are exactly the solved attempts that M2 found near;
- the first draw of every redrawn attempt equals its M2 start block exactly, and every start
  that was not redrawn equals its M2 start block exactly;
- an accepted start has d0 > 0.1 angstrom, unless it reached the draw cap;
- M3 agrees with M2 and the stored replay in field, checkpoints, records, seeds and code,
  and the sha256 of the three scripts are the registered ones.

Reported, not a stop: attempts not redrawn whose outcome differs from the stored replay.
Their draws are identical by construction, but they share padded solves with redrawn rows,
and batched numerics could couple rows. Left out and counted:
- a case with no redrawn attempt that does not reproduce;
- a near state that was not treated, because it was solved only in a single-member fallback
  solve, which has no redraw;
- a near state still near after 1000 draws.
For left-out states the redraw count is printed with them counted invalid and valid.

Registered statistic. R = n_new / n_old, where n_old and n_new are the numbers of invalid
first draws and invalid redraws among the paired near states, with the exact two-sided
McNemar p on the discordant pairs. The point estimate and p decide.

Decision.
- (S) R <= 0.5 and p <= 0.01: the near start causes most of its invalid outcomes. The next
  step is a start treatment as an opt-in sampler option, or a change in training, designed
  and registered separately; the owner chooses.
- (C) R >= 0.8: redrawing the start does not remove the failures, so the cover is the main
  cause. The next step is a cover-side investigation of shared one-parameter axes.
- (I) otherwise: owner decision.

Expectation, stated before the run. If the start is the cause, n_new falls towards the rate
of far starts in comparable covers. In covers with a shared axis that rate was 26 / 621 =
0.042 in M2, three seeds pooled, so a few of the 91. If the cover is the cause, n_new stays
near 38.

Reported, not part of the decision:
- matches among the 91 near states; a sign test over records of the change in invalid count;
  the first-draw invalid rate of the far starts of the same records;
- the whole replay and oracle arms, stored against M3: no structure, structurally valid,
  matched, records with a match, failed stages, oracle first-draw matches;
- the near oracle draws, paired, for invalidity and for matches, with McNemar p. In this
  field there are 60; 17 are invalid and 2 match, against 8 invalid and 441 matched of the
  940 far draws;
- the near states split by a shared one-parameter Wyckoff axis in the cover (the M2
  skeptic's definition);
- draws per redrawn attempt, and the redraw outcome by the d0 of the used start.

Scope. One field. The other two control seeds are not run; that is an owner decision after
the result.

Scripts: `diagnostics/m3_redraw.py`, `ab_carrier/m3_targets.json`, queue
`diagnostics/pc10/m3_queue.sh`, analysis `ab_carrier/m3_analysis.py` (run on the CPU of
tcm-pc10). The queue first runs a three-record smoke check. It must redraw exactly M2's near
starts of those records, keep every other start and outcome, and use no fallback solve, or
the queue exits. Then it waits for GPU room and checks its own STOP file.

Corrected before the run, after a review of the three scripts (session workflow
`wf_61c9830d-d1b`). A numpy stub harness ran the real `m3_redraw.py` end to end on the
snapshot's call structure, including the halving fallback and the draw cap. The analysis
was tested on synthetic M3 files built on the real M2 file and stored replay. A skeptic read
the design. No M3 result exists. No defect was major; these changes were made:
- What the treatment is, stated exactly. The treated sampler keeps the cover law and, for
  each cover c, replaces the base law B by its conditional law given A = {d0 > 0.1
  angstrom}. That is not the joint law conditioned on A, because B(A | c) depends on the
  cover. In this field 69 of 276 replay states in shared-axis covers start near, against 22
  of 724 in other covers. A likelihood under this sampler carries the factor 1 / B(A | c),
  and the draws per attempt estimate it.
- A near state whose redrawn start makes its padded solve fail now counts as an invalid
  redraw in R. Before, it was left out. The fallback integrates the untreated first draw,
  so its M3 outcome is not the treated one. Such a failure can only come from a redrawn row,
  because no padded solve failed in M2 with the same members. R and its McNemar p are also
  printed with these states counted valid. The wrapper records every failed padded solve
  with its case, space groups and draws, and the analysis lists any that holds no redrawn
  row. An attempt unsolved in M3 while no padded solve failed is a stop.
- A state still near at the draw cap is left out of the pair, as registered. R and its
  McNemar p are also printed with it counted an invalid redraw and a valid redraw, its first
  draw kept, and the failed solves still counted invalid. "Reached the cap" is now checked: near_used must
  equal the nearness of the used start (for an unresolved block, the exact search's d0),
  near_used must mean 1000 draws, and an attempt that was not redrawn must not be near.
- Changed: a case is left out of the statistics only when it holds no M2 near start, replay
  or oracle, and does not reproduce. Before, the condition was "no redrawn attempt". Capped
  states are still left out of the pair. The treatment checks run over every record,
  including left-out cases.
- Added stops: the covers of every solved attempt equal M2's; the request seeds of every
  case agree in M3, M2 and the stored replay; the stored replay is the registered file
  (sha256 `d2d6c00c...9075`) and agrees with M3 in protocol seed, checkpoints, records,
  candidates, oracle draws and code; the replay module of the oracle path has M2's sha256;
  step 0 of every recorded trajectory equals the start
  the redraw accepted. The wrapper records this last check instead of raising, so a failure
  keeps the run's output. A start that is unresolved even by the exact search at 0.1
  angstrom stops the run; it needs a lattice plane spacing below about 0.013 angstrom.
- Wrapper: a stop raised inside sampling no longer loses its message, because evaluate_case
  turned it into a request error. A request error now stops the run with that message; the
  stored replay has none. After the redraws, the start blocks of the exact integrated state
  are recomputed, and any change against the per-iteration blocks is counted. When the first
  or the used start block is unresolved, the exact search's d0 of both is stored, None
  meaning no pair within 0.1 angstrom or a resolved block. Abbreviations of --radius and
  --max-reach are refused. If the final checks raise, the finished run is written to a side
  file before the stop.
- The smoke check is stricter than first written. It also requires no failed padded solve,
  no draw cap reached, no changed start block and at least one redraw; its 3 records hold 8
  replay and 13 oracle near starts. The full result is checked for schema, script and case
  count after a fresh run as well.
- Added reported statistics: a record bootstrap 95 % interval for R (2000 resamples of the
  sorted records holding paired near states, `random.Random(20261115).choices`); failed stages of the
  oracle arm; a prediction of the oracle redraw from M2 alone, from the far draws of the
  same records, which share the target cover; for invalid redraws with a solved start,
  whether the closest class at the end is the closest class of the first draw; over the
  solved paired states, the median ratio of the volume per atom of the used start to the
  first draw; unresolved used starts in their own bin; far attempts
  unsolved in M3.

Calibration, from the skeptic's simulation on the 91 states. With no overlap between old and
new failures, (S) needs n_new <= 17. With overlap it allows up to 19, the largest value with
R <= 0.5; 18 needs at least 1 shared failure and 19 at least 4. A pure start cause gives
(S) with probability about 1, and a pure cover cause gives (C) with probability 0.947 to 1,
depending on how the cover effect is modelled.
The expectation above, "a few of the 91", may be low: near-prone covers also have higher
far-start rates, so a start cause may give 5 to 10.

Threshold provenance. The 0.1 angstrom threshold was set before M2 from traced subsets of the
same 50 test records. M3 does not depend on it beyond defining the 91 states. A later
validity or match figure of a start-rejection sampler on these records would carry that
selection, so it would be measured on other records or stated with its provenance.

Two more review rounds before the run (session workflows `wf_b1cc796d-0ff` and
`wf_a4a50359-c52`) re-ran the stub harness and the synthetic suites on the changed scripts,
and checked this section against the code line by line. They found no major defect. Their
minor findings are folded into the text above: the sensitivities are printed as R with
McNemar p, an unresolved used start is checked against its exact-search d0, an unsolved
near state needs a failed padded solve of its own case that holds a redrawn row, failed
padded solves record their case and space groups, and the oracle path's module has M2's
sha256. An unresolved used start without an exact-search d0 pair is a stop.

Final sha256: `m3_redraw.py` `d81edefd...12c7`, `m2_starts.py` `5adfad80...0bd9`,
`m1_contacts.py` `1f8332a8...a432`, `ab_carrier/m3_targets.json` `6d760d6f...477f`,
`ab_carrier/m3_analysis.py` `1f38297d...1707` (it pins `m3_redraw.py`), `pc10/m3_queue.sh`
`2b6a50fd...8c29`, `pc10/common.sh` `f68b096e...8dac`. Inputs: M2 result `60ccf1fe...f844`,
stored replay `d2d6c00c...9075`.

M3 launch record (2026-11-15). The files on the host equal the final hashes
(`r3/ab_carrier/m3/`, `r3/pc10/m3_queue.sh`, `r3/pc10/common.sh`), and so do the two inputs.
The queue started on tcm-pc10 at 14:05:28 UTC (`r3/ab_carrier/m3/queue.log`), with 9014 MiB
of GPU memory free. No other M queue was running. The smoke check launched at once.
The smoke check passed at 14:10:04 UTC on 3 records. It redrew exactly M2's near starts of
those records (8 replay, 13 oracle), every first draw equals its M2 start, the 99 attempts
not redrawn equal the stored replay exactly, and no padded solve failed, no draw cap was
reached and no start block changed. The full replay started after it.

## Result of measurement M3 (read 2026-11-15)
The full replay finished at 14:44:44 UTC ("queue done"), 50 of 50 cases, output
`r3/ab_carrier/m3/m3_redraw_control_s20261030.json` (sha256 `b5fd5622...9318`). The launching
ssh session was closed by the host during the full replay, before 14:35; the queue ran
detached and was not affected, and its log kept being written. The registered analysis ran unchanged on tcm-pc10; printed output
`diagnostics/ab_carrier/m3_verdict_2026-11-15.txt` (sha256 `14fcd40f...3654`).

Validity. No case was left out; the 10 cases that reproduce exactly are the 10 records
without a near start. Every registered stop passed:
- the redrawn attempts are exactly M2's near starts, 91 replay and 60 oracle;
- all 151 first draws equal their M2 start blocks, and all 1849 other starts equal M2's;
- covers, request seeds, checkpoints, code and the replay module agree with M2 and the
  stored replay;
- every accepted start has d0 > 0.1 angstrom (smallest 0.1002), and no start reached the
  draw cap;
- no padded solve failed, no single-member fallback was used and no start block changed.
The 1849 attempts not redrawn equal the stored replay exactly, in decoded distance, match
and end state. The redraw cost a median of 2 draws (mean 2.62, at most 9).

Registered decision: (S).

| quantity | value |
|---|---|
| near replay states, paired | 91 |
| invalid: first draw and redraw / first draw only / redraw only / neither | 10 / 28 / 4 / 49 |
| invalid first draws, n_old | 38 / 91 = 0.418 |
| invalid redraws, n_new | 14 / 91 = 0.154 |
| R | 14 / 38 = 0.368 |
| exact McNemar two-sided p | 1.93e-05 (28 against 4) |
| record bootstrap 95 % of R | 0.224 to 0.533 |

Reported with it:

| quantity | stored replay | M3 |
|---|---|---|
| replay: structurally valid, of 1000 | 956 | 980 |
| replay: without structure | 5 | 0 |
| replay: matched attempts | 82 | 81 |
| replay: records with a match (match@20) | 13 | 13 |
| oracle: structurally valid, of 1000 | 975 | 984 |
| oracle: without structure | 3 | 0 |
| oracle: matched draws | 443 | 453 |
| oracle: records with a match | 32 | 32 |
| near oracle draws (60): invalid | 17 | 8 (McNemar p 0.0039) |
| near oracle draws (60): matched | 2 | 12 (10 gained, 0 lost; p 0.0020) |

- Records holding near states: 37; fewer invalid after the redraw in 18, more in 1, equal in
  18 (sign test p 7.6e-05). Far starts of the same records: first draw invalid 5 / 649.
- Shared one-parameter axis in the cover: invalid 36 -> 14 of 69; no shared axis: 2 -> 0 of
  22. Every invalid redraw is in a shared-axis cover.
- Redraw outcome by d0 of the used start: (0.1, 0.3] 13 / 50 invalid, (0.3, 0.5] 0 / 11,
  (0.5, 1] 1 / 13, (1, 4] 0 / 17.
- The prediction of the oracle redraw from M2 alone, from far draws of the same records, was
  12.2 matches and 3.8 invalid; observed 12 and 8.
- Near replay states matched: 1 first draw, 0 redraws. Median volume per atom of the used
  start over the first draw: 1.21. For 1 of the 14 invalid redraws the closest class at the
  end is the closest class of the first draw.

### Independent checks
Session workflow `wf_1a3377e9-23d`, read-only on local copies of the result files (hashes
verified); scripts and the returned results in `diagnostics/ab_carrier/m3_checks/`.
- A blind recomputation from the registration and the raw files matches every number of the
  verdict and the decision, and finds that no registered stop would fire. It also checked
  that the 1849 attempts not redrawn have end blocks identical to M2, and that every M1 file
  equals its stored replay.
- A skeptic found no error and confirmed (S). Its findings change how (S) is read and are
  stated below.

### What the result shows and what it does not
- The start is causal on these covers. Changing only the start of the same covers in the
  same field removes 24 of the 38 invalid first draws (28 fixed, 4 broken). In the oracle
  arm, where all 20 draws of a record share the target cover, near draws behave like far
  draws of the same cover after the redraw: 12 matches against 12.2 predicted. They are
  invalid 8 times, against 6.3 expected if the redraws and far draws of each record were
  exchangeable (exact conditional P(X >= 8) = 0.18; the printed 3.8 uses the far first draws
  alone). All 8 are in one record (10c6e7e8, R-3m).
- "Most" holds at the point estimate, not at 95 % confidence. The registered rule uses the
  point estimate. P(R > 0.5) is 0.045 under the record bootstrap and 0.075 under a state
  bootstrap (95 % 0.205 to 0.567). Leaving one record out gives R 0.333 to 0.412, and 7
  records chosen to raise R must be removed before it exceeds 0.5. R also depends on the
  0.5 angstrom validity rule: 0.250 at 0.3 angstrom, 0.231 at 0.4, 0.488 at 0.6, 0.574 at 0.7
  and 0.667 at 0.8, all with p <= 0.002. The redrawn outcomes improve but stay closer than
  far starts: median decoded shortest distance 0.620 angstrom for the near first draws,
  1.080 for the redraws, 1.407 for far starts of shared-axis covers in the same records.
- The remaining failures are the same mechanism at a new pair. In 12 of the 14 invalid
  redraws the closest class at the end is the closest class of the redrawn start, against 1
  of 14 for the first draw. The redraw put that pair just above the threshold: 13 of the 14
  have a used d0 in (0.1, 0.3]. Pooling first draws and redraws of the 91 covers by the d0
  of the integrated start, the invalid rate falls steadily: (0, 0.01] 8/12, (0.01, 0.03]
  8/19, (0.03, 0.1] 22/60, (0.1, 0.15] 10/17, (0.15, 0.2] 2/14, (0.2, 0.3] 1/19,
  (0.3, 0.5] 0/11, (0.5, 1] 1/13, (1, 4] 0/17. Redraws with d0 <= 0.2 are invalid 12/31;
  redraws with d0 > 0.2 are invalid 2/60, R 0.105 there. The threshold of 0.1 angstrom is
  too low for these covers.
- Near-prone covers keep extra hazard at the same d0. From far starts of the same cover class
  and d0 bin, n_new is predicted at 4.9 (this field) or 6.2 (three control seeds), against 14
  observed (Poisson P(X >= 14 | 6.2) = 0.0047). In (0.1, 0.3], redraws are invalid 13/50
  against 6/70 for far first draws of shared-axis covers (Fisher p 0.012). The draw counts
  imply P(near | cover) of about 0.38 in these covers, against 0.091 overall, so accepted
  starts pile up just above the threshold (55 % in (0.1, 0.3], against 34 % of far starts
  in shared-axis covers). This exceeds both expectations stated before the run, "a few" and
  "5 to 10". Against the floor of about 5, the redraw removes (38 - 14) / (38 - 5) = 0.73 of
  the excess. In the whole replay arm invalid outcomes fall from 44 to 20, about 0.65 of the
  arithmetic gap.
- Cell volume does not explain the effect. The volume ratio of the used to the first start
  is 1.209 for invalid redraws and 1.211 for valid ones. In M2, far shared-axis starts in
  (0.1, 0.3] by volume tercile within records are invalid 2/24, 1/23 and 3/23 (23/144,
  10/138, 19/138 over six fields). The ratio of 1.21 arises because near first draws have
  small cells (0.90 of the record median) and accepted starts larger ones (1.15).
- One field. The same 91 starts are harmful in all six fields (first draws invalid 38, 36,
  25, 42, 39 and 33 times; 17 states invalid in all six, 39 valid in all six), and the d0
  dose response repeats in every field. R itself is measured only in control seed 20261030.
  Predicted from each field's own far rates it would be 0.11 to 0.28 there, against 0.37
  observed here. The direction is likely to transfer; the size need not.
- Accuracy. Given the true cover, the oracle match rate rises by 10 of 1000 draws (0.443 to
  0.453; Clopper-Pearson 0.0048 to 0.0183 for the gain, record bootstrap 0.003 to 0.018),
  with 0 losses. All gains are in 7 records whose far draws match. 8 of the 10 came from
  first draws that were valid but did not match, so on the target cover a near start costs
  accuracy apart from validity. In the replay arm, near starts are never on the target cover
  (M2), and match@20 stays at 13 of 50. The one lost replay match (record 8f6174bd, attempt
  3, cover I-4, not the target Fm-3m) is a chance event: the same first start matched in 3 of
  6 fields, and the record keeps 7 other matches.
- Any new threshold chosen now would be post hoc. A threshold of 0.2 or 0.3 angstrom would be
  read off the M2 table and the M3 bins, on the same 50 records and the same field. The bins
  that would decide it are small (10/17 in (0.1, 0.15], Clopper-Pearson 0.33 to 0.82; 0/11
  in (0.3, 0.5], upper bound 0.285). Rejection at T moves the accepted mass to just above T,
  where the hazard in near-prone covers is not measured. A higher T also raises the draw
  count and the likelihood factor 1 / B(A | c).

Consequence. By the registered rule, the next step is a start treatment, either as an opt-in
sampler option or as a change in training. It is designed and registered separately, and the
owner chooses. Any threshold must be fixed before it is measured, and measured on records
not used here.

## Measurement M4: opt-in start-rejection sampler (registered 2026-11-16, before it was run)
Purpose: turn the M3 finding into a sampler option, choose its threshold by a fixed rule,
and confirm it on records that no geometry replay has used. M3 (S) showed that the
near-coincident start causes most of its invalid outcomes, at the point estimate (R =
0.368). It also showed that any threshold read off M2 or M3 would be post hoc. The owner
chose this option on 2026-11-16. After the first pre-run review, the owner also chose the
partitions: the threshold is chosen on the test records, the records of M1 to M3, and
confirmed once on the validation records. Evaluation only: no new model and no retraining.

### The option
`LearnedCrystalGenerator.start_min_separation`, in angstrom. The default `None` is off and
leaves every code path unchanged.

With a value s, every base draw of the geometry ODE whose two closest distinct atoms are at
most s apart is drawn again:
- d0 is read on the state the field sees at t = 0. That is the torus-canonical start,
  expanded on the cover's cell, whose lengths are angstrom in every domain. The search is
  exact over periodic images (`closest_distinct_pair_within`); images of one atom are not a
  pair.
- The new draw is the same call to `sample_product_base` with the attempt's own generator,
  which continues its stream. Each attempt has its own generator, so its accepted draw
  depends only on its cover and its own stream. The rows of one padded solve are redrawn
  in rounds. A far first draw is kept bitwise.
- An attempt fails with `RecoverableGenerationError`, and so has no structure, in two cases:
  all 1000 draws are near, or the image search needs more than 8 images per axis (a plane
  spacing below s / 7.5). The attempt ends as a `geometry/ode` failure.
- The error fails the whole solve. `mixed-v1` then halves the solve, restoring every
  generator, until the attempt is alone; `bucket-v1` replays every member alone. The
  siblings keep their draws but are integrated in smaller solves. So they equal, to
  rounding and not bitwise, the same attempts in a solve where no member fails: untreated
  ones if they were not redrawn, treated ones if they were.
- The option acts in both samplers and in the single-member fallback.

The law it samples:
- The cover law and the trained field are unchanged.
- For each cover c, the base law B becomes its conditional law given A = {d0 > s}, and
  given that the image search resolves. All 2000 M2 starts on the test records resolve
  (plane spacing at least 0.53 angstrom); the validation starts are not yet seen. This is
  not the joint law conditioned on A, because B(A | c) depends on the cover.
- For a geometry q whose start lies in A, the treated log density is the flow log density
  minus log B(A | c), one constant per cover. For any other q the density is zero.
- Given success, the cap leaves this law unchanged; it lowers a cover's success
  probability to 1 - (1 - B(A | c))^1000.
- `select_map_sample` compares densities within one cover only, so its choice keeps its
  meaning; its `log_geometry` values are the unconditioned densities. `rank_samples`
  compares across covers and refuses while the option is set. Samples carry no record of
  the option, so that guard follows the generator's current setting.
- Training still uses the unconditioned base.
- The CSP replay below switches the likelihood rule off, as before.

Tests (`tests/test_start_rejection.py`, 17 tests, all pass on the Mac; the 38 tests of
`test_mixed_sampler.py` and `test_flow_log_density.py` still pass):
- the search equals a brute-force image sum to 1e-12, including a strongly sheared cell
  whose minimizer lies several images away, and does not count an atom's own image;
- it refuses a cell that needs more than 8 images per axis;
- the option is off by default and validated;
- a threshold that rejects nothing changes nothing, bitwise;
- every integrated start of a treated solve is far, by an independent brute-force search;
- a far first draw is kept bitwise, and a near one is replaced by the first far draw of its
  own generator's stream;
- with a cap of one draw, exactly the attempts whose first draw is near fail, and the
  others equal the untreated run;
- the separation is in angstrom when the domain's reference length is not 1;
- the bucket sampler is treated the same way and equals the fused sampler to 1e-10;
- the likelihood guards and the checkpoint binding.

Relation to M3. At s = 0.1 the option is M3's treatment, apart from four points:
- At the cap, M3 integrated the near start; the option fails the attempt. M3 never reached
  the cap.
- M3 did not treat a single-member fallback solve; the option does. M3 had none.
- M3 stopped the run on an unresolved exact search; the option fails the attempt. M3 had
  none.
- M3 decided near on M2's r rounded to 7 significant digits; the option compares the
  unrounded value. They differ only for d0 within about 5e-8 angstrom of s.
Two checks below require equality with M3.

Snapshot. `r3/v7_eval_20261116_code` is `r3/v7_eval_20261112_code`, the snapshot of M1 to M3,
with `src/crystalatlas/learned.py` replaced. The queue refuses to start if anything else
differs or if any deployed file is not the registered one. The analysis pins the code
fingerprint of that tree, so every M4 file is tied to the M1 to M3 code plus the
registered `learned.py`.

### Measurement script
`diagnostics/m4_rejection.py` runs the paired field replay of `diagnose_field_replay_v7` on
one partition, with the option set to `--threshold`:
- covers from run 2's selector and the field of one control seed;
- protocol seed 20261530, the reduced-formula CSP track;
- 20 replay attempts and 20 oracle draws per record.

The option does the treatment. The script only observes, with wrappers that change no
value. For every member of every padded solve it records:
- the number of base draws its generator made in that solve;
- M2's start block (radius 4 angstrom) of the state that was integrated;
- the unrounded d0 up to 0.5 angstrom by the option's own search, on which the option
  decides. For M4 records the analysis decides near and far on this value, and requires it
  to agree with the independent radius-4 block to 1e-6 relative, a stop; the 7-digit
  rounding errs by at most 5e-7 relative. M2's records have no unrounded d0, so the
  selection decides its reference first draws on M2's 7-digit r. No M2 first draw lies
  within 2e-4 angstrom of a registered threshold, so the rounding window cannot make a
  stop fire on these records;
- for a block that the radius-4 search leaves unresolved, an exact search at the
  threshold (0.5 angstrom in an untreated run, so that every later threshold can read it).
  If that search is unresolved too (a plane spacing below about 0.067 angstrom), the run
  ends, where the option would only fail the attempt. No M2 start comes close.

For every replay attempt it records the failed stage, the error and the decoded volume per
atom. It counts the fallback and failed padded solves. A request error is kept as an
outcome, with every attempt of the case a miss without a structure. A fault of the
wrappers or of the observation ends the run; it cannot pass as a member failure. If the
run's own checks raise, the finished run is written to a side file first.

### Selection on the test records
Records and field: the 50 test records of M1 to M3, in carrier control seed 20261030, the
field of M2 and M3. These records are already used, so selecting on them costs no fresh
evidence. The untreated arm is the stored replay (sha256 `d2d6c00c...9075`); the first
draws come from M2 (sha256 `60ccf1fe...f844`).

Sequence, in angstrom: none (untreated), 0.1, 0.2, 0.3, 0.5.
- 0.1 is M3's threshold. 0.5 is the structural validity distance, at which a start's
  closest distinct pair already meets the validity rule. 0.2 and 0.3 lie between.
- The grid was fixed knowing M3's d0 bins on these records, where the remaining failures
  lie in (0.1, 0.3] angstrom.
- The run at 0.1 must equal M3 in every case, draw count and integrated start, a stop. Its
  outcomes are therefore M3's, already read: the step none -> 0.1 is known to be accepted.
- At 0.2 and 0.3, an attempt whose M3 start already had d0 above the threshold keeps it.
  So part of each later arm is known as well: 60 of the 91 near replay states and 43 of the
  60 near oracle draws at 0.2, and 41 and 34 at 0.3.

Unit: each of the 2000 attempts, the 1000 replay attempts and the 1000 oracle draws, paired
across thresholds by cover and generator seeds. The oracle draws are pooled with the replay
attempts because both are draws of the same sampler, on the selector's covers and on the
true cover, and pooling doubles the data. An attempt is invalid when it has no structure or
its decoded shortest distance is at most 0.5 angstrom; matched as in the protocol.

Step from the current threshold T to the next T':
- fixed: invalid at T and valid at T'; broken: the reverse;
- gained: matched at T' and not at T; lost: the reverse; over all 2000 attempts, and
  separately over the 1000 replay attempts.

The step is accepted when all three hold:
- robust gain: P(Bin(fixed + broken, 1/2) >= fixed) <= 0.05, both with every record and
  with any single record left out;
- no harm: P(Bin(gained + lost, 1/2) >= lost) > 0.10;
- no replay harm: the same over the replay attempts, > 0.10, so that oracle gains cannot
  hide a loss of CSP matches.

The sequence stops at the first rejected step. The selected threshold T* is the last
accepted value: none if the first step is rejected, 0.5 if every step is accepted. A
larger threshold is therefore taken only when it removes significantly more invalid
outcomes than it creates, in more than one record, and does not lose matches at the 0.10
level. The rule leans towards the smaller treatment on purpose, because a larger threshold
conditions the base law more strongly. Attempts of one record are not independent, so
these p values are a selection device, not a test.

Calibration:
- The smallest accepting splits of fixed / broken are 5 / 0, 7 / 1, 9 / 2 and 10 / 3, after
  the record that contributes most is removed.
- A step with no match gained is rejected at 4 lost; with 1, 2 or 3 gained, at 6, 7 or 9
  lost.
- Step 0.1 -> 0.2: of the 151 attempts whose 0.1 start lies in (0.1, 0.2] angstrom, 25
  are invalid at 0.1. 10 of those are in record 10c6e7e8, 9 of them on its R-3m target
  cover of 39 atoms. The robust condition requires the gain without that record too.
- Step 0.2 -> 0.3: first draws with d0 in (0.2, 0.3] were invalid 2 / 58 (replay) and
  3 / 53 (oracle), and M3's redraws there 1 / 19 and 1 / 9. So this step turns on a
  handful of discordant pairs.
- Step 0.3 -> 0.5: first draws in (0.3, 0.5] were invalid 0 / 113 and 3 / 90, so 0.5 is
  reachable in principle only. At 0.5 the draw cap may fail the draws of near-prone covers:
  all 20 oracle first draws of 10c6e7e8's target cover have d0 <= 0.345 angstrom. Each
  capped draw costs up to 1000 rounds at every level of the halving: minutes, not hours.
- The expected selection is 0.2 or 0.3.

### Confirmation on the validation records, once
Records: the 50 validation records of the pilot store, composition-disjoint from the
training and test records. No geometry replay has used them. They did select the
checkpoints of run 2's cover selector and of each control field (`--select-best`), and they
entered the cover-selector diagnostics. There the selector behaves differently than on the
test records: the nu gain over the uniform policy is -0.014 on validation against 0.564 on
test, and the row gain 0.622 against 1.277. So the share of near-prone covers on the
validation records may differ from the test records.

If T* is none, no confirmation is made, and the decision is (V): not supported. Otherwise,
in each of the three control fields 20261030, 20261031 and 20261032, two runs are made on
the validation records: an untreated run and a run at T*. Each attempt of the treated run
is paired with the same attempt of the untreated run of its field.

The three fields are three training seeds of the field, evaluated on identical covers, first
draws and redraws, because d0 depends only on the cover and the base draw. This is checked
as a stop. The fields are therefore not independent replicates. The inferential unit of the
confirmation is the record.

Decision statistics, with sums over the three fields:
- gain: for each record, the invalid replay attempts untreated minus treated. p_gain is the
  one-sided sign test over records, P(Bin(n+ + n-, 1/2) >= n+), where n+ and n- count
  records with fewer and with more invalid attempts. Records with no change do not count.
- harm: for each record, the matched replay attempts and oracle draws lost minus gained.
  p_harm is the one-sided sign test of records with a net loss against records with a net
  gain.
- replay harm: the same over the replay attempts only, p_harm_replay. The replay is the
  sampler's real output.

Decision:
- (H) p_harm <= 0.05 or p_harm_replay <= 0.05: harmful; not supported.
- (S) not (H), p_gain <= 0.01, and fewer invalid replay attempts in each of the three
  fields: supported as an opt-in sampler setting at T*, with no harm to matches detected
  at the power of the harm tests. It stays opt-in. A default, or use in run 3, is an owner
  decision.
- (N) not (H) and p_gain > 0.01: a validity gain is not shown; not supported. This includes
  the case of too few records with a change to reach the level.
- (I) otherwise, a gain over records but not in every field: owner decision.

The every-field condition is a consistency check, not a replication: the fields share
every start.

Calibration:
- p_gain <= 0.01 needs, for example, 7 records with fewer invalid attempts and none with
  more, 10 against 1, 12 against 2 or 14 against 3. On the test records at 0.1, M3 alone
  gave 18 against 1 in one field.
- The harm tests have low power. A one-sided p <= 0.05 needs 5 records against 0, 7
  against 1 or 9 against 2. On the test records, replay matches lie in 13 of the 50 records
  in field 20261030. So the replay matches could fall in up to 4 records, with none rising,
  without (H). The record bootstrap intervals of the change in matched replay attempts and
  in match@20, reported below, show the size that the tests cannot exclude.

Expectation, stated before the run. The validation records hold an unknown number of
near-prone covers, and the selector's different behaviour there may change it. If they
resemble the test records, invalid replay attempts fall by more than half in every field,
and matches do not change detectably.

Reported, not part of the decision:
- per field: replay structurally valid, without structure, matched attempts, match@20,
  match@1, consensus and the mean best RMS of matched records; oracle valid, matched draws
  and records with a match;
- the change in matched replay attempts and in match@20, summed over the fields, with
  record bootstrap 95 % intervals;
- pooled replay pairs, fixed and broken, with McNemar p, as a description only;
- draws: the fraction redrawn, the mean and the maximum, the records holding a redrawn
  replay attempt, and the attempts that failed at the cap or at the image search;
- invalidity by the d0 of the first draw and by the d0 of the integrated start, with
  unresolved starts in their own bin;
- sensitivity to the validity distance: invalid replay attempts at 0.6, 0.7 and 0.8
  angstrom per field, each with the record sign test. M3's ratio rose with the distance,
  from 0.368 at 0.5 to 0.667 at 0.8 angstrom;
- the shift of the starts and the structures: over redrawn attempts, the median ratio of
  the start volume per atom (used over first); over redrawn replay attempts of the
  confirmation, the median ratio of the decoded volume per atom (treated over untreated);
  and the median decoded shortest distance of redrawn against kept replay attempts;
- the selection step table; per step, every record holding a discordant pair with its
  fixed, broken, gained and lost counts, and the record whose removal sets the robust p;
  the draws of every selection run;
- the request errors of every confirmation run, with their messages;
- a record bootstrap 95 % interval of the pooled ratio of invalid replay attempts (2000
  resamples, `random.Random(20261116)`);
- kept attempts whose outcome differs, in cases with a failed padded solve;
- wall time against the untreated run.

An error in a reported line is printed and does not hide the selection or the decision.

### Checks, each a stop
The analysis `ab_carrier/m4_analysis.py` recomputes them from the raw files, not from the
run's own counts.
- Provenance:
  - every M4 file has the registered schema, script sha256, replay-module sha256 (M2's) and
    code fingerprint;
  - field and cover checkpoints, protocol seed, counts and record order are the registered
    ones;
  - the stored replay, M2 and M3 are the registered files, and each run used the
    registered references;
  - the analysis refuses to run while its hash pins are unset, and the queue checks the
    hash of every deployed file before it starts.
- Covers and starts:
  - every attempt has its reference's cover;
  - an attempt is redrawn exactly when its reference first draw has d0 <= T;
  - an attempt not redrawn has the reference's start block exactly. In a case with a
    failed padded solve, in the run or in its reference, the start is rebuilt in a smaller
    solve, and batched kernels need not be bitwise equal across batch sizes. There it must
    agree to 1e-5 relative in d0 and volume per atom, and the difference is counted;
  - every integrated start has d0 > T;
  - the unrounded d0 agrees with the radius-4 block;
  - no untreated run redraws.
- Outcomes and solves:
  - An attempt not redrawn has the reference's outcome exactly, unless its case had a
    failed padded solve in the run or in its reference. In the selection this reproduces
    the stored replay on the attempts whose first draw is far: 1849, 1746, 1635 and 1432 of
    2000 at 0.1, 0.2, 0.3 and 0.5.
  - Request errors are those of the reference.
  - An unsolved attempt is allowed in four cases: a failed padded solve in its case, a
    request error, a reference attempt that is unsolved with the same failed stage, or, in
    an untreated run without a reference, no structure. The last is a failure before any
    padded solve, for example a cover the bridge cannot expand.
  - A fallback solve is allowed only in a case with a failed padded solve, or where the
    reference had one.
- Fields: in the confirmation, the untreated runs of the three fields, and the treated runs,
  have equal covers, draw counts, start blocks and unrounded d0 for every attempt solved in
  both. In a case with a failed padded solve in either run, starts and d0 agree to 1e-5
  relative.
- Smoke 1, untreated, on the first 3 test records: every protocol field equals the stored
  replay, every start equals M2's, and nothing is redrawn or solved alone.
- Smoke 2, at 0.1 on the same records: cases, draw counts and integrated starts equal M3's.
- Smoke 3, untreated, on the first 3 validation records, so the validation path runs before
  any long run.
- The selection run at 0.1 equals M3 in all 50 cases.
The verdict repeats every check, the smokes included.

### What M4 cannot show
- One model: the pilot, trained on 404 records.
- The reduced-formula CSP track only: no primitive-count track and no de novo generation.
- The likelihood rule is not evaluated.
- The selection uses one field and records that M1 to M3 had already used.
- Any later CSP figure of the option on the test records is in sample. After M4 neither
  pilot partition is clean for this option, because the validation records are spent on
  this one confirmation.
- The validation records selected the checkpoints of the selector and of each field.
- The confirmation does not compare T* with smaller thresholds on the validation records.
- Structure quality beyond validity, match and RMS is not measured.

Cost: 3 smoke runs of 3 records, 2 to 4 selection runs and 6 confirmation runs, each about
20 to 35 minutes on tcm-pc10. The queue `diagnostics/pc10/m4_queue.sh` makes the three
untreated validation runs before the three treated ones. It waits for 6 GB of free GPU
memory before every run, checks its STOP file, holds a lock so that a second instance
exits, and starts no other queue. At the end it runs the registered verdict once, on the
CPU, into `r3/ab_carrier/m4/m4_verdict.txt`.

### Changed before the run
A first draft of this section (written earlier on 2026-11-16) chose the threshold on the
validation records and confirmed it on the test records. The first pre-run review (session
workflow `wf_622cb53f-88b`) checked the option, the harness and the design. Its findings:
- The test records are the records of M1 to M3. At 0.1, one confirmation field would have
  been M3 itself. The owner then chose to swap the partitions, as above.
- Pooling three fields that share every start as independent pairs overstates the
  evidence. The record is now the unit of the confirmation.
- Oracle gains could hide a loss of CSP matches. A replay-only harm test was added.
- Kept attempts whose outcome differs now stop the analysis. Before, they were only
  reported.
- The option divided s by the domain's reference length, although the expanded cell is
  already in angstrom. This did not affect the registered domain, whose reference length is
  1. It is fixed, with a test.
- An unsolved reference attempt no longer stops the checks or crashes the run.
- Unresolved blocks of untreated runs now get an exact search at 0.5 angstrom.
- The analysis now fails closed while unpinned, and checks the replay-module hash and the
  code fingerprint of all files.
- Unresolved starts have their own bin, and capped attempts are counted, not failed solves.
- Shift reports were added, and the 'cannot show' list was extended.
- Several sentences were made exact: the d0 rounding window, the role of the rounds,
  shared redraws, the density on A, and the cap's effect on the law.
The tests of the cap and of the search were made discriminating.

A second pre-run review (session workflow `wf_133de958-24b`) checked the revised option,
harness, queue and design, with a completeness critic. It found no major defect. Changed:
- the code fingerprint is pinned, so the files are tied to the M1 to M3 code;
- the kept-outcome exemption also counts failed padded solves of the reference;
- the shift report no longer crashes on an unsolved reference;
- the selection step also requires no replay harm, and a gain that survives leaving out any
  one record;
- harm calibration was added, and (S) and (N) were reworded;
- sensitivity to the validity distance was added;
- the observation is guarded, so a fault ends the run;
- the verdict repeats the smoke checks in full, and compares record ids across the six
  validation files;
- the unrounded d0 is recorded and used for the stops;
- untreated failures before any padded solve are allowed when they leave no structure;
- a validation smoke was added, and the untreated validation runs come first;
- the queue's snapshot diff reads diff's error output;
- the text was made exact: sibling equality per run and sampler, the log density, the
  image-search event in the law, a fourth difference from M3, the validity distance, the
  M3 qualifier, the not-out-of-sample items, and the use of the validation records for
  checkpoint selection.

A third, focused review (session workflow `wf_5b14c26b-55f`) checked the fixes after the
second review, the registration against the code line by line, and the queue in an end to
end dry run with stubbed runs and the real analysis. It found no major defect, and every
number in this section recomputed. Changed:
- the step table names the records holding discordant pairs and the record that sets the
  robust p;
- a kept start rebuilt in a smaller solve after a failed padded solve may differ in the
  last bits on the GPU; there it must agree to 1e-5 relative, and the difference is counted
  instead of stopping the run;
- request errors are printed with their messages;
- the selection's reported lines are guarded like the rest;
- the d0 bins use the unrounded d0 for an unresolved block and count a d0 of exactly 0;
- the queue holds a lock, and runs the verdict into a file at the end;
- the text was made exact about the rounding window, the 10c6e7e8 cover, the decoded-volume
  ratio, and an unresolved exact search in an untreated run.

Final sha256, registered before the run:
- the option: `src/crystalatlas/learned.py` `8db353f0...0f56`, tests
  `tests/test_start_rejection.py` `e5c95719...525e`;
- `diagnostics/m4_rejection.py` `6a7984de...cc27`, which imports `m1_contacts.py`
  `1f8332a8...a432` and `m2_starts.py` `5adfad80...0bd9`;
- `diagnostics/ab_carrier/m4_analysis.py` `5ac70f2e...e186`, which pins `m4_rejection.py`
  and the code fingerprint `941fbdd1...74b1`. That fingerprint is the snapshot of M1 to M3
  (`102a0b14...f553`, as recorded by M2) with this `learned.py`;
- `diagnostics/pc10/m4_queue.sh` `de310849...72cc`, which pins `learned.py`, the two M4
  scripts and the two imported scripts; `diagnostics/pc10/common.sh` `f68b096e...8dac`.

Inputs: M2 result `60ccf1fe...f844`, M3 result `b5fd5622...9318`, stored replay of field
20261030 `d2d6c00c...9075`, replay module `a5c825bd...3436`.

M4 launch record (2026-11-16). The files on the host equal the final hashes
(`r3/v7_eval_20261116_code/src/crystalatlas/learned.py`, `r3/ab_carrier/m4/`,
`r3/pc10/m4_queue.sh`, `r3/pc10/common.sh`), and so do the inputs. On the host the old
snapshot has the code fingerprint `102a0b14...f553` and the new one `941fbdd1...74b1`; they
differ only in `learned.py`. The queue started on tcm-pc10 at 06:40:45 UTC
(`r3/ab_carrier/m4/queue.log`), with 15855 MiB of GPU memory free and no other queue of
ours running.
The three smoke checks passed: smoke 1 at 06:44:34 UTC (120 attempts equal to the stored
replay and M2), smoke 2 at 06:48:19 UTC (cases, draws and starts equal to M3; 21 redrawn)
and smoke 3 at 06:51:21 UTC (120 validation attempts solved, none redrawn). The selection
started after them. Cases took 15 to 46 seconds each.

M4 selection, read from the queue's analysis log on 2026-11-16 (the registered verdict
repeats it). Every selection run passed its checks: 151, 254 and 365 attempts redrawn at
0.1, 0.2 and 0.3, and the kept attempts (1849, 1746, 1635) equal the stored replay. No
padded solve failed and no attempt was unsolved. The run at 0.1 equals M3 in all 50
cases.

| step | invalid of 2000 | fixed / broken | p_gain | robust p_gain | matched | gained / lost | replay gained / lost | accepted |
|---|---|---|---|---|---|---|---|---|
| none -> 0.1 | 69 -> 36 | 37 / 4 | 5.1e-08 | 7.0e-07 | 525 -> 534 | 10 / 1 | 0 / 1 | yes |
| 0.1 -> 0.2 | 36 -> 18 | 20 / 2 | 6.1e-05 | 0.011 | 534 -> 539 | 7 / 2 | 0 / 0 | yes |
| 0.2 -> 0.3 | 18 -> 11 | 10 / 3 | 0.046 | 0.113 | 539 -> 547 | 8 / 0 | 0 / 0 | no |

Selected threshold: 0.2 angstrom. The step to 0.3 passes the pooled gain (0.046) but not
the robust condition. Left out, record 10c6e7e8 leaves too little gain, so the rule stops at
0.2. Every robust p is set by that record. The confirmation on the validation records
started at 07:56:33 UTC.

## Result of measurement M4 (read 2026-11-16)
The queue ran from 06:40:45 to 09:50:34 UTC on tcm-pc10. It had no failure, no STOP and no
wait for GPU memory. It wrote the registered verdict with the registered analysis:
`diagnostics/ab_carrier/m4_verdict_2026-11-16.txt` (sha256 `0dd48f12...969d`). The result
files are in `r3/ab_carrier/m4/`; the three untreated validation runs have sha256
`db079dd9...`, `af628179...` and `7931d285...`, the three treated runs `610d0b2e...`,
`6da2efc6...` and `203d4f25...`.

Validity. Every registered stop passed:
- all 12 M4 files carry the registered schema, script sha256, code fingerprint
  `941fbdd1...74b1`, replay module, checkpoints, protocol seed and references;
- the three smokes passed, and the selection run at 0.1 equals M3 in all 50 cases;
- the redrawn attempts are exactly those whose first draw has d0 <= T. The kept attempts
  equal their reference in start and outcome, bitwise: 1849, 1746 and 1635 in the selection,
  1711 per field in the confirmation. No integrated start has d0 <= T;
- no padded solve failed, no fallback solve ran, no attempt reached the draw cap, no search
  was unresolved, and there was no request error;
- the three fields share covers, draw counts and starts in all 2000 attempts, untreated and
  treated.

Selection on the test records: 0.2 angstrom (table in the selection record above).

Registered decision: (S). Rejecting starts with d0 <= 0.2 angstrom is supported as an opt-in
sampler setting. No harm to matches was detected, at the power of the harm tests.

| field | replay invalid, untreated -> 0.2 | fixed / broken | oracle matched draws | replay matched attempts | match@20 |
|---|---|---|---|---|---|
| 20261030 | 58 -> 25 | 38 / 5 | 347 -> 356 | 59 -> 59 | 10 -> 10 |
| 20261031 | 51 -> 29 | 30 / 8 | 309 -> 318 | 56 -> 56 | 8 -> 8 |
| 20261032 | 65 -> 26 | 44 / 5 | 359 -> 370 | 53 -> 55 | 8 -> 8 |

Record sign tests, summed over the fields:
- validity: 27 records with fewer invalid replay attempts, 0 with more; p_gain = 2^-27 =
  7.45e-09;
- matches: 3 records with a net loss, 11 with a net gain; p_harm 0.994;
- replay matches: 0 against 1; p_harm_replay 1;
- fewer invalid replay attempts in every field: yes.

Reported with it:
- pooled replay invalid 174 -> 80, ratio 0.460, record bootstrap 95 % 0.312 to 0.593;
- other validity distances, summed over the fields: 223 -> 107 at 0.6 angstrom, 271 -> 157 at
  0.7 and 338 -> 219 at 0.8; record sign tests 29 / 0, 31 / 0 and 32 / 1;
- per-field record sign tests at 0.5 angstrom: 18 / 1, 14 / 2 and 22 / 0;
- draws: 289 of 2000 attempts redrawn in each field (14.4 %), mean 1.333 draws, at most 78,
  none at the cap; wall time treated over untreated 1.007, 1.005 and 0.999;
- replay matched attempts 168 -> 170 and match@20 26 -> 26 over the fields; match@1,
  consensus and best RMS unchanged; oracle matched draws 1015 -> 1044 of 3000, records with
  an oracle match 105 -> 110;
- shift over redrawn attempts: start volume per atom, used over first, median 1.551; decoded
  volume per atom, treated over untreated, 1.040, 1.045 and 1.030; median decoded shortest
  distance 1.27 to 1.39 angstrom for redrawn against 1.85 to 1.88 for kept replay attempts.

### Independent checks
Session workflow `wf_0dbc6e68-b54`, read-only on local copies of the result files (hashes
equal the host's); scripts and returned results in `diagnostics/ab_carrier/m4_checks/`.
- A blind recomputation from the registration and the raw files, made before reading the
  analysis or the verdict, matches every number of the verdict, and no registered stop fires.
- An audit re-ran the registered verdict and every queue analysis mode. Both outputs are
  byte-identical to the queue's. Per-file provenance is clean, all 459 request seeds
  recompute, and every result file was written inside its own run's time window. Other
  results:
  - The run at 0.1 equals M3 in all 2000 attempts.
  - The closest M2 first draw to a threshold is 2.16e-4 angstrom away.
  - The largest gap between the 7-digit r and the unrounded d0 is 4.7e-7 relative.
  - The code fingerprint of the working tree differs from the snapshot's, because the
    working tree's `config.py` and `invariant_transformer.py` differ from the snapshot in
    docstrings. The fingerprint 941fbdd1 was computed on the host from the snapshot
    itself (launch record). The second pre-run review also ran the 17 option tests on a
    copy of the snapshot with this `learned.py`, and all 17 pass.
- A skeptic found no error. Its findings change how (S) is read and are stated below.

### What the result shows and what it does not
- **Validity.** Out of sample, the option removes about half of the invalid CSP attempts.
  The pooled ratio is 0.460 (95 % 0.31 to 0.59), and 27 records improve against none. The
  gain needs 21 of the 27 improving records removed before p_gain exceeds 0.01. No cover
  gains more than 1 attempt per field. It holds at every validity distance from 0.5 to
  0.8 angstrom.
- **Effect size.** The effect is about half the in-sample one. At 0.2 on the selection
  records the ratio was 0.205 (44 -> 9). Over the redrawn attempts alone, the validation
  ratio is 147 -> 53 = 0.361, about M3's R = 0.368 at 0.1, so the test-record advantage of
  0.2 looks inflated by being measured in sample. The expectation stated before the run,
  "more than half in every field", fails in field 20261031 (ratio 0.569). The bootstrap
  gives P(pooled ratio > 0.5) = 0.25.
- **No-structure failures.** 18 of the 94 removed invalid replay attempts were decode
  failures without a structure. All had a first draw with d0 <= 0.058 angstrom.
- **The remaining failures are the same mechanism just above the new threshold.** 72 of
  the 80 remaining invalid replay attempts start at d0 <= 0.5 angstrom, 40 of them in
  (0.2, 0.3], and 79 of 80 are in covers with two orbits on one shared axis. Redrawn starts
  stay about 10 times more often invalid than kept first draws: 53 of 507 against 27 of
  2493. In shared-axis covers, starts that land in (0.2, 0.3] are invalid more often when
  redrawn than when they are first draws: 27 of 114 against 8 of 105. This repeats M3's
  finding on fresh records: near-prone covers keep extra hazard at the same d0. 27 of the
  80 are kept first draws that this option cannot change; record 6e984830 alone holds 19.
- **Accuracy.** (S) is a validity result. CSP accuracy does not change: replay matches go
  168 -> 170, and match@20, match@1, consensus and RMS are identical. The replay harm test
  can hardly see a loss by construction: only 3 of the 169 redrawn replay attempts per
  field are on the target space group, and the untreated redrawn attempts held 4 matches.
  "No harm detected" means a small change bounded by design.
- **Oracle gain.** Given the true cover, oracle matches rise by 29 of 3000 draws; the
  record bootstrap of the net gain is +4 to +56. That is the rate of far draws of the same
  records (54 matches in the redrawn draws against 56.2 predicted), as in M3. It is not a
  registered test, and two records carry 14 of the 29.
- **The option shifts starts more than structures.** Accepted starts have cells about 1.55
  times larger in volume per atom. Decoded structures are only 3 to 4 % larger in the
  median, with no target volumes to compare against.
- **0.2 is confirmed against no treatment only, not against 0.1 or 0.3.** On the test
  records the step to 0.3 fails the robust condition. Leaving out any one of 8 of the 11
  records with a validity change breaks it, not only 10c6e7e8. Its 8 oracle gains with
  none lost lie in 5 records. A claim that 0.3 is better is untested, and no clean pilot
  records remain to test it on.
- **The validation records differ from the test records.** They hold more near-prone
  replay covers (329 shared-axis against 276) and fewer attempts on the target space group
  (72 against 90). Their untreated match rates are lower: oracle 347 against 443 and replay
  59 against 82 in field 20261030. They selected the checkpoints with the unconditioned
  base, which cannot favour the treatment, so the paired difference is unlikely to be
  biased; absolute CSP figures on them are not clean.
- **Shared starts.** The three fields share every start, so they are not replicates. The
  record is the unit, and the every-field condition is a consistency check.

Consequence. By the registered rule, `start_min_separation = 0.2` is supported as an opt-in
sampler setting. It stays opt-in, and a default or its use in run 3 is the owner's
decision. After M4 both pilot partitions are spent for this option. Comparing 0.2 with 0.1
or 0.3 needs new records, for example the full MP-20 corpus, which waits on the rights
declaration. The remaining failures are the near-start mechanism in shared-axis covers, now
just above the threshold. Whether training with the same conditioned base removes them is
the open question of the training-side option; M4 does not test it.
