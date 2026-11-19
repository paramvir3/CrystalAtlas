# Amendment A-F10L: the learned half of F10 (approved 2026-10-26)

**Status: approved by the owner on 2026-10-26.**  The evaluation script is
`artifacts/f10_flow_oracle_2026-10-19/evaluate_heldout_f10.py`.  It refuses to
run unless this status line reads "approved" and the file's SHA-256 matches.
Theory §4.10 allows tolerances to change only through a versioned amendment
made *before* the affected experiment runs.
The held-out toys E1 and E2 were not run with a learned field before this
approval.  Everything measured before it came from the development toy and
served only as calibration.

## 1. Scope

F10 (theory 04, "Tiny-distribution oracle") compares a learned flow with a law
that is known exactly.  Its deterministic half passed with the oracle's own
velocity: density L¹ is 2.3e-8 and relative moment errors are ≤ 1.4e-7, against
the §4.10 bound of 1e-3.  This amendment covers the learned half: the production
geometry field, trained with the production CFM objective on one fixed cover, is
compared with the exact law of that objective.

The §4.10 deterministic-quadrature bounds (1e-3) do not apply here.  They bound
the numerical error of an exact pipeline.  A learned field carries a statistical
approximation error, which needs its own predeclared bounds (§6).

## 2. Target law

The objective uses `geometry_residual_gauge = "normalizer-v3"`.  Each path draws
its residual action uniformly from the cover's certified group: per-orbit
in-group gauges, identical-orbit relabellings, and every Euclidean-normalizer
automorphism, proper and improper.  The comparison law is the exact oracle
carrying the same group (`toy_oracle(model, toy, layout_gauge=True)`).

Reason (owner decision 2026-10-25): an equivariant field stagnates at achiral
configurations, because fixed-point subspaces of equivariant vector fields are
flow-invariant.  It therefore cannot move probability between enantiomorphic
sectors of a chiral cover.  On the development toy the chiral (normalizer-v2)
target forces a torus TV of at least 0.031 at any training budget
(`artifacts/f10_flow_oracle_2026-10-19/symmetry_barriers_dev.json`).  The racemic
target has equal sector masses and is representable.

## 3. Architecture under test

The subject is the `ModelConfig` geometry defaults at production capacity
(8.9 M field parameters): `invariant_global`, the `truncated-v1` lattice head,
and no radial pair widths.  The `DomainConfig` defaults apply, including the
`loewner-v2` chart.  The only option changed is
`geometry_residual_gauge = "normalizer-v3"` (§2).

The development comparison, at 3000 steps and batch 256, gave a z-marginal TV
of 0.1785 for these defaults.  The review-M2 radial widths (0.5, 1, 2 Å) gave
0.171, and radial widths with the commutant-v2 head gave 0.176.  Neither option
is therefore needed to pass, so neither is part of the subject.  Any other
configuration needs its own F10 run under this amendment.  The reason is
measured: the reducible excess E‖v_θ − v*‖²_H is at most 0.95% of the
per-sample CFM loss.  The error is limited by the gradient signal-to-noise
ratio, not by features (`loss_decomposition_dev.{py,log}`).

## 4. Training protocol

* Adam with a cosine decay of the learning rate from 1e-3 to 1e-5 over
  S = 30 000 steps at batch B = 256, i.e. 7.68 M path samples.  On the
  development toy the z-marginal TV fell with the sample count N as about
  N^−0.54: 0.1785, 0.108 and 0.0515 at 0.77 M, 2.56 M and 7.68 M.  At 10.24 M
  (10 000 × 1024) it was 0.0507, so the error depends on N rather than on how it
  is split between steps and batch
  (`artifacts/f10_flow_oracle_2026-10-19/budget_scan_v3/`).
* float32 on CUDA.  On the trained development field the float32/float64 gap in
  endpoint L¹ is 1.2e-6, and `tests/test_flow_oracle_learned.py` checks the
  device plumbing at float64 on both CPU and CUDA.
* Training seeds 0, 1, 2.  Every pass criterion applies to the worst seed.
* Path times are uniform on (0, 1), as in `train_on_toy`.

## 5. Measurements

**Primary: endpoint law by forward Monte Carlo.**  N = 32768 base draws (draw
seed 3) are integrated through the production RK4 flow (`NumericsConfig().ode_steps`
= 64 steps).  The same draws are also integrated with the oracle velocity.  That
control fixes the estimator floor, which on the development toy is z-TV 0.005
and a Fourier error of at most about 2 standard errors.  The reported quantities
are:

1. **z-marginal total variation** against the exact record mixture, on bins of
   width 0.25 over [min z − 6σ_z, max z + 6σ_z];
2. **torus Fourier moments** |E e^{2πi n·a} − exact|.  For d_η = 1 these are
   n = 1, …, 8 (the development calibration measured n = 1, 2, 3, 4, 8).  For
   d_η = 2 they are every integer n with |n|_∞ ≤ 4, except 0;
3. **z moments**: the relative errors of E z and E z²;
4. **joint moments** |E[z e^{2πi n·a}] − exact| for |n|_∞ ≤ 2, n ≠ 0.  These
   are reported without a threshold, because the development calibration did not
   measure them;
5. **mass outside the window**.

Forward Monte Carlo replaces deterministic quadrature as the primary estimator.
On the development toy, the quadrature grid (z step 0.25, 16 torus points)
reported only 0.855 of the learned flow's mass inside the window, while Monte
Carlo found 0.9994.  A learned density with sharp torus structure makes that grid
unreliable.  Monte Carlo is unbiased, its standard error is known, and the
control measures its floor.

**Diagnostic (no threshold): per-time velocity error.**  ρ_v at
t ∈ {0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99}, pooled and split into its lattice and
torus parts, with 256 production path samples per time.  It is in the production
loss metric.  The pooled 0.1–0.9 average hides component and late-time failures
on the development toy, so it is not a pass criterion.

## 6. Pass thresholds

A toy passes when, for each of the three seeds, all of the following hold on
the N = 32768 forward Monte Carlo of §5:

| quantity | threshold | development toy (seed 0, §4 budget) | estimator floor (oracle control) |
|---|---|---|---|
| z-marginal TV | ≤ 0.08 | 0.0515 | 0.0052 |
| relative error of E z and of E z² | ≤ 0.03 each | 0.011 and 0.006 | — |
| every declared torus Fourier error | ≤ 0.025 (4.5 standard errors) | ≤ 0.0079 | ≤ 0.0069 |
| mass outside the window | ≤ 0.005 | 0 | 0 |

How the values were chosen:
* The TV and moment thresholds sit 1.5 times (TV) to 5 times (E z²) above the
  development values.  That margin is for the held-out toys and for seed
  variation, which the single development seed did not measure.
* The TV threshold sits 15 times above the Monte Carlo floor, so estimator noise
  alone cannot cause a failure.  It is also far below the values of defective
  fields: the under-trained development field (3000 steps, TV 0.1785) fails it.
* The Fourier threshold is 4.5 standard errors (1/√N = 0.0055).  E2 tests 80
  modes (40 independent, since n and −n are conjugate), and the chance that
  noise alone breaks one is below 3e-4.  The barrier-limited chiral field broke
  it on the development toy (n = 8: 0.091).
* The deterministic-quadrature bound of §4.10 (1e-3) stays in force for the
  exact pipeline.  It does not apply to learned fields (§1).

**Estimator check on the held-out toys (oracle only, before approval).**  The
§5 estimator was applied with the exact oracle velocity on E1 and E2, at the
full N and with no learned field.  It passes every threshold.  E1 gives z-TV
0.0048 and a largest Fourier error of 0.0069.  E2 gives z-TV 0.0049 and a
largest error of 0.0119 over its 80 modes
(`evaluate_heldout_f10.py --control-only`;
`artifacts/f10_flow_oracle_2026-10-19/budget_scan_v3/control_E{1,2}.json`).

A failure is reported as measured.  The thresholds may not be relaxed after E1
or E2 has run.

## 7. Held-out toys (fixed by this amendment)

Both use a one-dimensional lattice chart (cubic) under `loewner-v2`, uniform
record weights, and carbon (Z = 6) and oxygen (Z = 8).  The lattice latents z
are chart coordinates (`layout.chart.decode_latent([z])` gives the metric).

* **E1, centrosymmetric, one orbit.**  Pa-3 (No. 205), 8c (x, x, x), carbon,
  ν = 8, d_η = 1.  Records (z, x) = (−0.5, 2/25), (0.3, 9/50), (1.0, 37/100);
  σ_z = 0.2, σ_η = 0.12.  The certified group has order 4, {±x, 1/2 ± x}.
  normalizer-v3 equals v2 here, because the group contains the inversion.  E1
  tests the lattice fit and the torus fit apart from chirality.
* **E2, chiral, two orbits.**  P2_13 (No. 198), 4a carbon (x₁) and 4a oxygen (x₂),
  ν = 4, d_η = 2.  Records (z, x₁, x₂) = (−0.4, 11/100, 63/100),
  (0.35, 29/100, 43/50), (1.1, 19/50, 7/100); σ_z = 0.25, σ_η = 0.18.  The
  certified racemic group has order 8; the proper part has order 4.  E2 tests a
  joint torus law, cross-orbit coupling and racemization.

**Oracle admissibility, checked before any run.**  The exact oracle certifies
a relative error bound of at most 1e-9.  The bound is the Fourier tail plus
summation rounding, divided by the heat-kernel density floor.  The smallest
admitted scales are σ_η = 0.105 for E1 (1-D torus) and 0.17 for E2 (2-D torus,
where the larger covering radius lowers the floor).  The first draft declared
σ_η = 0.1 and 0.125, which the oracle refuses.  The declared values carry a
margin: σ_η = 0.12 gives a bound of 5.4e-11 and σ_η = 0.18 gives 1.5e-10.  A
larger σ_η makes the torus law smoother.  That is the price of an exactly
certified oracle on a 2-D torus.

**Representability check, done before any run.**  An untrained equivariant field
has structural velocity zeros, which hold at every z and t.  On E1 they sit at
x = 0, 1/4, 1/2, 3/4.  These are exactly the fixed points of the gauge
reflections −x and 1/2 − x, where the exact law's velocity vanishes too.  On E2
they are isolated points of the 2-D torus, for example (0, 0) and (1/4, 1/2).
A codimension-2 point cannot separate the torus.  Neither toy therefore has a
barrier that the racemic target leaves
(`artifacts/f10_flow_oracle_2026-10-19/check_heldout_representability.{py,log}`).

The development toy (P2_13 4a carbon, z = (−0.3, 0.5, 1.2),
x = (3/25, 3/10, 41/100), σ_z = 0.25, σ_η = 0.125) is used for calibration only.
