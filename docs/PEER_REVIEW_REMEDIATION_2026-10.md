# October 2026 peer-review remediation

This record supplements, and does not rewrite, the
[review of 2026-10-19](../artifacts/peer_review_2026-10-19/REVIEW.md) and its
reproductions. A passing regression means the specified failure is addressed.
It does not mean every downstream claim has been re-verified. The record is
updated as findings are closed.

## Status

| Finding | Status | Correction | Evidence |
|---|---|---|---|
| B4: portability gate demanded bit equality of a derived float | Closed | Stored bytes stay authenticated exactly: the target fingerprint is recomputed over the *stored* latent bytes. The locally rebuilt latent must agree within `DERIVED_LATENT_RELATIVE_TOLERANCE = 1e-12 × max(1, ‖z‖∞)` | `tests/test_training_data.py`: 4e-16 cross-stack rounding accepted, served batch equals stored bytes; 1e-9 deviation rejected; substitution attack still rejected. `verify_b4_cross_stack.log`: all 193 pilot records validate on numpy 2.4.6; 59 latents are not bit-identical, at most 1.9e-15 |
| B3: no physical structure matcher | Closed under the amended criterion (approved 2026-10-19; see *External agreement*). The original ≥99% pymatgen-agreement criterion was not met (93.85%) | `crystalatlas.physical_matching` (below) | `tests/test_physical_matching.py`: 26 tests plus a pymatgen adapter test, which passes in the adapter venv. `verify_b3_invariance.log`: 900 borderline trials, 0 representation-dependent decisions. `verify_b3_mutations.log`: 7 of 7 defects caught |
| M1 (F8 part): endpoint-weighting counterexample untested | Closed | — | `tests/test_flow_population_target.py` (6 tests). On analytic mixtures the production `conditional_flow_matching_loss` is minimized by `E[u\|q_t]` under current-state weighting, and by `(Σp W)^{-1} Σ p W u` under endpoint-dependent weighting |
| M1 (F8b and the deterministic half of F10) | Closed | `crystalatlas.flow_oracle.TinyFlowOracle`: the exact marginal density, current and velocity of the law the production objective samples on one cover. That law is the Haar base, the Gaussian–torus-heat endpoint, a uniform residual action applied after the half-open coordinate lift, and the linear path. The oracle certifies a per-component relative error bound (Fourier tail plus summation rounding over the density floor) and refuses ill-conditioned toys | `tests/test_flow_oracle.py` (14 tests). There are two toys, both with d_z = 1: one orbit parameter with the sign gauge, and a two-parameter skew basis with an action that is not a signed permutation. Test findings: the target equals an independent image-sum to ≤ 1.6e-14; the marginal and record-conditional continuity equations hold; production samples have the oracle law, and E[u − v(q_t)] f(q_t) = 0 within 5 SE. At the production 64 steps, RK4 matches DOP853 (rtol 1e-9, atol 1e-11) forward and backward to ≤ 3.0e-8 against the 1e-7 bound. The theory §4.10 deterministic-quadrature gate passes: density L¹ is 2.3e-8 and 3.6e-8, and relative moment errors are ≤ 2.7e-8 (12 moments) and ≤ 1.4e-7 (26 moments), against 1e-3. The pushforward density uses exact forward-mode AD through production RK4. Seeded mutations: dropping the action push gives endpoint L¹ = 0.52 (an in-suite test). In `verify_f10_mutations.log` and `verify_f10_lift_mutation.log`: (i) a dropped action push and a flipped current sign are each caught by both the continuity and the Monte Carlo tests; (ii) a production path without its action is caught by the Monte Carlo test, while the continuity test, which never touches production paths, cannot see it; (iii) a Cartesian nearest-image lift replacing eq. 1.32 is caught on the skew two-parameter toy, since in one dimension the two lifts coincide. Measured values: `artifacts/f10_flow_oracle_2026-10-19/deterministic_gate.json` |
| M1 (learned half of F10) | Open with a split verdict: amendment A-F10L approved 2026-10-26. Held-out E1 FAILED (2 of 2 completed seeds miss the z-TV threshold). Held-out **E2 PASSED** on all three seeds (seed 0 on tcm-pc10 2026-10-26; seeds 1–2 on tcm-pc38 2026-10-29/30 under the same approved hash) | `crystalatlas.flow_oracle_learned` on a real cover: P2_13 (cubic, d_z = 1) with a 4a (x, x, x) orbit, d_η = 1, and residual group {I}. It trains the production field with the production batched CFM objective (`train_on_toy`). It measures ρ_v = E‖v_θ − v*‖²_H / E‖v*‖²_H against the exact oracle velocity, in the production loss metric H at q_t. It also measures the endpoint density L1 and moment errors by deterministic quadrature through the learned production RK4 flow. The Jacobian is reverse-mode, because the Loewner adjoint has no forward mode, and nodes are chunked, which is exact and bounds memory. **Cost:** profiling showed 77% of a training step spent rebuilding `model.layout(cover)` per record, with O(|G|³) closure re-validation. `CrystalAtlasModel.layout` now keeps an exact bounded per-model cache (4096 covers). The one tensor that aliased a layout array (`TorchCoverGeometry.period_basis`, via `torch.as_tensor`) is now a copy, so a shared layout cannot be written through a buffer. The production-capacity field still costs 0.82 s/step and about 116 min per endpoint quadrature on this CPU, against the review's estimate of minutes. The calibration therefore uses the same architecture at reduced capacity, declared as such | `tests/test_flow_oracle_learned.py` (6) checks: the oracle's own velocity gives ρ_v = 0 exactly, with endpoint L1 and moments ≤ 1e-3 on the real cover; the reverse-mode pushforward matches the forward-mode deterministic gate (rtol 1e-10); an untrained field gives ρ_v > 0.5 and L1 > 0.1; the objective trains; and the layout cache is exact and bounded, with no aliasing. Development-toy calibration: `artifacts/f10_flow_oracle_2026-10-19/calibrate_learned_f10.py`. **GPU diagnosis (2026-10-25, development toy only):** (1) The quadrature's 15% "mass outside the window" was a grid artifact. Forward Monte Carlo through the production RK4 flow finds 0.06% outside the window. Its oracle-velocity control on the same draws gives a z-TV floor of 0.005 at 32768 draws. (2) Achiral barrier: the field's torus velocity is identically 0 at the FCC points of 4a. The fix is normalizer-v3 (row below). (3) The z-marginal TV of about 0.17 is the same with legacy features (0.1785), with M2 radial widths (0.171), and with radial widths plus the commutant-v2 head (0.176). The learned v_z(z) stays affine even when the pooled features depend on z. A loss decomposition on production paths shows why: the reducible excess E‖v_θ − v*‖²_H is at most 0.95% of the per-sample loss (t = 0.7). The rest is target variance, so the limit is the gradient signal-to-noise ratio (`loss_decomposition_dev.{py,log}`). Rao–Blackwellizing the Gaussian lattice target reduces that noise only 1.6× at t = 0.7 (9–41× at t ≥ 0.9). The remainder is record-posterior uncertainty (`rao_blackwell_z_variance_dev.{py,log}`). **Budget scan (normalizer-v3, legacy features):** the z-marginal TV is 0.1785, 0.108 and 0.0515 at 0.77 M, 2.56 M and 7.68 M path samples, about N^−0.54. It is 0.0507 at 10.24 M (10 000 × 1024), so it depends on the sample count, not on steps against batch. At 30 000 × 256, E z is 1.1% low and E z² 0.6% low, the worst lattice ρ_v is 0.027, and every torus Fourier error is at Monte Carlo noise (≤ 0.0079, SE 0.0055). Evidence: `budget_scan_v3/`. Amendment A-F10L (`theory/amendments/F10_LEARNED_AMENDMENT.md`, approved by the owner on 2026-10-26) defines the held-out toys E1 (Pa-3 8c) and E2 (P2_13, two 4a orbits). It also records their representability check and the oracle-admissible dequantization scales (σ_η = 0.12 and 0.18; the first-declared 0.1 and 0.125 are refused by the exact oracle). It fixes the budget (30 000 × 256, seeds 0–2) and the thresholds: z-TV ≤ 0.08, z moments ≤ 3%, Fourier errors ≤ 0.025, mass outside the window ≤ 0.005. `evaluate_heldout_f10.py` implements the protocol and refuses to run until the amendment is approved and its hash matches. The held-out runs E1 and E2 started on 2026-10-26 under the approved hash `15297d13…5901` (`F10_LEARNED_AMENDMENT.sha256`). The owner cancelled them after 5 h 15 min to free tcm-pc10. Three of six seeds had completed (`heldout_a_f10l/heldout_E{1,2}.{json,log}`). **E1 FAILED under A-F10L:** every seed must pass, and seeds 0 and 1 both missed the z-marginal TV threshold of 0.08 (0.0947 and 0.0930; oracle control 0.0048). Seed 1 also missed the 3% E z bound (0.042). Both seeds passed the moment-squared, Fourier (≤ 0.0103) and window checks. The late-time lattice velocity error is about 0.11 at t = 0.9 for both seeds, against 0.027 on the development toy at the same budget. E1's narrower lattice modes (σ_z = 0.2) need more samples, which is the same signal-to-noise limit. **E2 PASSED (3 of 3 seeds):** z-marginal TV 0.0543 / 0.0539 / 0.0558 against 0.08 (oracle control 0.0049, Monte Carlo SE 0.0055); E z errors 0.03% / 1.23% / 0.57% and E z² errors 0.91% / 0.86% / 0.98% against 3%; every torus Fourier error within 0.025; mass outside the window 0 for every seed. Seeds 1–2 ran on 2026-10-29/30 on tcm-pc38 (5600 s and 5361 s of training each) with `evaluate_heldout_f10.py --seeds 1,2 --resume`, which keeps seed 0 from the 2026-10-26 run and records in the report that the kept seed ran on the earlier code state with identical amendment hash, thresholds, budget and declared options (`heldout_a_f10l/heldout_E2.json`, `heldout_E2_seeds12_pc38.log`). The per-time velocity error in the loss metric stays below 0.033 at every probed time; the torus-block relative error alone grows to 2–9 at t = 0.99 because the exact torus velocity vanishes there, which the endpoint Fourier checks (passed) are the protocol's measure of. No threshold was changed. **A-F10L overall: E1 failed, E2 passed; the learned half of F10 is therefore not closed, and E1's narrower lattice modes (σ_z = 0.2) remain the open case** |
| M3: lattice head writes into an unnamed basis | Closed; `commutant-v2` is the default since 2026-10-29 (review C3: `truncated-v1` slot 0 is the c-axis strain for tetragonal and the a-axis strain for hexagonal settings). `truncated-v1` is explicit-only and keeps its historical payload | `ModelConfig.geometry_lattice_head="commutant-v2"`: input `C b` and output `Cᵀ A C z + Cᵀ r` in canonical Sym(3) coordinates, where C is the chart commutant basis | `tests/test_lattice_head_commutant.py` (6 tests): equals the legacy head when C is the first d axes; exactly equivariant under `C → CQ`; linear-growth bound kept; end-to-end single and packed training steps. Legacy default and fingerprint unchanged |
| M2: attention sees no interatomic distance | Closed (opt-in); the E4 ablation gate and the learned 15 Å test need training | `ModelConfig.geometry_radial_gaussian_widths`: `ρ_w = Σ_n s(\|r + nL\|) exp(−\|r + nL\|²/2w²)` over every image within 6w, with a C¹ cutoff s | `tests/test_radial_pair_features.py` (5 tests): equals the certified image sum in a skew cell; rotation- and basis-invariant; resolves a 0.8 Å versus 2.5 Å contact in a 15 Å cell by > 10⁴; trains end to end. The legacy default and fingerprint are unchanged. **Robustness fix (2026-10-25):** the image box was refused whenever it exceeded `max_radial_images = 4096`. The CFM base reaches cells near the chart's 0.75 Å shortest-vector floor, so a 2 Å width (cutoff 12 Å) needs about 19³ images, and training would crash on a rare small cell. The sum is now taken over chunks of `max_radial_images` images (the memory bound) and is refused only beyond `max_radial_image_chunks = 64` chunks (the time bound). The default cap of 262 144 images covers the worst case of 35³ = 42 875 at the Loewner floor. Anything that fits in one chunk runs bit for bit as before. `test_small_cells_are_summed_in_bounded_chunks_exactly` checks chunks of 128 against the certified image sum in a 1.3 Å skew cell (rtol 1e-12) |
| M6: no primitive-count conditioning | Closed (separate CSP track) | `GenerationRequest(primitive_atom_count=N)`: (ν, G) masks enforce `ν\|c\| = I_G·N`. Tables whose generic hidden translation index exceeds 1 are redrawn (`cover_legality.generic_hidden_translation_index`) | `tests/test_primitive_count_conditioning.py` (4 tests), including body-centred P-1 (1a + 1h, index 2) versus generic 2i (index 1), and learned CSP samples meeting the constraint. Reported log-probabilities are those of the masked law; the acceptance normalizer of the redraw is not computed |
| M4: σ floor without an endpoint estimate | Mechanism closed; σ sweep needs GPU training | `NumericsConfig.ode_terminal_time`: integrate to t, then `x̂₁ = x_t + (1 − t) v` | `tests/test_endpoint_estimate.py`: `t = 1` reproduces RK4 bit for bit; the analytic single-target field is recovered exactly from t ∈ {0.5, 0.9, 0.99} |
| M5: effective batch size about one | Closed (opt-in) | `MixedTopologyBatchSampler` yields `MixedTrainingBatch` values: tuples of validated per-fibre `TrainingBatch` parts, so every exact per-topology invariant and audit still holds. Composition and cover losses are record-weighted means over the parts. Geometry is one padded dense call (`geometry_cfm_objective_padded` → `IntrinsicGeometryField.forward_padded`). Atom masks cover attention keys, partner means, vector sources and pooling. Each crystal's tangent solve is exact (real atoms only, its own coordinates, identity padding, per-crystal rank certificate). Padded velocities are zero and padded metrics are the identity. The legacy sampler and its fingerprint are unchanged | `tests/test_padded_batching.py`: padded backbone, field and objective equal independent per-crystal calls to ≤ 1e-12, across four topologies with d_z ∈ {1, 6, 2, 1} and d_η ∈ {1, 3, 1, 0}. `tests/test_mixed_topology_sampler.py` and `tests/test_mixed_training_objective.py` cover coverage, determinism, the pair budget, DDP balance, resume, and record weighting. On the 193-record pilot store (`artifacts/m5_padded_batching_2026-10-19/verify_m5_pilot.json`), median occupancy at batch size 32 is **1 → 32** for both measures: structure batches go from 132 to 7 and unique-composition batches from 131 to 6. The first real mixed batch spans 27 fibres and 18 groups, with d_z ∈ {1, 2, 3, 4, 6}, d_η from 0 to 51, and 4–40 atoms. Its per-sample geometry losses equal single-crystal objectives to 8.7e-15, and its cover and composition losses equal singleton batches to 1.7e-16 and exactly. On this CPU the padded call is not faster (38.3 s vs 38.4 s): per-crystal exact decoding and layout construction dominate. Accelerator utilization is not measured |
| M7: ranking quantum trade-off | Closed with evidence (decision 2026-10-20): certified interval pruning is not implemented, because the relative quantum removes the bin-wall rejections it targeted | `relative_metric_quantum` (B2) | `artifacts/m7_ranking_quantum_2026-10-19/` (τ = 1e-3 Å, 500 ranked records): legacy 236 accepted (228 bin walls, 36 resolution). With c = 1e-11, 1e-10, 1e-9: 426, 485, 498 accepted. **0 unresolved ties for every c up to 1e-7**; 2 resource-limit rejections remain at c ≥ 1e-9 |
| m4: decoder audits take minutes | Closed | Exact vectorized threshold tests (`periodic.periodic_pairs_within`, proof in docstring) and an exact vectorized minimum distance | `tests/test_decoder_audit_speed.py`: identical adjacency and containment to the exhaustive search. 40-atom Fm-3m decode 43.8 s → 0.13 s; FullPrim 1.1 s → 0.006 s |
| m5 (L1 and typing parts): L1 tested only for P1 and P-1; mypy on 5 files | Closed. The §4.9 mutation suite is in the next row (B3 has its own 7-item suite) | CI's strict mypy surface now also covers `cover_legality`, `physical_matching`, `physical_validity` and `symmetry_descriptors`, and a pre-existing `config.py` annotation error is fixed | 9 files pass `mypy --follow-imports=skip` strict. L1: `tests/test_lattice.py::test_l1_chart_dimension_matches_the_crystal_system_for_all_230_settings` |
| m5 (§4.9 mutation suite): no executable counterpart | Closed: 21 of 21 executable mutations killed, including mutation 12 under the soft-v2 F6 fix. Mutations 4, 8 and 9 stay listed gaps by decision (2026-10-20): their killing obligations (the D1b proper-null certificate and the D4 overlapping-cover oracle) are open theory items, not review findings | `tests/mutation_suite.py` is the frozen manifest. Each entry is an exact textual source mutation tied to its theory obligations and concrete killing tests. `scripts/run_mutation_suite.py` patches a temporary copy of `src/`. It requires each killer to pass unmutated, and a skipped or xfailed control counts as an error, never a pass. It then requires pytest test failures (exit code 1) on the mutant. `tests/test_mutation_suite_manifest.py` runs in CI and checks that numbering is exactly 1–24, every snippet occurs once, every mutant compiles, and every killer resolves | `artifacts/peer_review_2026-10-19/mutation_suite_run.{json,log}`: killed 1–3, 5–7, 10, 11, 13–24. Killing tests written because none existed: F1/F8b dequantization-kernel tail test (5); S3 masked-law contract for primitive-count conditioning (7); S3 DNG input provenance (11); F7 strongly skewed exhaustive image oracle (14), since the old oracle needed only \|n\| ≤ 2 and a fixed cube survived; F7 edge-budget overflow (17); E3 decode-failure accounting (18); C3/D1 rock-salt formula-unit test (19); D0b three-mode shared-path conformance (21, 24); D0b null-composition rejection (22). Building the suite also exposed the F6 cutoff defect below |
| m1: selector has thin descriptors | Closed (opt-in) | `ModelConfig.selector_symmetry_descriptors`. Group logits gain a bilinear term over 18 operation-derived descriptors: the point-group rotation-type histogram, log order, a centring one-hot, and polar and chiral flags. Rows gain the rotation-type histogram of their exact site stabilizer. `diagnostics.per_group_calibration` reports per-group calibration | `tests/test_symmetry_descriptors.py`: stabilizer order equals the catalogued site-symmetry order for every row of all 230 settings; the histograms separate exactly spglib's 32 point-group classes; flags verified; end-to-end sampling. `tests/test_group_calibration.py`. The legacy state dict and fingerprint are unchanged |
| m2: overlapping covers are moot in practice | Closed (documentation) | `docs/TRAINING_AND_DIAGNOSTICS.md`: training sees only maximal canonical covers | — |
| m3: validity reported only crystallographically; reference law only P1/P-1 | Closed, except charge neutrality | `physical_validity.py`: exact minimum interatomic distance including self-images, and the ≥ 0.5 Å rate | `tests/test_physical_validity.py`. Charge neutrality is not computed, because it needs a vetted oxidation-state table bound as a dependency. `controls.py` adds the uniform-geometry-in-true-cover control (`tests/test_uniform_cover_control.py`). The reference law now covers all 230 groups (2026-10-20). **Validity proof:** a pooled parameter's orbit has the row's declared multiplicity, so each of its points has exactly the row's site-symmetry group. Orbits of one row are therefore equal or disjoint, and orbits of different rows are disjoint because the Wyckoff positions partition the cell and the grids exclude specializations (not because of the stabilizers: Pm-3m 3c and 3d share a site group and are distinct orbits; corrected 2026-10-29, review m1). `reference.certify_disjoint_orbit_pools` checks this exactly for every pooled orbit, including fixed sites. **Sufficiency:** each row's grid denominator is the smallest odd q ≥ the declared base that supplies the capacity // multiplicity orbits a cover can need. Every feasible cover leaf is therefore realizable, with no retry. A row larger than the capacity is unoccupiable and gets no pool. The default P1/P-1 law is unchanged. Evidence: `tests/test_reference_all_groups.py` (10, eight diverse groups; exact audits of sampled crystals in Fm-3m, P6_3/mmc, R-3m, Pmmm, Ia-3d) and `artifacts/peer_review_2026-10-19/verify_reference_law_all_groups.{py,json,log}`. All 230 settings are certified at the default domain (1060 pooled rows, q ≤ 41) and the MP-20 domain (1056 rows, q ≤ 21) |
| m6: the composition training measure is not the theory's (3.15a) | Closed (opt-in) | `TrainingMeasure.SYSTEM_FORMULA_POLYMORPH` on the stored training path. Every record is visited once, with exact rational weights N p_i for cover and geometry and N P(c)/n(c) for composition (`training_data.system_formula_polymorph_weights`). S is the identity element-set key, c the composition key, and C the identity polymorph key (a reversible grouping). The objective accepts these batches only from `ObjectiveDataMeasure.from_dataset(dataset, search_measure=True)` and checks every batch weight exactly. Legacy measures and contracts are unchanged | `tests/test_search_measure.py` (10). Covers exact 3.15a weights on a hand-built toy; agreement with `data.record_probabilities` on 200 random records (rtol 1e-15); refusal of inconsistent keys; batch wiring; reduction to `STRUCTURE` on balanced polymorphs (bitwise); exact non-uniform weighting of the cover and geometry losses; refusal of undeclared weights; mixed-topology batches |
| m7: no power analysis | Closed | `diagnostics.paired_discordance_sample_size` | `tests/test_power_analysis.py`: δ = 0.05, p_d = 0.2, α = 0.05 at 80% power gives n = 628 |
| Minor: domain limits only in rejection strings | Closed | `docs/DATASET_PIPELINE.md` "Declared domain limits" (re-derived: ‖G‖ < 219.9 Å² and c/a factor e^{√6/2} = 3.40) | — |
| Minor: `apply_float` returned 1.0 | Closed | Endpoint 1.0 mapped to 0.0, as in `FloatingStructure` | `tests/test_algebra_apply_float.py` |
| F6 defect found while building the §4.9 mutation suite: the `equivariant_local` ablation is discontinuous at its cutoff | Closed (opt-in fix, approved 2026-10-20). The legacy normalization is unchanged and documented | `ModelConfig.geometry_local_vector_normalization = "soft-v2"` → `IntrinsicFieldConfig.local_vector_normalization` → `PeriodicTransformerConfig.vector_normalization`. The legacy `rms-v1` map N(v) = v/√(⟨v²⟩ + ε) has Jacobian I/√ε at v = 0. The production local backbone uses the transformer default ε = 1e-8, because the field does not forward its own 1e-4. Vector channels start at zero, so a cutoff-gated message ∝ χ(r) → 0 is renormalized to O(1) weight at each of the two normalizations per block. `soft-v2` uses N(v) = v/√(1 + ⟨v²⟩), with Jacobian eigenvalues (1 + r²/n)^(−3/2) and (1 + r²/n)^(−1/2), both ≤ 1. The map is therefore 1-Lipschitz, has Jacobian exactly I at 0, has output RMS below 1, and is O(3)-equivariant | `tests/test_equivariant_transformer.py`: unit properties (Jacobian at 0, Lipschitz bound, boundedness, O(3) equivariance) and SE(3) equivariance with chirality sensitivity under both normalizations. Gap across r_c ± δ for δ = 1e-3 to 1e-6: soft-v2 gives 2.44e-6, 2.44e-8, 2.44e-10, 2.44e-12, a ratio of exactly 100 per decade, i.e. C¹ with gap ∝ χ ∝ δ². Legacy rms-v1 gives 4–6 at every δ. Production-width field (6 layers): the velocity gap is > 0.1 with legacy (−6.83 → 0) and ≤ 1e-9 with soft-v2. The legacy continuity test stays `xfail(strict=True)` as documentation. Mutation 12 is now executable and killed |
| Residual-gauge gap found by the learned F10 run (2026-10-20) | Fixed (opt-in `ModelConfig.geometry_residual_gauge = "normalizer-v2"`; owner chose option (a)). The legacy `relabel-v1` stays the default and bitwise unchanged | **Gap:** the production CFM objective sampled residual actions only from identical-orbit relabellings and P-1 signs. Other parameter maps that leave the crystal unchanged, or map it onto a properly congruent copy, were not symmetrized. **Fix:** `crystalatlas.normalizer` computes the proper Euclidean normalizer of each setting for general metrics, modulo G and the continuous origin. Linear parts W preserve every metric the point group preserves. Translations come from an exact grid search with denominator 24 (completeness checked at 48), and every element is verified to conjugate G onto G exactly. `crystalatlas.residual_gauge` derives each cover's residual group exactly in raw Wyckoff parameters: per-orbit in-group gauges (x(p) → x(pA + a); P-1's sign is one), identical-orbit relabellings, and normalizer automorphisms (cross-row maps included). Each is certified on exact decoded point sets and conjugated to an affine torus map η → ηA + m. The objective samples it uniformly (`LearnedCrystalGenerator._sample_residual_torus_action`). `conditional_flow_path` and the exact oracle gained the translation part. **Measured:** P2_13's proper normalizer has index 4 (I4_132; full normalizer Ia-3d), acting on 4a as {x, x + 1/2, 1/4 − x, 3/4 − x}. The 2-image gauge first used in the F10 harness was therefore incomplete; the exact computation corrected it. Under normalizer-v2 the objective's own paths match the exact oracle carrying the full gauge (|z| < 4). Legacy paths deviate (z ≈ 8). The `invariant_global` field is exactly invariant under the translation gauge (1e-9), but only approximately under the rotation gauge, because its vector readout uses the pinned lattice rows as carriers (documented design). Symmetrizing the target is the physically correct law either way | `tests/test_normalizer.py` (19), `tests/test_residual_gauge.py` (12, including exact η-level congruence for the polar P2_1 cover), `tests/test_affine_torus_actions.py` (4: path, oracle against the record-duplication construction, closure, sampling), and `tests/test_residual_gauge_objective.py` (4). All-group certification: `artifacts/peer_review_2026-10-19/verify_residual_gauge_all_groups.{py,json,log}`. All 230 settings pass, with 0 failures in 725 s: every positive-dimensional Wyckoff row (1115 single-orbit covers) and every normalizer-related row pair (291 two-orbit covers, the cross-row slot map) is certified on exact point sets |
| Symmetry barrier found by the learned F10 run (2026-10-25): an equivariant field cannot move probability across achiral configurations of a chiral cover | Fixed (opt-in `ModelConfig.geometry_residual_gauge = "normalizer-v3"`; owner decision 2026-10-25: racemize within a cover). normalizer-v2 and relabel-v1 are unchanged | **Finding:** on the development toy (P2_13 4a), the 4a orbit is FCC, hence centrosymmetric, at x = 0, 1/4, 1/2, 3/4. There the learned torus velocity is identically zero, 1e-18 before and after training, while the exact normalizer-v2 velocity reaches 0.10. Fixed-point subspaces of an equivariant vector field are flow-invariant (Golubitsky–Stewart–Schaeffer), so the learned flow keeps the base mass 1/8 in each of the 8 sectors between these points and the proper-gauge fixed points 1/8 + k/4. The chiral target needs 0.1174/0.1326, which forces a torus TV of at least 0.031 at any budget. These points are fixed by improper normalizer elements: inversion at the origin normalizes P2_13 and maps the orbit at x to the orbit at −x. The finding matches Dumitrescu et al. (ICLR 2025): E(3)-equivariant generators cannot learn chirality. **Fix:** `normalizer.euclidean_normalizer` keeps det W = −1. `cover_residual_gauge(layout, improper=True)` certifies the improper maps on exact point sets. The objective and `toy_oracle(..., layout_gauge=True)` use them under normalizer-v3, so the target is racemic within each cover, and the 8 sectors carry equal target mass. **Theorem, checked on all 230 settings:** [N : N⁺] = 2 exactly for the 43 Sohncke types outside the 11 enantiomorphic pairs, and 1 otherwise. If G holds an improper operation, the cosets coincide. For the 22 enantiomorphic types, no improper isometry maps G onto itself. So normalizer-v3 differs from v2 only on those 43 types. Across covers, enantiomorphic types such as P3_1 and P3_2 stay distinct, and the selector learns their ratio | `tests/test_normalizer.py` (+21: table sizes, P2_13 index 8 with the inversion coset, v3 = v2 on centrosymmetric groups, the index rule on 13 representative types), `tests/test_residual_gauge.py` (+8: P2_13 4a racemic group {±x + k/4}, v3 = v2 on centrosymmetric covers, exact point-set congruence of every improper action on P2_1, P2_13, Pmmm, P-1), `tests/test_residual_gauge_objective.py` (+4: versioned option, uniform sampling over the order-8 group, objective paths match the racemic oracle, and a matched-filter test that separates v2 from v3 at z = 8.2). Evidence: `artifacts/f10_flow_oracle_2026-10-19/symmetry_barriers_dev.{py,json,log}` and `artifacts/peer_review_2026-10-19/verify_full_normalizer_all_groups.{py,json,log}` (230 settings, 43 of index 2, 211 racemic single-orbit covers and 79 cross-row pairs certified, 0 failures, 362 s) |
| GPU throughput of the learned path (2026-10-25, RTX 5060 Ti) | Closed (performance only; per-record numerics unchanged) | Profiling put almost all of a training step in per-record Python, not in the network. Four changes: `TorchCoverGeometry.forward_batch` and `lattice_pullback_metric_batch` (the Loewner closed form, batched); `reduce_cartesian_to_intrinsic_packed` and `translation_quotient_normal_matrix_packed` in the packed field (one Float64 solve per batch, with the same per-crystal rank and condition certificate and the unbatched messages, through one host sync); batched time features; and an exact read-only cache of encoded targets. The packed objective uses the batched expansion when all records share one cover. The padded reducer's certificate is vectorized the same way. **Measured:** a packed field call on 512 states went from 878 ms to 12.8 ms (float32). A production training step went from 0.267 s to 0.0385 s at batch 64 and from 0.87 s to 0.080 s at batch 256 (reduced field); at production capacity it takes 0.058 s and 0.099 s | `tests/test_nn_bridge.py` (+8: batched equals per-record expansion, pullback and gradients for both charts and dtypes, including at repeated eigenvalues; a mutation of the batched adjoint is caught), `tests/test_nn_field.py` (+4: packed reduction and metric equal per crystal; the first uncertified crystal is refused with the per-crystal message; batched time features), `tests/test_objective_caches.py` (+2: the encoded-target cache is read-only; a single-cover batch uses one batched expansion, equal to per-record within 1e-12). Remote run on tcm-pc10 (CUDA): 252 passed, 1 xfailed across the 17 affected modules. Full suite on the final code (CUDA, `CUBLAS_WORKSPACE_CONFIG=:4096:8`): 1305 passed, 2 skipped (CUDA-host benchmark guard; pymatgen absent), 1 xfailed (`artifacts/peer_review_2026-10-19/full_suite_cuda_rtx5060ti_2026-10-25.log`). The modules affected by the later radial-chunk fix were rerun: 80 passed. Mutation suite on the final code: 21 of 21 executable mutations killed, gaps 4, 8 and 9 unchanged (`mutation_suite_run_v3.{json,log}`) |
| Purity-contract gap found while fixing B3 | Closed | `matcher_callable_sha256` scanned only the outer code object. A helper used inside a generator expression, lambda, or nested function was not fingerprinted, and an import there was not rejected. All nested code objects are now scanned | `tests/test_diagnostics_rigor.py::test_matcher_fingerprint_covers_globals_and_imports_in_nested_code` (failed before the fix) |
| B1: tolerance hierarchy | τ selected: **τ = 1e-4 Å** by the predeclared rule, kept by the owner on 2026-10-20 after the large-τ mechanism analysis. Closed: **τ = 1e-4 Å**, selected by the predeclared rule over the complete grid. The τ = 1e-2 abort was diagnosed and fixed, and all arms finished | `SpglibPolicy.idealization_tolerance_angstrom` (below) | `tests/test_idealization_policy.py` (11 tests). `artifacts/s2_symmetry_tolerance_2026-10-19/s2_study_512.partial.json`: the legacy arm reproduces the v6 report record for record (0 semantic mismatches over 512). Accepted, with worst-to-best family rate overall / excluding chart-domain rejections: legacy 193 (0.021 / 0.021); τ = 1e-5 358 (0.228 / 0.313); **τ = 1e-4 431 (0.571 / 0.957)**; τ = 1e-3 423 (0.557 / 0.935). Both readings of the rule select τ = 1e-4. At τ = 1e-4 trigonal acceptance is 40/70 and hexagonal 38/46, against 1/70 and 3/46 in v6. 73 of the 81 rejections are chart-domain (B2). Label agreement among accepted records is 0.940. **The τ = 1e-2 abort:** for mp-1305082 (Pc) at symprec = 1e-2, `find_primitive` searches symmetry again on the idealized conventional cell and returns another basis in another orientation, so H = L_conv L_prim⁻¹ = I − 4.08e-4 e₃₁ is not integral. The two cells are isometric: their Niggli metrics agree to 4e-14 Å². The returned cell is M L_conv R with M = (−a, −b, c + 2a) unimodular and R a proper 2-fold rotation about z. The cause is lattice pseudo-symmetry: c + a is perpendicular to a up to ε = 1.126e-3 Å, so within τ the lattice alone has holohedry mmm while the crystal has 2/m. This gives H₃₁ = −2ε/a exactly. That broke the exact-integrality invariant, and the later site-mapping invariant also assumes one shared frame. All five such rejections at τ = 1e-2 and 1e-1 have a lattice holohedry at τ (mmm or 4/mmm) above the crystal's 2/m. The converse fails: 3 of 7 (τ = 1e-2) and 9 of 19 (τ = 1e-1) pseudo-symmetric low-symmetry records are accepted. A full 512-record scan at τ = 1e-2 found exactly one other record of a related kind. mp-1179655 (C2/c here, upstream Ibam) has an exactly integral affine map (1.1e-16), but its round trip moves six sites by 1.15e-3 (fractional) along b, which is 5.7e-3 Å = 0.57τ. Mechanism: the round trip re-anchors the primitive origin on an atom. On that exactly symmetric cell, spglib at symprec = τ returns the inversion translation (0, 0, ½), which holds only within τ. At symprec ≤ 1e-6 it returns the exact (0.001146, 0.998854, ½). mp-1211371 at τ = 1e-1 behaves the same way. Both rejection kinds therefore mean that, at this τ, the canonical form depends on how the crystal is presented (basis or origin). They are not artifacts of the check. Replacing `find_primitive` by spglib's documented deterministic map L_p = P_cᵀ L_s would remove the first symptom without removing that dependence. The map gives the same primitive cell as `find_primitive` for every record that passed standardization: 433, 420 and 411 records at τ = 1e-4, 1e-2 and 1e-1. It was not adopted. Evidence: `artifacts/s2_symmetry_tolerance_2026-10-19/typed_rejection_mechanism.{py,json,log}`. Under the dimensioned policy, both checks (the H integrality and the pinned-round-trip motif) are typed `AmbiguousPreprocessingError` rejections that record the measured deviation. The legacy policy keeps the fatal checks, and malformed maps, non-isometries and species changes stay fatal everywhere. No accepted output changes under either policy, and the τ = 1e-5 to 1e-3 arms are unaffected because no record reached those branches. Tests: `tests/test_primitive_rebasis.py::test_non_integral_primitive_search_is_fatal_only_under_the_legacy_policy` and `::test_pseudo_symmetric_basis_flip_is_isometric_but_not_integer_related`, `tests/test_pinned_roundtrip.py::test_roundtrip_motif_drift_is_typed_only_under_the_dimensioned_policy`. The τ = 1e-2 and 1e-1 arms on this code (`s2_study_512_large_tau.json`) gave: τ = 1e-2 accepted 419 (0.529 / 0.929); τ = 1e-1 accepted 409 (0.505 / 0.832). The new typed rejections occur 1 + 1 times at 1e-2 and 4 + 1 at 1e-1, and never at τ ≤ 1e-3. Upstream-label agreement rises with τ (0.940, 0.957, 0.981, 1.000 for 1e-4 through 1e-1), as expected, because the upstream labels were assigned at looser tolerances. At τ = 1e-4, 6% of accepted structures take a subgroup of the upstream label. That is the conservative direction, and the predeclared rule optimizes family coverage, not label agreement |
| B2: chart support, resolution, atom bound | Closed: the `loewner-v2` chart is the default since 2026-10-20 (owner decision); `ball-v1` must be requested explicitly and keeps its historical fingerprints. Resolution and atom bound were already implemented. At τ = 1e-4, chart support covers 496/496 held-out records (ball: 450). On real data, 506/506 and 505/506 canonicalized records lift into targets (ball: 433 and 458); the one rejection is a ν-vocabulary limit, not the chart | `SpglibPolicy.relative_metric_quantum`, `DomainConfig.max_primitive_atoms`, `cover_legality`, `lattice.LoewnerSPDChart`, `nn.bridge.loewner_whitened_metric` (below) | `tests/test_relative_metric_quantum.py` (4), `tests/test_primitive_atom_bound.py` (5), `tests/test_loewner_chart.py` (46). `artifacts/b2_chart_support_2026-10-20/` holds per-family coverage, the norm decomposition, and held-out validation of the predeclared rule. Rejection is uncorrelated with cell size (rank correlation 0.06) |

## Review of 2026-10-29

The second review ([REVIEW.md](../artifacts/peer_review_2026-10-29/REVIEW.md),
with its probes) was answered as follows. Opt-in fixes leave the legacy default,
fingerprint and payload unchanged; the owner decides whether they become defaults.

**Verification on tcm-pc38 (2026-10-29, RTX 5060 Ti, CUDA tests included):** the full
suite ran 1343 passed, 2 failed, 2 skipped, 1 xfailed in 51 min
(`artifacts/peer_review_2026-10-29/full_suite_pc38_2026-10-29.log`); the mutation suite
killed 21 of 21 executable mutations with the three listed gaps (4, 8, 9) unchanged
(`mutation_suite_pc38_2026-10-29.{json,log}`). The two failures were fixed the same day
and re-verified on the host (62 tests of the four affected files): (i)
`test_legacy_domain_identity_is_unchanged` pinned the default configuration fingerprint,
which the C3 default flip changes; it now names `truncated-v1` explicitly and reproduces
the historical hash; (ii) `test_nonfinite_inputs_fail_closed[lattice]`: on the Linux
LAPACK backend an SVD of non-finite data raises before the standalone Transformer's
fused finiteness read, so `_canonicalize_inputs` now reads its flags before the
singular-value check (the field's deferred path already handled that case). A full-suite
re-run after these two changes has not been made; the changes touch only that read order
and the test.

| Finding | Status | Correction | Evidence |
|---|---|---|---|
| C1: dequantization noise isotropic in fractional coordinates, anisotropic in space | Closed (opt-in `CFMDequantization.intrinsic_kernel="cartesian-v2"`; `fractional-v1` stays the default) | Torus heat kernel with the record-dependent covariance Σ = σ²(ĴᵀP_NĴ)⁻¹ (theory (1.30d)), so tr(MΣ) = σ²d and σ is the RMS centred Cartesian displacement per free coordinate in Å. `nn.bridge.intrinsic_noise_covariance` / `intrinsic_noise_transform`; `flow.sample_dequantized_endpoint(eta_noise_transform=)`; `TinyFlowOracle(eta_covariances=...)` evaluates the exact Fourier law with per-record covariances; `flow_oracle_learned.toy_oracle(intrinsic_kernel=)` | `tests/test_cartesian_dequantization.py` (7): covariance = σ²M⁻¹; on a 3 × 3 × 30 Å P1 cell the c-axis to in-plane Cartesian displacement ratio is 0.85–1.18 under `cartesian-v2` and > 60 under `fractional-v1`; legacy bitwise; the oracle matches the sampler's moments and the production objective's residual |
| C2: GL(3,Z) lattice-basis gauge not quotiented for low-symmetry covers | Closed (opt-in `CFMDequantization.basis_augmentation="shear-v1"`, owner chose augmentation over a reduction-invariant chart on 2026-10-29) | `basis_augmentation.py`: with probability `basis_augmentation_probability` (default 0.5) a record is moved to a neighbouring presentation by one of its admissible elementary shears W = I ± E_ij ((W, 0) normalizes the setting exactly and maps every cover row onto itself; `_row_affine` gives the exact η map); G' = W⁻¹ G W⁻ᵀ; an image outside the Loewner interval is refused. Theory (1.30e). The `cartesian-v2` transform is evaluated at the moved record; `none` makes no draw and is bitwise legacy | `tests/test_basis_augmentation.py` (5): the 12 transvections; a P1 two-atom cover admits all 12 and every image decodes to the same crystal (positions x → xW, metric W⁻¹GW⁻ᵀ, volume kept); P-1 with 1a + 1b admits exactly the 8 shears fixing (0, 0, ½); Pm-3m admits none; the objective option is opt-in, moves the endpoint, is validated and recorded in the contract. Probe D: \|z₁ − z₂\| = 1.54 between two reduced bases of one P1 lattice |
| C3: default lattice-head slots mean different things per family | Closed: `commutant-v2` is the default (M3 row above). The default configuration fingerprint therefore changes; the historical identity is reproduced by naming `truncated-v1` explicitly | `ModelConfig.geometry_lattice_head="commutant-v2"`; the F10 scripts pin `truncated-v1` for A-F10L | `tests/test_lattice_head_commutant.py`; `tests/test_primitive_atom_bound.py::test_legacy_domain_identity_is_unchanged` pins the legacy fingerprint with the explicit head |
| M1: the selector cannot see where a Wyckoff site is | Closed (opt-in `ModelConfig.selector_row_geometry_features`) | `symmetry_descriptors.row_geometry_descriptor`: the exact chart (offset, coordinate matrix padded to 3 × 3) of the row's canonical representative under the full Euclidean normalizer, plus the canonical and own row indices (14 entries); `row_normalizer_classes` groups rows by the exact `_row_affine` test under N_E ∘ G, cached per setting. `operation_derived_row_features(..., geometry=True)` appends the block; `CoverSelectorConfig.row_geometry_features` | `tests/test_row_geometry_features.py` (8): Pm-3m 1a/1b, 3c/3d, 6e/6f, 12i/12j, 24k/24l share the class part while 12h/12i, 12h/12j, 24k/24m, 8g/12h differ; I4/mmm 8i/8j equal and 8g/8h/8i pairwise distinct (they collide on every legacy feature); Fm-3m 4a/4b, 48h/48i; P6₃/mmc 2c/2d; the block is appended only when asked and the legacy features are bitwise unchanged; the option is versioned in the fingerprint and payload and a model samples with it |
| M2: the production chart is absent from the theory | Closed | Theorem 2.3b in `theory/02_THEOREMS.md`: the spectral logistic map W(Z) = l I + (u − l) σ(Z − ln c I) is a smooth bijection of the commutant onto the open Loewner interval {l G₀ ≺ G ≺ u G₀}, W(0) = I, with the floor σ_min(L) ≥ √(λ_min(G₀) l); the base lives in a logit coordinate. (1.25a), (1.30d) and the kernel options in `01_FOUNDATIONS.md` §1.7 and §1.10; §3.4–3.7 of `03_ARCHITECTURE.md` | — |
| M3: the field runs on the conventional cell (up to 16× for centred lattices) | Closed (opt-in `ModelConfig.geometry_expansion_cell="primitive-v2"`, owner decision 2026-10-29; dense backbone only) | `TorchCoverGeometry(layout, expansion_cell=...)`: one coset representative per class modulo the centring translations, lattice P L (P = catalogue centring reduction), coordinates x P⁻¹; the chart, decoder and (1.30d) keep the conventional cell; `intrinsic_noise_covariance` multiplies the primitive metric by I_c; the objectives normalize by the bridge's atom count, so the coordinate loss is identical. Theory §3.6 step 2 | `tests/test_primitive_expansion.py` (6): on Fm-3m, Im-3m, R-3m and C2/m covers the primitive expansion has N / I_c atoms, each conventional atom maps onto a primitive atom (mod the primitive lattice) with the same species and Cartesian Jacobian, M_conv = I_c M_prim to 1e-12, the kernel covariance agrees, batched and padded expansions agree; the option is versioned and refused with the local backbone; a model trains and samples on rock salt with 2 field atoms and an 8-atom decode |
| M4: flow sample efficiency | Measured, not new (budget scan N^−0.54); no change | — | — |
| m1: docstring reason wrong | Closed | `reference.py` and the m3 row above: disjointness follows from the Wyckoff partition and the exclusion of specializations | — |
| m2: block-metric units | Closed | `CFMBlockWeights(lattice, coordinates)` on `CrystalAtlasTrainingObjective`, plumbed as `lattice_weight`, `coordinate_weight` into `conditional_flow_matching_loss` (3.14); unit weights are bitwise legacy and leave the contract unchanged; current-state constants keep the minimizer (Theorem 2.9) | `tests/test_block_weights.py` (3): weights scale the blocks, unity is bitwise, validation, contract payload, per-record objective |
| m3: one failed member aborts a sampling batch | Closed | `LearnedCrystalGenerator._integrate_members`: when the batched solve raises a `RecoverableGenerationError`, `TangentRankError` or `ValueError`, every member is replayed alone from its recorded generator state; the failing member gets a `geometry/ode` trace. `ResourceLimitError` and `InvariantViolation` still propagate | `tests/test_sampling_member_fallback.py` (3): the survivors equal the un-injected run to 1e-10; resource and invariant errors abort |
| m4: base mass far from physical volumes | Deferred by the owner (2026-10-29) until MP-20 statistics exist; the candidate is an opt-in data-fitted Gaussian base in z with a residual-invariance check | — | — |
| m5: time features | Open (harmless) | — | — |
| H1: host synchronizations in the field | Closed | `nn/host_sync.DeviceChecks`: every finiteness, handedness, rank and definiteness flag is stacked on the device and read once per field call; the first failing flag raises its original error; a `RuntimeError` inside the deferred block yields to an earlier failed check. The reducers, the time features, the commutant basis and the Transformer accept `checks` and otherwise read their own flags. The integrator and the bridge validate the period basis once instead of twice per RK4 stage, and the sampling closure encodes the composition context once | `tests/test_host_sync.py` (5): one `.item()` and no `bool(tensor)` per packed and per single field call (was 25 per packed call on a four-atom P1 cover); messages and exception types kept. Probe (CPU, tiny model, 2 RK4 steps, batch 4): 296 host reads per integration before, 15 after |
| H2: mixed-cover batches remain per-record | Closed | Each bridge keeps fused per-atom expansion constants A[n] = M_o C_o R_j, c[n] = o_o R_j + t_j (non-persistent buffers); `nn.bridge.expand_padded` expands a whole mixed-cover batch in two `einsum` calls and `lattice_pullback_metrics` batches the pullback per shared bridge; `geometry_cfm_objective_padded` uses them (the per-record metric-from-latent remains one small call per record) | `tests/test_padded_expansion.py` (2): equals the per-record expansion to 1e-12 (positions mod 1, Jacobians, lattices, padding) with gradients, and the padded objective still equals the sequential one |

## B1: one dimensioned tolerance hierarchy

### What changed

`SpglibPolicy.idealization_tolerance_angstrom` (default `None`) replaces every raw-to-ideal comparison by one statement in angstrom.

- **Lattice.** Every point of the primitive cell moves by at most τ under idealization. `lattice_idealization_displacement` computes this exactly at the 8 cell vertices: the norm is convex, so the cube maximum lies at a vertex. Four gates use it:
  - three recoverable gates in `standardization.py`;
  - the fatal post-init certificate in `CanonicalizedStructure`.
- **Volume.** The raw-to-primitive volume/index relations in `ingestion.py` (×2) and `dataset_identity.py` use `idealized_volume_matches`. The bound is `|V'/V − 1| ≤ (1 + √3·d/σ_min)³ − 1` from the recorded displacement d. Derivation: `‖Δ‖₂ ≤ √3·d`, then Weyl's inequality.
- **Internal consistency.** Checks between exact or idealized objects keep the tight `roundtrip_tolerance`.
- **Metadata.** The raw-to-ideal lattice and site displacements are recorded in the canonicalization receipt as optional fields, written and fingerprinted only under the dimensioned policy.

`CatalogueTargetAssigner(merge_quantization_equivalent_orbits=True)` fixes a latent assigner defect exposed by B1. With rhombohedral thirds, 12-decimal quantization of different orbit points expands to exact orbits that differ by ≤ 1.4e-11 Å; for mp-2832 that gave 7 "distinct" 18f orbits. These were rejected as ambiguous. They are now one orbit when pairwise within 100·10⁻¹²·σ_max(L).

### Legacy preservation

With `None` and `False`, the defaults, every payload and message is unchanged:

- the policy payload, provenance fingerprint, builder fingerprint and receipt fingerprint;
- the storage schema, which reads both old and new records;
- the v6 rejection messages.

**Evidence.** The legacy arm of `scripts/s2_symmetry_tolerance_study.py` reruns all 512 pilot records on numpy 2.4.6. It gives builder fingerprint `a6e24ded…`, 193 passed, and **0 status or reason mismatches** against `preprocessing512_v6.json` (`artifacts/s2_symmetry_tolerance_2026-10-19/s2_legacy_arm_512.json`).

### Remaining

The S2 grid τ ∈ {1e-5, 1e-4, 1e-3, 1e-2, 1e-1} Å runs the deployed v7 candidate:

- dimensioned τ;
- relative quantum 1e-9;
- orbit merging;
- primitive bound 20.

It reports per-family acceptance overall and excluding chart-domain rejections, plus upstream-label agreement. τ is then chosen by the predeclared rule: the smallest τ with no family below half the best family's rate.

## B2: chart support, resolution, atom bound

### Resolution

`relative_metric_quantum = c` gives a ranking quantum of `c·max_i G_ii`, computed without LAPACK. For every admitted `c ≥ 1e-11`:

- the resolution guard can never trigger (`768·eps < c/8`);
- ranking is exactly equivariant under power-of-two rescaling.

At `c = 1e-9` the guard band is about 10⁻³ of a bin. The legacy band was 0.1–0.23 of a bin, which drove the 143 bin-wall rejections.

### Atom bound

`DomainConfig(max_primitive_atoms=20, max_raw_atoms=80)` admits a cover iff `ν|c| ≤ 20·I_G`, with 80 as a conventional resource cap. `I_G` divides ν because the reduced formula has gcd 1, so the primitive cell holds `ν|c|/I_G` atoms. `cover_legality.py` provides the ν and group masks for both the sampler and the training targets, so their likelihood supports cannot diverge. With the legacy bound the masks are bit-identical to before, and so are the stored cover decisions.

### Chart support

Measured at the selected τ = 1e-4 Å, with the relative quantum 1e-9 and the primitive bound 20
(`artifacts/b2_chart_support_2026-10-20/b2_chart_support_tau1e-4.json`, local-only because it holds per-record real-structure metrics; its `.log` is tracked; 506 of 512 records
canonicalized). Neither tested support meets the ≥ 99% per-family criterion.

| Family | n | In production ball (B = 2) | In 2-fold cross-fitted per-setting shape (B = 2) | Max ‖b‖ | Max isotropic part | Max anisotropic part |
|---|---|---|---|---|---|---|
| triclinic | 27 | 26 | 26 | 2.14 | 0.95 | 2.08 |
| monoclinic | 76 | 67 | 72 | 2.62 | 0.94 | 2.60 |
| orthorhombic | 106 | 89 | 93 | 3.85 | 0.92 | 3.81 |
| tetragonal | 84 | 72 | 42 | 3.03 | 1.01 | 3.03 |
| trigonal | 63 | 36 | 49 | 5.06 | 0.94 | 5.03 |
| hexagonal | 40 | 33 | 38 | 3.50 | 1.24 | 3.45 |
| cubic | 110 | 110 | 107 | 1.75 | 1.75 | 0.00 |

The isotropic part is |tr A|/√3 of A = log(G₀^{-1/2} G G₀^{-1/2}); the anisotropic part is its
traceless remainder.

- **Cell size is not the cause.** Rank correlation between ‖b‖ and conventional atom count is 0.06,
  so the second acceptance criterion holds.
- **Shape is the cause.** Rejections come from shape anisotropy (layered long axes), not volume: the
  isotropic part is ≤ 1.24 except one cubic record at 91 Å³/atom.
- **Per-setting fitted centres are not enough.** They help trigonal and hexagonal but hurt tetragonal
  (bimodal shapes within one setting).
- **A larger ball trades away the shortest-vector floor.** The layout certifies
  σ_min ≥ √λ_min(G₀)·e^{−B/2} ≥ 0.75 Å. A ball large enough for trigonal (B ≈ 5) also admits the
  mirror-image short-axis shapes, so this floor would fall to about 0.2 Å. Real records have
  √λ_min(G) ≥ 1.77 Å.

**Resolution: the Loewner-interval chart (`DomainConfig(lattice_chart="loewner-v2")`, the default since 2026-10-20).**

*Derivation.* With G = S e^A S and S = G₀^{1/2}, every unit x satisfies
xᵀGx ≥ λ_min(G₀)·e^{λ_min(A)}. The shortest-vector floor therefore depends only on the most
negative eigenvalue of A, not on ‖A‖_F. The ball spends one budget on two physically different
deviations: a benign long axis (layered c) and a dangerous short one. The new support is the open
matrix interval

  l(N)·G₀ < G < u·G₀ (Loewner order), inside the point-group-invariant cone.

Equivalently, every lattice direction's length lies within [√l, √u] of the reference. The lower
bound comes from physics, not a fit: l(N) = s_min²/λ_min(G₀(N)), raised by whole ulps, so that
σ_min(L) ≥ s_min = 0.75 Å is certified exactly. The upper bound u = e^M is the only fitted parameter.

*Latent map.* For Z in the commutant,
W(Z) = (u e^Z + c l I)(e^Z + c I)^{-1} = l I + (u − l) σ(Z − ln c I), with c = (u − 1)/(1 − l), and
G = S W S. Its properties:
- W is a monotone scalar function of Z, so it maps the spectrum bijectively onto (l, u).
- It preserves the commutant, so point-group invariance is exact.
- W(0) = I, so the latent origin is the reference shape.
- It is defined at repeated eigenvalues.

The exact chart (`lattice.LoewnerSPDChart`) evaluates the map spectrally with an overflow-free
logistic. The torch twin (`nn.bridge.loewner_whitened_metric`) uses an exact Daleckii–Krein backward
whose closed-form divided differences, [σ(a) − σ(b)]/(a − b) = (sinh h/h)/(4 cosh(a/2) cosh(b/2)),
are evaluated in log-cosh form. Eigenvectors are never differentiated.

*Predeclared fit and held-out validation*
(`artifacts/b2_chart_support_2026-10-20/evaluate_loewner_support.py`, `loewner_support_validation.json`):
- **Rule**, fixed before the validation sample was drawn: M = ⌈2(max λ_max(log W) + 0.25)⌉/2 on sample A
  (the 512-record pilot, seed 20260913). This gives M = 4.5.
- **Validation sample:** a disjoint training sample B (seed 20261020), with 10 overlapping records
  excluded.
- **Held-out coverage:** 496/496 overall (100%) against 450/496 for the ball. Trigonal is 42/42 against
  28/42.
- **Floor slack:** the minimum is 1.91 in log units, so the physical floor never binds. The held-out
  max λ_max is 3.885 < 4.5.
- **Statistical bound:** 0/496 misses give a 95% upper bound of 0.6% on the miss rate. Per-family
  ≥ 99% cannot be certified statistically at n ≈ 40, and the review's in-sample criterion will be
  measured on the full training split when it is built.

*Integration and real-data evidence.*
- `IntrinsicCoverLayout(lattice_chart=...)` and `DomainConfig.layout_options()` carry the chart into
  `learned.py` and both target-lift paths.
- The uniform control and the reference law sample uniformly on the support, within a volumetric
  window. They use batched cube rejection, which is exact; on the support alone, acceptance is
  0.27% for P1 and 89% for Fm-3m.
- **Volumetric window (correction found by the post-B2 literature check, 2026-10-20).** Uniform on
  the Loewner box alone is not a physical baseline. The box admits ln(V/V₀) = tr(log W)/2 up to
  (3/2) ln u = 6.75, i.e. V/V₀ ≈ 854.
  - **Measured effect:** 20–72% of that law's mass lies outside the volumes of real crystals
    (P-1 20%, Fm-3m 72%), and 8–35% has V/V₀ > 10. The legacy ball law kept |ln(V/V₀)| ≤ √3 only
    as a side effect of its symmetric radius.
  - **Data:** over 1002 distinct real lattices, ln(V/V₀) has mean −0.078, sd 0.385 and range
    [−0.88, +1.51]. The extreme log-metric eigenvalues are deviatoric (shape), not volumetric, so
    the box's upper bound u is set by shape.
  - **Literature:** AIRSS draws random cells around a natural volume proportional to the atom
    count, within about ±50% (Pickard and Needs, [arXiv:1101.3987](https://arxiv.org/abs/1101.3987),
    §3.1–3.2).
  - **Fix:** under `loewner-v2` the control and the reference law sample uniformly on the support
    intersected with `DomainConfig.loewner_control_log_volume_window` (lo < ln(V/V₀) < hi). The
    window is fitted on sample A by the rule of u applied to each side. It gives (−1.5, 2.0), and
    covers 506/506 fit and 496/496 held-out records
    (`artifacts/b2_chart_support_2026-10-20/evaluate_volumetric_window.py`,
    `volumetric_window_validation.{json,log}`).
  - **Scope:** the learned support is unchanged. Its base law is N(0, I) in the latent, and the
    chart maps Z = 0 exactly to G₀. The data mean of ln(V/V₀) is −0.08, about 0.2 sd from 0. The ball-v1 payload is unchanged. Tests:
    `tests/test_loewner_chart.py::test_log_volume_window_*` and
    `::test_uniform_control_and_reference_law_respect_the_log_volume_window`.
- The legacy domain payload and every legacy fingerprint are unchanged when `ball-v1` is requested
  explicitly. `DomainConfig(lattice_chart="ball-v1")` reproduces the v6 report payload and the
  historical configuration fingerprint `04cf6e2c…`.
- **Default switch (owner decision, 2026-10-20).** `loewner-v2` is the default of `DomainConfig` and
  `IntrinsicCoverLayout`.
  - **Frozen paths pinned to ball-v1:** scripts and artifacts that reproduce frozen runs now name
    `ball-v1` explicitly. These are `compile_mp20_research_pilot.py`, `train_research_pilot.py`,
    the S2 study arms, `objective_benchmark_micro_config()` (a pinned workload), and the
    historical audit and GPU-evidence scripts. Without the pin, a rerun would silently use the new
    chart under an unchanged script.
  - **Fail-loud identity:** a legacy dataset or checkpoint loaded under the new default fails
    loudly. The portability gate reports a stored-latent deviation of 0.40 > 1e-12.
  - **Inactive parameters are refused:** a non-default value for a parameter of the inactive chart
    raises. That is `metric_radius` / `metric_bound` under `loewner-v2`, and the Loewner bounds
    under `ball-v1`, which are also left out of its fingerprint. Before the guard, three test
    fixtures with `metric_radius` = 1.0 or 1.25 had silently changed chart.
  - **Second derivatives are refused:** the Loewner adjoint is first-order. torch's
    `once_differentiable` raises only when the incoming gradient requires grad. A `create_graph`
    Jacobian such as `lattice_pullback_metric(z, create_graph=True)` instead returned silently
    zero second derivatives. The map now raises. Production detaches the pullback weight and uses
    first derivatives only, so no result changes. First-order gradcheck and the Daleckii–Krein
    tests are unchanged.
  - **Float64 island:** the Loewner path keeps the float64 chart island under bfloat16 autocast.
  - **Tests:** `tests/test_loewner_chart.py::test_parameters_of_the_inactive_chart_must_stay_at_their_defaults`,
    `::test_loewner_bridge_refuses_second_derivatives_instead_of_dropping_them`,
    `::test_loewner_chart_island_ignores_enclosing_neural_autocast`. The ball-specific tests
    (bridge, control, reference law, v6 contracts) now name `ball-v1`.
  - **Verification on the new default:** the full suite gives 1156 passed, 0 failed, 30 skipped,
    1 xfailed. The mutation suite still kills 21/21 executable mutations, with gaps 4, 8, 9
    (`artifacts/peer_review_2026-10-19/mutation_suite_run_loewner_default.{json,log}`).

Rerunning the B2 study with `--lattice-chart loewner-v2`
(`b2_loewner_lift_seed{20260913,20261020}.json`, local-only per-record metrics; the `.log` files are tracked) gives:
- **Sample A:** 506/506 canonicalized records lift into training targets, against 433 for the ball.
- **Sample B:** 505/506 lift, against 458 for the ball. The single rejection is mp-570690 (R3m, ν = 27).
  It exceeds the separately declared selector vocabulary `max_formula_units = 16` and is not a chart
  rejection.

Tests: `tests/test_loewner_chart.py` (46):
- round trips and strict containment for ‖z‖ up to 500;
- invariance in five settings, the exact floor certificate, and refusal outside the interval;
- the ball being contained in the box;
- `gradcheck` at degenerate and saturated spectra;
- a torch–NumPy match and the pullback metric against finite differences;
- exactness of the uniform law (KS test);
- end-to-end generation and training.

The centre is the mean chart coefficient, which decodes to a point-group-invariant metric. Smoke test, first 40 pilot records, deployed policy at τ = 10⁻³ Å: 31 lift and 8 fail only the chart ball (the v6 pipeline accepted 14 of these 40).

## B3: certified physical matcher

### Interface

- `encode_physical_structure(lattice, fractional, Z) -> bytes`
  1. LLL-reduce the input basis, an exact proper change of basis.
  2. Apply the certified primitive reduction (`geometry.primitive_reduce_arrays`, at 0.01 Å).
  3. Emit a canonical ASCII record.
- `physical_structure_match(a, b) -> bool`, the frozen diagnostic matcher.
- `physical_structure_rms(a, b) -> float | None`, the RMS displacement in units of (V/N)^{1/3}.
- `physical_structure_witness(a, b)`, the auditable witness `(N, t, π)`.
- `physical_structure_matcher_contract()`, the `EquivalenceMatcherContract` binding the code and every tolerance.
- `evaluate_physical_geometry_decomposition(...)` validates every payload, then runs the oracle-versus-predicted diagnostic.

### Relation

- Both crystals are rescaled to unit volume per atom.
- One crystal is the reference. Its reference bases are all right-handed bases whose rows attain the successive minima, within 1e-9 relative. This set is a lattice invariant. It contains, for example, both the 60/60/60° and the 90/60/60° reduced cells of a face-centred cubic lattice.
- A correspondence is a `det N = +1` basis of the other lattice.
  - Its row lengths are within a factor 1.3 of a reference basis, in both directions.
  - Its row angles are within 10° of that basis.
- Site displacements are exact closest-image distances in the mean of the two normalized metrics.
- A search succeeds if some correspondence, translation, and species-preserving bijection keeps every displacement ≤ 0.5. The pair matches if the search succeeds with either crystal as the reference.

### Guarantees

| Property | Statement | Test |
|---|---|---|
| Soundness | Every acceptance carries a witness. Every displacement in it is checked with an exact closest-vector solution | Witness audited independently with numpy and `periodic.closest_periodic_image` |
| Completeness | Any witness within `stol/2` is found (anchor translation plus triangle inequality on the torus) | Planted witnesses at 0.49·stol/2 under random proper basis change, rotation, origin, and permutation |
| Exact threshold | With a forced bijection, the decision is exactly `‖Δ‖ ≤ 2·stol` and the RMS is `‖Δ‖/2` | 120 random displacements against the analytic answer |
| SO(3) only | A chiral motif and its mirror image do not match | Rejection certified by a triangle-inequality lower bound of 0.63 > 0.5 |
| Invariance | Unchanged under basis change, rotation, origin shift, permutation, supercell, and argument order | 900 borderline hexagonal, FCC, and cubic trials with both crystals rewritten: 0 changes |
| Reproducibility | Python integers and IEEE-754 `+ − × ÷ √` only; no libm transcendental. cos 10° and sin 10° are correctly rounded literals from an 80-digit series | Cube root checked to within 1 ulp with exact rationals; cosine window equivalent to the angle test |

`verify_b3_mutations.log` records seven deliberate defects, each applied to a
copy of the source. All seven are caught by the suite:

- improper (O(3)) maps;
- no least-squares refinement;
- one refinement step instead of three;
- componentwise wrapping in place of the closest-vector search;
- a single reference basis;
- a single reference role;
- an anchor chosen by coordinate order.

The anchor regression case and the three-step case were found by search,
because no random test exercised them.

### Documented limits

- **Heuristic band.** For witnesses between `stol/2` and `stol` the search is heuristic: one anchor atom per role, then three least-squares refinement steps (`_REFINEMENT_STEPS`).
- **Tie cases.** Invariance can fail only in three cases:
  - successive-minimum ties within 1e-9 that are not exact symmetry images;
  - displacements lying on a threshold to within rounding;
  - exact assignment-cost ties between two exactly symmetric structures, where the refinement follows site order.
- **Permissiveness.** The union over reference bases and the OR over roles make the relation more permissive than a single LLL reference. On the 300-trial borderline set of seed 0, matches rose from 78 to 145. The single-reference variant changed 27 of those 300 decisions under re-representation and was rejected for that reason.
- **Cost.** Pure Python.
  - Matches resolve in about 1–46 ms.
  - Lattice-compatible non-matches of 20-atom high-symmetry cells take about 0.9–1.9 s, measured under concurrent load.
  - Composition or lattice mismatches return early.

### External agreement with pymatgen

All numbers come from the isolated adapter venv: pymatgen 2026.9.24 with pymatgen-core 2026.10.16, and `StructureMatcher(ltol=0.3, stol=0.5, angle_tol=10, primitive_cell=True, scale=True, attempt_supercell=False)`. The pairs are frozen with seed 20261019 over the 512 hash-random MP-20 pilot structures (`verify_b3_pymatgen_agreement.json`).

| Pair set | n | Agreement | Both match | Neither | Only CrystalAtlas | Only pymatgen |
|---|---:|---:|---:|---:|---:|---:|
| Exact re-representations | 512 | 100% | 512 | 0 | 0 | 0 |
| Near misses (strain ≤ 0.3, displacement ≤ 0.5) | 512 | 87.7% | 231 | 218 | 63 | 0 |
| All | 1024 | 93.85% | 743 | 218 | 63 | 0 |

**Finding 1: CrystalAtlas ⊇ StructureMatcher on this set.** Every pair pymatgen accepts, CrystalAtlas also accepts.

**Finding 2: the 63 extra acceptances are real witnesses.** For each, `verify_b3_disagreement_witnesses.py` re-scores the CrystalAtlas witness under pymatgen's own conventions:

- geometric-mean volume scaling;
- the lattice built from averaged (a, b, c, α, β, γ);
- normalization by (n/V_avg)^{1/3};
- exact closest images.

All 63 have a largest distance ≤ 0.5. They are matches under pymatgen's own definition that its search does not find: one Niggli reference, one role, and one least-squares shift after an unconstrained assignment.

**Finding 3: pymatgen is not order-consistent on these pairs.** `fit(a, b) ≠ fit(b, a)` for 9.6% of the 512 near misses (`verify_b3_semantics_ablation.json`). It is invariant under re-representation (0%), thanks to Niggli reduction. The two invariance mechanisms of CrystalAtlas (OR over roles, union over reference bases) account for only about 1 percentage point of the disagreement.

**Recommendation.** The ≥99% decision-agreement criterion cannot be met by a symmetric, representation-invariant relation that accepts every certified witness, and it is not met here. Its reference matcher disagrees with itself under argument swap on 9.6% of near misses.

I recommend replacing the criterion with what was measured:

- (a) 100% agreement on exact re-representations;
- (b) no pymatgen acceptance missed;
- (c) every extra acceptance carries a witness valid under pymatgen's own distance convention.

**Amendment (approved by the project owner, 2026-10-19).** The B3 acceptance criterion is replaced by these three measured conditions. All three hold on the frozen set (`verify_b3_pymatgen_agreement.json`, `verify_b3_disagreement_witnesses.json`):

1. 100% decision agreement on exact re-representations: 512 of 512.
2. No StructureMatcher acceptance missed: 0 pymatgen-only pairs in 1024.
3. Every additional acceptance carries a witness whose largest displacement is ≤ stol under pymatgen's own distance convention: 63 of 63.

The original review's invariance and mirror criteria are unchanged and met. The ≥99% figure remains on record as not met.

### Policy choices to confirm

- `PRIMITIVE_REDUCTION_TOLERANCE_ANGSTROM = 0.01`. This is the encoder's primitive-reduction tolerance. It is declared, not derived, and is recorded in the payload header and the contract.
- `ltol = 0.3`, `stol = 0.5`, `angle_tol = 10°`. These are the conventional MP-20 CSP values, frozen in the contract.

### Residual design constraint

The tiny-overfit gate requires the scaffold and geometry evidence to share one target corpus digest. Its policy fingerprint includes that digest, so both arms need identical target bytes. `tests/test_tiny_overfit_end_to_end.py` therefore uses a scaffold-identity line followed by the canonical physical record. Its geometry matcher delegates to `physical_structure_match`, replacing the former indexed fixture matcher, which was not representation-invariant. Splitting target-corpus digests by identity level would remove this constraint. It is not done here.
