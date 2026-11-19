# Research decisions after remediation

This is a decision record, not a claim of a new theorem, reproduced leaderboard
result, or an exhaustive literature review. Primary sources were checked on
7 September 2026. See also the existing
[novelty and prior-art map](../theory/05_NOVELTY_AND_PRIOR_ART.md).

## What the requested paper supports

[Clari](https://arxiv.org/html/2606.03199v1) studies molecular-graph-conditioned
organic CSP using a non-equivariant flow-matching Transformer. Pair-bias
attention, explicit unit cells, self-conditioning, priors, and training-time
choices are useful comparison points. Its ablations and atom-count-dependent
inference batching motivate measurements, not universal defaults for inorganic
generation. Its conditioning and evaluation tasks differ from atomic-species
de novo generation.

Decision for CrystalAtlas: keep the present scalar-invariant Transformer and
covariant readout as a provisional candidate. The immediate systems change is
deterministic pair/workspace-bounded ODE chunking with preserved attempt IDs
and RNG streams. Clari does not establish that this architecture, or flow
matching in general, beats diffusion on our frozen corpus. Self-conditioning,
fitted lattice priors, and timestep changes remain controlled ablation
candidates. No paper code, weights, or restricted crystal database is imported
by this remediation.

## Additional relevant primary work

- [Space Group Conditional Flow Matching](https://arxiv.org/abs/2509.23822)
  explicitly conditions noise and vector fields on crystallographic symmetry.
  Symmetry-constrained flows and group averaging are therefore prior art, not
  novelty claims for CrystalAtlas. Its existence reinforces the need to test
  the full noise law, not only network covariance.
- [CrystalDiT, version 3](https://arxiv.org/abs/2508.16614v3) provides a unified
  diffusion-Transformer comparator. This is evidence that diffusion and
  Transformers are compatible choices; attention architecture and generative
  training objective are different design decisions. Published results are not
  a matched comparison with our untrained package.
- [DynaCrys](https://arxiv.org/abs/2608.07401), submitted August 2026, couples
  symbolic space-group transitions, Wyckoff occupations and elements with
  continuous geometry. Its reported use of two relaxation/evaluation engines
  is a relevant evaluation design. A dynamic symbolic process would change
  CrystalAtlas's present fixed-cover probability model, so it is a separate
  research proposal, not a drop-in repair.

## Preregistered next experiments

The following are proposed tests, not implemented improvements or claimed gains:

1. Compare the existing endpoint law with a symmetry-compatible empirical
   lattice prior fitted only on the training split. Report reconstruction,
   volume/density distributions, raw failures, and relaxation outcomes; low
   collision counts alone can reward unphysically sparse cells.
2. Test self-conditioning with identical backbone capacity and separately
   report extra training compute, function evaluations, and total latency.
3. Compare flow/diffusion objectives at matched training and inference budgets
   on identical legal covers and data splits. Freeze noise/path definitions
   and numerical tolerances before seeing test results.
4. Compare invariant and non-equivariant alternatives only with a declared
   augmentation protocol, orientation tests, chirality tests, and matched
   parameter/compute budgets. Do not conflate an invariant scalar representation
   with an invariant Cartesian velocity: the latter must transform covariantly.
5. Use the same CSP/S-DNG/DNG conditioning, candidate budget, deduplication,
   energy reference, and relaxation engine for baseline comparisons. Preserve
   failed attempts in every denominator and distinguish pre/post-relaxation
   symmetry and novelty.

The candidate contribution remains the integrated, explicitly specified
probability/decoder/trust contract and its testable implementation. Whether it
adds predictive value must be established empirically; novelty and superiority
cannot be obtained by renaming established components.
