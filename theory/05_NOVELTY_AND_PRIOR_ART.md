# 5. Novelty, attribution, and prior-art boundary

## 5.1 Direct answer

The normative source of this architecture is its declared physical quotient,
crystallographic constraints, probability law, and learning objective.  Some
underlying mathematical primitives—Wyckoff representations, exact-composition
enumeration, grammar-constrained decoding, invariant lattice coordinates,
symmetry-constrained diffusion or flow matching, orbit-level networks, and
periodic attention—are established.  Citing that fact delimits our claims; it
does not make the corresponding external systems design sources.

The workspace now contains an independently written alpha implementation and
its exact-oracle test suite.  It contains no imported generator code,
checkpoint, or configuration from a compared model.  That provenance boundary
does not prove that every scientific idea is novel.  Independent mathematical
derivation, software provenance, copyright, patent freedom to operate, and
research novelty are different questions.

The architecture is specified from its declared physical quotient,
crystallographic constraints, probability measure, and learning objective.
Named generators below are **comparison objects and claim-boundary evidence,
not sources of architectural requirements**.  Nothing in their code,
checkpoints, configurations, equation wording, or preprocessing may enter the
future core implementation.  Benchmark adapters remain isolated.  Conversely,
we will not hide a relevant overlap: a result can be independently derived and
still be established prior art, in which case it must be cited and cannot be
claimed as our invention.

### Novelty-admission rule

A proposed core component is admitted only if it:

1. follows from a stated mathematical, crystallographic, physical, or
   statistical failure of the current model rather than imitation of a paper;
2. has a complete equation-level definition and a simpler alternative against
   which it can be falsified;
3. survives an independent prior-art and provenance audit after derivation;
4. removes a measurable bottleneck under a preregistered matched-budget test;
   and
5. does not duplicate an established method under new terminology.

If the audit finds the construction already established, it may remain as a
cited foundation or baseline, but it is removed from the novelty claim.  If it
adds complexity without a decisive test, it does not enter the core.

## 5.2 What is established

The following table fixes the attribution boundary.  It is a research map, not
an exhaustive patent or legal search.

| Concept or property with relevant precedent | Representative prior work | Consequence for our claims |
|---|---|---|
| Authoritative space-group/Wyckoff data and standard-cell conventions | [IUCr International Tables symmetry database](https://journals.iucr.org/j/issues/2023/06/00/dv5006/index.html), [spglib definitions](https://spglib.readthedocs.io/en/latest/definition.html) | Catalogue operations, Hall settings, centering transforms, and standardization conventions are external crystallographic foundations that must be pinned and verified. |
| Space-group/Wyckoff autoregressive crystal generation | [CrystalFormer](https://arxiv.org/abs/2403.15734), [WyFormer](https://arxiv.org/abs/2503.02407), [WyckoffAUGen](https://openreview.net/forum?id=1ne8lalwRN) | We did not invent Wyckoff tokenization or constrained Wyckoff autoregression. |
| Coordinate-free element-decorated Wyckoff protostructures | [Wren](https://arxiv.org/abs/2106.11132) | The discrete scaffold object itself is established. |
| Exact composition-compatible Wyckoff enumeration | [RandSpg](https://doi.org/10.1016/j.cpc.2016.12.005), [PyXtal](https://arxiv.org/abs/1911.11123), [Parackal et al.](https://arxiv.org/abs/2309.16454) | Formula feasibility and Diophantine enumeration are not new. |
| Generating functions and dynamic programming for Wyckoff sequences | [Hornfeck and Cerveny](https://doi.org/10.1107/S2053273323002437) | The suffix table is an application of established combinatorics, not a new DP theorem. |
| Learned formula-to-space-group/Wyckoff constrained selection | [ShotgunCSP-GW](https://www.nature.com/articles/s41524-024-01471-8), [NextCrystal](https://arxiv.org/abs/2602.17176) | Learned exact-stoichiometric scaffold search is already close prior art. |
| Row-by-element Wyckoff count representation | [WyckoffDiff](https://arxiv.org/abs/2502.06485) | We did not invent count-table tokenization. |
| Legality-constrained symbolic crystal decoding | [DynaCrys](https://arxiv.org/abs/2608.07401) | We cannot claim the first learned legal crystal decoder. |
| Completion/reachability masking in autoregressive models | [Learning to Handle Complex Constraints for Vehicle Routing Problems (PIP)](https://arxiv.org/abs/2410.21066), [Attention Meets Reachability](https://arxiv.org/abs/2603.05540) | Future-completion masking is an established general structured-decoding principle. |
| Invariant logarithmic lattice coordinates and Wyckoff-constrained generation | [DiffCSP++](https://arxiv.org/abs/2402.03992) | Invariant log-metric coordinates and exact symmetry constraints are not individually new. |
| Space-group/Wyckoff conditional flow matching | [SGFM](https://arxiv.org/abs/2509.23822), [SymADiT](https://arxiv.org/abs/2605.10115), [SPFlow](https://doi.org/10.1016/j.mtphys.2026.102063) | We did not invent intrinsic symmetry-aware crystal flow matching or orbit-token flow. |
| Symmetry-aware Bayesian-flow crystal generation | [SymmBFN](https://arxiv.org/abs/2502.03146) | A symmetry-aware continuous/discrete generative process is not new by itself. |
| Group--subgroup and symbolic/geometry generation | [DynaCrys](https://arxiv.org/abs/2608.07401), [SbCD](https://arxiv.org/abs/2608.13457) | A two-stage discrete/continuous architecture is established. |
| Full-cell expansion followed by symmetry-preserving reduction | [DynaCrys](https://arxiv.org/abs/2608.07401) | Expand--network--project is not a wholly new pattern. |
| Periodic image-summed attention in logits and values | [Infinitely Connected Crystalformer](https://arxiv.org/abs/2403.11686) | Periodic key/value geometry is overlapping prior art that must be cited; the present equations are independently specified from our covariance/cutoff contract and create no code-level dependency. |
| Geometry-aware periodic attention for crystal generation | [Crystalite](https://arxiv.org/abs/2604.02270) | Metric-aware attention inside a crystal generator is established. |
| Masked discrete composition-conditioned CSP with symmetry tokens and stratified sampling | [MaskGXT](https://arxiv.org/abs/2606.22866) | Strong masked-generation and polymorph-coverage baselines must be reproduced; neither masked crystal tokens nor space-group-stratified candidate allocation is ours. |
| Asymmetric-unit/orbit graphs with symmetry and periodic edges | [Ruff et al.](https://doi.org/10.1039/D4DD00018H) | Orbit-level aggregation of symmetry-related contacts is established. |
| Joint lattice-coordinate flow matching without explicit symmetry charts | [FlowMM](https://arxiv.org/abs/2406.04713), [CrystalFlow](https://arxiv.org/abs/2412.11693) | Conditional flow is a modelling choice, not the novelty by itself. |
| Unit-cell flow matching with pair-bias attention, informed lattice priors, quotient alignment, auxiliary geometry losses, self-conditioning, and energy-ranked inference | [Clari](https://arxiv.org/abs/2606.03199) | These are close empirical precedents and mandatory ablation hypotheses, not components to copy. Clari conditions on molecular graphs for organic CSP, so its solve-rate numbers are not a direct species-conditioned inorganic DNG comparison. |
| Riemannian joint flow over molecular-crystal lattice, centroid torus, and rigid-body orientation | [MolCrystalFlow](https://arxiv.org/abs/2602.16020) | Native-manifold flow, data-informed lattice priors, quotient-aware transport, and post-generation energy ranking are established. Its rigid-molecule task and representation differ from atomic inorganic DNG. |
| Unified full-cell diffusion Transformers | [CrystalDiT](https://arxiv.org/abs/2508.16614), [ADiT](https://arxiv.org/abs/2503.03965) | Unified atom/lattice diffusion Transformers are established and are first-class DNG references under matched data/domain/budget. |
| Large-scale equivariant graph diffusion and property-conditioned generation | [MatterGen](https://doi.org/10.1038/s41586-025-08628-5) | MatterGen uses an equivariant graph-based denoising network rather than a Transformer; large-scale chemical-system and property-conditioned generation is established. |
| Element-type-only and exact-composition-conditioned diffusion | [UniMat](https://proceedings.iclr.cc/paper_files/paper/2024/hash/cea5ebedc384c5f4b4ef6b6eb29c197f-Abstract-Conference.html) | Generating crystals from supplied element types alone, as well as from exact ratios, is prior art; species-set conditioning is not our task novelty. |
| Chemical-system navigation by formula filtering and conditional diffusion | [Chemeleon](https://doi.org/10.1038/s41467-025-59636-y) | This is a comparison-only candidate, conditional on independent reproducibility, provenance, and licence gates; none of its components is adopted by the proposed model. |
| Chemical-system-conditioned discovery generation | [MatterGen](https://doi.org/10.1038/s41586-025-08628-5), [UniMat](https://proceedings.iclr.cc/paper_files/paper/2024/hash/cea5ebedc384c5f4b4ef6b6eb29c197f-Abstract-Conference.html), [Chemeleon](https://doi.org/10.1038/s41467-025-59636-y), and task-specific adapters above | Conditioning generation on a supplied chemical system and stability/property target is established.  Exact all-and-only species compliance and the downstream crystallographic law must be compared empirically, not claimed as an unprecedented task. |
| Reward-guided de-novo crystal search | [Chemeleon2-RL](https://doi.org/10.1038/s42256-026-01262-4) | Reward-guided improvements in MLFF-based metastability/uniqueness/novelty are established; such metrics are not interchangeable with a blinded DFT discovery endpoint. |
| Autoregressive composition/formula generation followed by conditional structure generation | Established hierarchical generative modelling and the unconditional modes of several crystal generators above | The outer laws \(p_\omega(S)p_\omega(c\mid S)P(dC\mid c)\), support-first serialization, clamped-prefix conditionalization, and chain-rule normalization are scope-enabling components, not independent novelty claims. |

Dates, versions, and exact technical overlap must be checked again immediately
before submission because this is a rapidly moving literature.

### Search summary

The present audit was last updated **1 September 2026**.  It searched arXiv and
publisher primary pages by title, abstract, and full text; followed references
and citations from the closest papers above; and used official repositories
only to determine implementation/checkpoint availability.  The query families
were:

1. `crystal generation Wyckoff autoregressive exact composition`;
2. `space group conditional flow matching crystal`;
3. `Wyckoff row element count diffusion generation`;
4. `periodic image attention crystal generator`;
5. `reachability completion mask constrained autoregressive decoding`; and
6. combinations of `Hall`, `intrinsic`, `orbit`, `quotient`, `flow`,
   `stoichiometry`, and `primitive reduction` with `crystal generation`.

The final update also ran the exact queries `CrystalDiT crystal generation
arXiv official`, `MatterGen generative model inorganic materials Nature
official paper`, and `ADiT crystal generation arXiv materials` to separate
de-novo architecture references from task-matched CSP baselines.  The
species-conditioned extension additionally used the exact queries
`chemical system conditioned crystal generation stoichiometry`,
`MatterGen chemical system energy above hull`, `crystal composition
generation autoregressive formula`, and `joint composition structure crystal
generation`.  The final benchmark audit added `UniMat element types only
conditional crystal generation`, `Chemeleon SMACT chemical system crystal`,
and `Chemeleon2 RL mSUN crystal generation`.  Preserve their full screening
artifacts before the next freeze.  The September update additionally screened
`unit cell flow matching crystal pair bias attention`, `Clari crystal flow
matching`, `molecular crystal Riemannian flow matching`, and
`MolCrystalFlow`, including their method and experiment appendices.

Inclusion required a primary paper or authoritative crystallographic source
that overlapped at least one declared mathematical component.  The audit did
not search patents, nonpublic manuscripts, or every language/database, so a
negative result is not proof of priority.  Before submission, preserve the
complete exact queries, result exports, result counts, screening decisions,
versions, and retrieval dates as a machine-readable provenance artifact.  Until
that artifact exists, this section is a search summary, not a reproducible
systematic-review log.

### Closest-method claims matrix

This compact matrix records only the distinctions relevant to the present
theory; it is not a ranking of the cited methods.

| Method | Discrete crystallographic state | Continuous geometry process | Exact supplied-formula completion at every AR step | Overlapping mixed-dimensional containment pushforward | Main overlap with this proposal |
|---|---|---|---|---|---|
| Present theory, if all gates pass | Support-first species/stoichiometry law followed by Wyckoff row--element counts | Intrinsic conditional flow | Yes, through composition and arithmetic suffix oracles | Yes; raw cover images may overlap | Proposed conjunction under audit; DNG/S-DNG chain rules are not novelty |
| DynaCrys | Legality-constrained symbolic scaffold | Symmetry-preserving geometry diffusion | Not the same supplied-formula suffix law | Not the same explicit pushforward law identified in our review | Closest integrated symbolic/geometry system |
| SGFM | Supplied space-group/Wyckoff state | Symmetry-constrained flow matching | No learned selector of the proposed form | No | Closest intrinsic-flow overlap |
| SymADiT | One token per symmetry orbit | Latent flow matching | No proposed suffix-completion selector | No | Orbit tokens plus flow |
| DiffCSP++ | Space-group/Wyckoff constraints | Symmetry-constrained continuous generation | No proposed suffix-completion selector | No | Invariant lattice and Wyckoff geometry |
| Clari | Molecular graph; no supplied space group | Full unit-cell flow matching with pair-bias Transformer | Not this formula/cover grammar | No | Closest large pair-attention/unit-cell-flow and inference-efficiency precedent for organic CSP |
| MolCrystalFlow | Supplied molecular building blocks | Riemannian lattice/torus/SO(3) joint flow | Not this formula/cover grammar | No | Closest molecular-crystal native-manifold and informed-prior flow precedent |
| SymmBFN | Symmetry-aware state | Bayesian-flow process | No proposed suffix-completion selector | No | Symmetry-aware generative process |
| WyckoffDiff | Row--element count-like state | Discrete diffusion for protostructures | No proposed locally normalized completion oracle | No | Closest count representation |

Every “No” in this table means only “not identified in the cited method under
the stated definition,” not that the method lacks other legality, symmetry, or
conditioning mechanisms.

## 5.3 The narrow candidate contribution

Adding the normalized outer composition pmf makes DNG a first-class task, and
the support-first factorization makes exact chemical-system DNG a first-class
conditional task.  Neither enlarges the novelty claim.  They are the ordinary
chain rule on a finite canonical composition space, implemented by standard
grammar-constrained autoregression.  The potential contribution remains in
the precise crystallographic synthesis described below.  The optional
learned/uniform stoichiometry mixture is likewise a standard exploration
mixture, not a new probability family.

In the primary-source search documented above, we did not identify the
exact full conjunction proposed here:

1. a locally normalized, formula-exact **row-count autoregression** whose every
   action is certified by an arithmetic suffix-reachability table;
2. a normalized probability measure explicitly defined as the pushforward of
   overlapping symmetry-containment covers of different dimensions;
3. exact continuous-origin reduction plus exchangeable treatment of repeated
   identical orbit instances; and
4. one shared intrinsic continuous flow whose vector field uses
   full-cell periodic geometry and is reduced back to legal Wyckoff tangent
   coordinates, trained with a current-state Cartesian pullback block metric.

This is a **candidate synthesis contribution**, not yet a demonstrated novel
method and certainly not yet a breakthrough.  Each item has nearby prior art;
the contribution, if any, lies in their precise compatibility and in the exact
guarantees of the combined probability model.

A safe provisional sentence is:

> We study a completion-certified, exact-formula distribution over
> element-decorated Wyckoff count tables, coupled to a mixed-dimensional
> overlapping containment-cover law and a shared intrinsic conditional flow with
> symmetry-expanded periodic geometry.  The same model supports unconditional,
> exact-species-set, and exact-formula generation through a standard
> support-first composition factorization.  Established Wyckoff, constrained
> decoding, invariant-lattice, flow-matching, and periodic-attention components
> overlap relevant prior art and are explicitly credited; the combined
> specification and future independent implementation do not import external
> model code, equation text, checkpoints, or configurations.

Do not write “the first,” “unprecedented,” “solves CSP/DNG,” or “breakthrough” until
an updated systematic search and decisive experiments support the narrower
claim.

## 5.4 What the mathematical contribution could be

The strongest theorem-level result is not a new law of physics.  It is the
coherent conjunction of these facts on a declared bounded crystal domain:

\[
\begin{aligned}
&\text{exact formula-safe discrete support},\\
&\text{positive mass on symmetry-constrained covers of different dimensions},\\
&\text{complete invariant-SPD coordinates in a fixed setting},\\
&\text{exchangeable and gauge-compatible continuous dynamics},\\
&\text{exact space-group containment under decoding}.
\end{aligned}
\]

The coverage, normalization, no-dead-end, and invariance results are meaningful
even if they ultimately do not improve a benchmark.  However, much of the
underlying mathematics—group actions, quotient measures, exponential SPD
coordinates, reachability automata, and flow matching—is classical or already
used in neighbouring models.  A paper must isolate exactly which theorem is
new relative to those sources rather than claiming the entire construction.

## 5.5 What is not solved

The method cannot prove from a supplied species set, a supplied composition,
or an unconditional composition sample:

- which polymorph is realized at a given temperature, pressure, or synthesis
  history;
- thermodynamic, mechanical, or dynamical stability;
- recovery of unobserved polymorph probabilities from a biased database;
- correctness on arbitrary unseen chemistries; or
- treatment of disorder, magnetism, incommensurate phases, or unbounded
  supercells outside the declared domain.

Exact crystallographic legality is an important constraint, not a physical
oracle.  A correct theorem here prevents invalid representations; it does not
create information absent from the input and training distribution.
Likewise, \(p_\omega(S)\) and \(p_\omega(c\mid S)\) model a declared
selected or reweighted data law, not thermodynamic prevalence over chemical
systems or stoichiometries.

## 5.6 Independent-design and implementation provenance policy

When implementation begins, it must follow this process:

1. freeze and version these equations and proof obligations;
2. generate an implementation design from this specification, not by
   transcribing another repository;
3. record every consulted paper, repository, file, licence, commit, and date in
   a provenance ledger;
4. use independently written tests and small exact oracles wherever possible;
5. isolate any third-party catalogue or utility behind a documented licence and
   verify its generated data independently;
6. run code- and text-similarity checks against the closest open-source
   projects before release;
7. preserve copyright notices and comply with all dependencies' licences; and
8. seek legal advice for patent freedom to operate if commercialization is
   contemplated.

No scientific review can guarantee absence of patent claims or certify legal
noninfringement.  It can establish transparent attribution and independent
derivation.

## 5.7 Evidence ladder for future claims

Use the following language only after the corresponding evidence exists:

| Claim | Minimum evidence |
|---|---|
| “Mathematically consistent” | Definitions close, proofs survive independent review, and no counterexample violates assumptions. |
| “Implemented correctly” | All exact-oracle and numerical obligations in `04_PROOF_OBLIGATIONS.md` pass. |
| “A novel combination” | Updated systematic literature search finds no equivalent full construction; closest work is cited and distinctions are demonstrated. |
| “Improves CSP” | Preregistered, matched-information and matched-budget benchmarks with uncertainty and ablations. |
| “Improves de-novo generation” | No-formula-supplied, matched-data and matched-budget DNG benchmarks with separate composition, structure, diversity, coverage, and cost diagnostics. |
| “Improves chemical-system discovery” | Exact same species sets, data, raw/compute/relaxation/DFT budget, competing-phase hull, and novelty evaluator; significantly more distinct DFT-verified near-hull formulas/prototypes with paired uncertainty over preregistered systems than MatterGen and traditional search baselines. |
| “Generalizes” | Composition-, prototype-, and scaffold-family-disjoint tests, followed by genuinely prospective evaluation. |
| “Finds novel materials” | Independent structural confirmation plus appropriate stability/synthesis validation outside this theory-only package. |
| “Breakthrough” | Large, reproducible practical gain and independent confirmation; never inferred from architectural elegance alone. |

## 5.8 Falsifiers of the research hypothesis

The architecture should be abandoned or simplified if any of the following is
observed:

1. catalogue regularity, FullPrim round-trip, or actual-stabilizer cover
   assignment fails on a material fraction of the declared domain;
2. residual gauge handling makes exact likelihood or stable learning
   impractical;
3. the legal selector has poor recall at the fixed candidate budget despite
   perfect formula validity;
4. the composition generator collapses, is badly miscalibrated, or
   \(p_\omega(c\mid S)\) contributes no useful held-out stoichiometry or
   fixed-budget stable-novel discovery gain over unconditional filtering and
   traditional search;
5. oracle-chart geometry is not better than a simpler full-atom baseline;
6. the current-state metric and cutoff-complete periodic-image features give no reproducible
   advantage at matched compute; or
7. a simpler published method provides the same guarantees and performance.

Negative results still validate the proof-first process: they prevent an
elegant but ineffective idea from being mistaken for predictive progress.
