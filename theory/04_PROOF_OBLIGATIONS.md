# 4. Proof obligations and implementation gates

This document converts every theoretical statement into a check that can fail.
It is deliberately stricter than an ordinary unit-test plan.  A future package
must distinguish four kinds of evidence:

- **P — analytic proof:** a mathematical argument under explicit assumptions;
- **O — exact oracle:** exhaustive or symbolic agreement on a finite problem;
- **N — numerical property test:** floating-point verification over adversarial
  and randomized examples; and
- **E — empirical experiment:** a statistical result on held-out data.

An empirical result cannot repair a failed proof obligation.  Conversely, a
proof of legality or invariance cannot establish predictive accuracy.

## 4.1 Status at the provisional unified DNG/S-DNG/CSP specification review

An alpha implementation and finite-oracle test suite now exist.  That fact does
not mark this ledger complete: an O/N obligation passes only when a
content-addressed receipt names its exact domain, tolerance, implementation,
and result.  E obligations remain unstarted because no authorized production
corpus or trained predictive checkpoint is bundled.  The analytic arguments
in `02_THEOREMS.md` remain conditional on the explicit catalogue-null,
finite-gauge, and global-flow assumptions recorded below.

## 4.1a Traceability ledger

This ledger is normative and must be updated with artifact links, frozen
domains, seeds, and results when work begins.

| Claim | Critical assumption | Invariant/test IDs | Evidence | Domain/tolerance | Status | Gate(s) |
|---|---|---|---|---|---|---|
| Primitive physical quotient | exact centering map \(C_g\), right-handed section, complete primitive reduction and gauge convention | C1--C5 | P/O/N | all supported settings; rational identities exact | implementation present; all-230 live-backend receipt incomplete | G1, G3 |
| Canonical DNG/S-DNG composition law | finite architecture-independent reduced-formula domain; support-first serialization; exact full and chemical-system-restricted live-prefix oracles; normalized outer pmf and exact conditional; identical shared downstream kernel | D0a--D0b, D3--D5, S3--S4 | P/O/N | every tiny formula and exact species fibre, then frozen production bounds | implementation and tiny oracles present; production receipt incomplete | G1, G4, G7 |
| Formula and cover support | exact \(B\); proper-null \(\Sigma_\alpha\) catalogue theorem | D1, D1b, D2--D4 | P/O | all bounded tiny fibres, then every retained cover | arithmetic implementation present; D1b table-wide catalogue certificate open | G1, G4 |
| Invariant SPD chart | exact point-group representation; bounded latent map | L1--L5 | P/O/N | every supported point group; float64 oracle | implementation and finite tests present; release receipt incomplete | G2 |
| Decoder containment | exact rational Wyckoff catalogue and centering map | W1--W6 | P/O/N | every Wyckoff row and centering type | decoder/catalogue implementation present; live all-230 gate incomplete | G1--G3 |
| Normalized CSP/S-DNG/DNG laws | standard-Borel tagged and physical spaces; measurable total decoder/projection; normalized outer, chemical-system-conditional, and inner mixtures; null defects; globally defined endpoint flow | C5, D0b, D1b, D3--D4, F1--F4, F8b, F9 | P/O/N | declared bounded composition/cover family | analytic proof provisional | G3--G6 |
| Cover-compatible flow | full-support endpoint, exchangeability, global ODE | F1--F10 | P/O/N | analytic toys then all retained covers | alpha CFM/ODE code present; F8b/F9/F10 release evidence incomplete | G5--G6 |
| Predictive superiority | matched data, information, budget, evaluator | E1--E5 | E | preregistered benchmark | not started; no claim | G8 |

## 4.2 Foundational crystallography obligations

### C1. One coordinate convention everywhere

**Claim.**  With row fractional coordinates,

\[
r=fL,\qquad G=LL^{\mathsf T},\qquad
f\mapsto fW+w,
\]

and metric invariance is \(WGW^{\mathsf T}=G\).

**Required evidence.**

- P: derive every basis-change, orbit-expansion, distance, and velocity formula
  from this convention.
- O: compare hand-computed triclinic examples with rational coordinates.
- N: random proper Cartesian rotations, right-handed integer unimodular
  changes of basis, origin
  shifts, and atom permutations must preserve all decoded distances and
  physical outputs.

**Failure condition.**  Any unexplained transpose or a componentwise
minimum-image calculation in a skew cell blocks implementation.

### C2. Catalogue integrity

**Claim.**  The pinned catalogue contains one declared Hall realization for
each supported physical space-group type, together with exact affine
operations, explicit Wyckoff rows, multiplicities, site-symmetry data, and
free-coordinate charts.

**Required evidence.**

- O: enumerate the authoritative setting and Wyckoff-row key sets independently;
  prove a bijection with the stored catalogue and match frozen cardinalities,
  so a deleted row or setting cannot pass vacuously.
- O: compare every stored operation and Wyckoff multiplicity against a pinned
  authoritative crystallographic source.
- O: close each operation set under composition and inversion using exact
  rational arithmetic modulo integer translations.
- O: confirm orbit--stabilizer multiplicities for a generic rational point of
  every free Wyckoff chart.
- N: round-trip a stratified sample from every supported row through an
  independent symmetry analyser.

The source edition, retrieval date, transformation choices, and a cryptographic
digest of the generated catalogue are part of the theorem assumptions.

### C3. Conventional-to-primitive and translation-subgroup correctness

**Claim.**  A conventional decode contains the known centering subgroup
\(T_g\).  Exact reduction through it is followed by enumeration and quotient of
the complete species-preserving translation subgroup \(T(C)\), returning the
full primitive translation lattice.

**Required evidence.**

- O: enumerate candidate translations from all same-species pair differences
  and verify the whole coloured set.  This finite oracle must find every
  translation index allowed by the atom bound, including but not limited to
  indices two and three.
- O: verify all P, A/B/C, I, F, and R centering maps, conventional/primitive
   atom counts, and exact round-trips.
- O: for every \(C_g\), verify \(C_g^{-1}\) is integral, its induced torus-map
   kernel is exactly \(T_g\), and its determinant is \(I_g^{-1}\).
- O: for every conventional Hall operation, verify under the row convention
  that \(W_{\rm dcr}=C_gWC_g^{-1}\in GL(3,\mathbb Z)\) and
  \(w_{\rm dcr}=wC_g^{-1}\pmod{\mathbb Z^3}\); recheck closure and inversion
  after transformation and verify equality of the induced infinite Cartesian
  isometries.
- N: compare the independent hidden-translation check with at least two
  crystallographic libraries over randomized standard cells.
- N: verify that primitive reduction followed by canonical standardization and
  decoding preserves the physical equivalence class.

**Failure condition.**  Atom counts and Wyckoff multiplicities may not be
interpreted until the primitive/conventional convention is fixed.

### C4. Residual gauge, chirality, and normalizer quotient

**Claim.**  Basis, origin, residual affine-normalizer, Wyckoff-parameter, and
repeated-identical-orbit representatives do not receive different physical
probabilities.

**Required evidence.**

- P: define the closed origin subtorus, its exact quotient coordinates, the
  pinned right-handed catalogue section, and every remaining finite
  within-cover representative action.
- O: enumerate every finite gauge orbit for small charts and verify that
  canonicalization has exactly one output or that likelihoods are summed over
  all preimages.
- N: apply every stored within-cover representative generator and confirm
  covariance of the field and equality of decoded physical distributions.
  Label-changing normalizer paths may remain as overlapping covers; their raw
  selector weights need not be equal.
- N: test mirror-related chiral structures in P1 and orientation-reversing
  basis counterexamples.  Only \(U\in SL(3,\mathbb Z)\) is a basis gauge; a
  mirror must remain distinct under the declared \(SO(3)\) target.

Canonical sorting alone is not accepted as a proof when it is discontinuous at
ties.

### C5. Measurable physical space, decoder, and classifiers

**Claim.**  The standard-Borel assumption in Theorem 2.5 and every pushforward
event used by the model are mathematically defined.

**Required evidence.**

- P: equip each bounded, fixed-cardinality primitive presentation with its
  Borel structure; define the physical quotient through an explicit Borel
  canonical section or prove that the declared quotient is standard Borel.
- P: prove that the reduced-composition map is measurable and that the finite
  tagged coproduct
  \(\widetilde{\mathscr X}_{\rm DNG}^\dagger
  =\bigsqcup_c\mathscr X_c^\dagger\), the untagged physical codomain
  \(\mathscr X_{\rm phys,DNG}^\dagger\), and the total map \(J^\dagger\) have
  the declared standard-Borel/measurability properties.
- P: prove that conventional decoding, complete translation enumeration,
  FullPrim, the chosen canonicalization, the total extension
  \(\bar D_\alpha^\dagger\), and every endpoint classifier are measurable.
- O: on the finite exact domain, compare all canonical representatives and
  classifier partitions under every allowed gauge transformation.
- N: verify that floating ambiguity is mapped to an explicit measurable
  ambiguity/failure category rather than resolved by data-dependent retries.

**Failure condition.**  If FullPrim or canonicalization is specified only as
an informal tie-breaking routine, the normalized pushforward theorem remains
an assumption and theory freeze fails.

## 4.3 Exact discrete-selector obligations

### D0a. Canonical de-novo composition language is exact

Freeze \(\mathcal E_{\rm DNG}\), \(K_{\max}\), the reduced-count bound, and the
exact finite set \(\mathcal C_{\rm DNG}\).  On a tiny domain, independently
enumerate every nonzero integer composition vector, reduce it by exact gcd,
and apply the element/count bounds.  The result must equal the leaves of the
canonical prefix oracle exactly.  Independently construct the universal
\(\nu=1,g=P1,h_{1a,Z}=c_Z\) witness for every leaf and verify its arithmetic
count equations and raw atom bound; downstream support must not be decided by
a learned model or benchmark-dependent filter.

The frozen serialization is support-first,

\[
s(c)=(K,Z_1,\ldots,Z_K,c_{Z_1},\ldots,c_{Z_K},\mathrm{EOS}),
\qquad Z_1<\cdots<Z_K.
\]

For every exact species set \(S=\{Z_1,\ldots,Z_K\}\) in
\(\mathfrak S_{\rm DNG}\), independently enumerate

\[
\mathcal C(S)=\{c\in\mathcal C_{\rm DNG}:\operatorname{supp}(c)=S\}.
\]

The leaves below the complete support prefix
\(a(S)=(K,Z_1,\ldots,Z_K)\) must equal \(\mathcal C(S)\) exactly.  The prefix
must be live if and only if \(S\in\mathfrak S_{\rm DNG}\).  In particular,
every accepted coefficient is positive, every terminal vector has gcd one,
and no element outside \(S\) can occur.  This must be checked from the frozen
grammar, not by generating unconditional formulas and filtering them.

Prove algebraically that \(\operatorname{red}(mc)=c\) for every positive
integer \(m\).  Under a frozen finite raw-input multiplier bound \(m_{\max}\),
exhaustively test every leaf, every permutation, and
\(1\le m\le m_{\max}\) for canonicalization to the same flat typed string with
strictly increasing elements and gcd one.  For every nonterminal live prefix,
its allowed children must be exactly those admitting at least one accepted
leaf; an EOS leaf terminates and has no successor softmax.

### D0b. Canonical de-novo composition probabilities are exact

Enumerated leaf probabilities must sum to one and agree with an independent
tree evaluator; every accepted leaf must have positive mass under finite
logits.  For every accepted \(S\), the restricted suffix probabilities must
sum to one on \(\mathcal C(S)\).  Compare the clamped-prefix evaluator against
the joint tree by the exact identity

\[
p_\omega(c\mid S)
=
\frac{p_\omega(c)}{p_\omega(S)},
\qquad
p_\omega(S)=\sum_{c'\in\mathcal C(S)}p_\omega(c'),
\qquad c\in\mathcal C(S).
\]

With fixed realized \(c\), base randomness, selector randomness, and ODE
tolerances, unconditional DNG, S-DNG, and CSP must have exactly the same
downstream path probabilities, trajectories, and decoded output.  Verify
shared parameter identity for \((\phi,\theta)\), absence of a species-set or
task token in the conditional kernel after \(c\) is fixed, and rejection of a
null or absent composition before selector scoring.

If the optional uniform-mixture exploration proposal is used, independently
count \(M_S=|\mathcal C(S)|\), verify normalization and the pointwise lower
bound \(p_{\omega,\epsilon}(c\mid S)\ge\epsilon/M_S\), and compare its
empirical \(B\)-draw hit frequencies with the exact binomial probability.
This proposal is never substituted silently for the \(\epsilon=0\)
ancestral reference law.

**Failure condition.**  An invalid, duplicate, unreduced, or dead-end formula;
an empty allowed set at a nonterminal prefix; a probability sum different from
one; a clamped suffix different from the true joint conditional; incorrect
all-and-only species support; or a task-dependent downstream conditional
invalidates DNG/S-DNG.  Report
one-time oracle construction and storage, load cost, per-token lookup/scoring,
and amortized sampling cost separately at production bounds.

### D1. Feasibility recursion is sound and complete

For each small \((g,\mathbf N)\), independently enumerate all row-count tables.  Compare
that set exactly with the accepting paths of the suffix recursion

\[
B_{g,r}(\rho)
=
\bigvee_{a\in\mathcal H_{gr}:\,\mu_{gr}a\leq\rho}
B_{g,r+1}(\rho-\mu_{gr}a).
\]

Required cases include multiple species, repeated free rows, zero-dimensional
capacity-one rows, impossible formulas, and formulas with several distinct
completions.

**Pass criterion.**  Zero false positives and zero false negatives under exact
integer arithmetic.

### D1b. Every arithmetic cover has a proper null defect locus

For every arithmetic table in the declared bounds, certify that the
collision/multiplicity-defect set \(\Sigma_\alpha\) is a finite union of proper
lower-dimensional affine or algebraic subsets of the full raw cover.

A certificate includes an exact regular witness and symbolic proof that each
listed equality is not identically true.  On tiny covers, enumerate rational
adversaries and compare against the catalogue classification.  A one-atom P1
cover is **not** rejected: it is regular but decodes to a larger actual
stabilizer.  The containment-cover model deliberately permits this.

If any arithmetic branch lacks a proper-null certificate, the declared
catalogue/domain fails before the selector is built; it is not removed by a
post-hoc sample filter.

### D2. Every local action is completion-safe

For every reachable arithmetic-prefix state, enumerate all next actions and
verify

\[
a\in\mathcal A^{\rm arith}_{gr}(\rho)
\iff
\text{there exists at least one legal suffix after }a.
\]

For stochastic property tests, freeze the domain, pseudorandom generator,
seed list, and sample count before implementation.  No path may dead-end,
violate the formula, reuse a capacity-one row, or terminate with a nonzero
residual.  Exhaustive enumeration, not random sampling, remains the oracle on
the frozen tiny domain.

### D3. The autoregressive law is normalized

On tiny fibres, enumerate all complete arithmetic tables and compute
both:

1. the product of locally masked conditional probabilities; and
2. the probability returned by an independent exhaustive tree evaluator.

Their values must agree in high precision and sum to one.  Gradients of the
enumerated log normalizer must agree with automatic differentiation and finite
differences.

This test must also demonstrate that local completion masking is a direct legal
distribution, not generally the unconstrained base model conditioned on final
legality.

### D4. Overlapping-cover event probabilities are summed

Construct tiny examples where two raw covers decode to the same physical event.
Verify

\[
P_\Theta(E\mid c)
=
\sum_\alpha\pi_\phi(\alpha\mid c)
\rho_{\theta,\alpha}\{q:\bar D_\alpha^\dagger(q)\in E\}.
\]

The actual maximal-group classifier must include ambiguity and failure values
so it defines a total measurable partition.  Its induced masses must sum to
one.  A raw cover weight may never be reported as an actual group probability.
A maximum over covers or one arbitrary preimage is not an exact physical event
probability.

For DNG, also verify the outer total-probability identity

\[
\widetilde P_\Theta^{\rm DNG}(B)
=
\sum_{c\in\mathcal C_{\rm DNG}}
p_\omega(c)P_\Theta(B_c\mid c),
\]

where \(B_c=\{C:(c,C)\in B\}\), on the tagged coproduct, and independently verify
\(P_\Theta^{\rm DNG}=J^\dagger_\#\widetilde P_\Theta^{\rm DNG}\) on the
untagged physical codomain.  Check total mass one, zero total failure mass,
and recovery of the conditional kernel after conditioning on every
positive-mass decoded reduced composition.

### D5. Complexity is measured honestly

Report reachable-state count, peak memory, transition count, and wall time as
functions of the number of present elements, conventional-cell counts, space
group, and row capacities.  The algorithm is pseudo-polynomial; no general
linear- or polynomial-time claim in the binary input length is permitted.
Report the canonical-composition prefix-oracle and Transformer costs
separately; do not hide them inside downstream selector time.

## 4.4 Lattice-chart obligations

### L1. Exact invariant subspace

For each pinned point-group representation:

- construct an invariant \(G_0\succ0\);
- verify \(Q_W=G_0^{-1/2}WG_0^{1/2}\) is orthogonal;
- solve the symmetric commutant by exact or high-precision linear algebra; and
- verify every basis matrix satisfies
  \(Q_WE_jQ_W^{\mathsf T}=E_j\).

The computed dimension must agree with an independent crystal-system
derivation.

### L2. Positive definiteness and symmetry for every parameter

For adversarial \(b\) spanning the declared numerical range, verify

\[
G_g(b)=G_0^{1/2}\exp\!\left(\sum_jb_jE_j\right)G_0^{1/2}
\]

is symmetric positive definite and satisfies \(WG_gW^{\mathsf T}=G_g\) for
every group operation.

### L3. Surjectivity round-trip

Generate invariant SPD matrices independently, encode them using the whitened
matrix logarithm, and decode.  The reconstruction error, symmetry residual,
and log-determinant error must meet the numerical tolerance.  Include nearly
degenerate but still allowed metrics to expose conditioning failures.

### L4. Basis covariance

For random \(U\in SL(3,\mathbb Z)\), transform both the group representation and
metric.  The decoded Cartesian crystal and every physical diagnostic must be
unchanged after alignment.  A chart is complete only in its pinned setting;
this test checks covariance, not a false claim of global uniqueness modulo all
bases.

### L5. Bounded latent map and lattice pullback

Verify \(b=\kappa_g(z)\) is a smooth bijection from \(\mathbb R^{d_g}\) onto the
declared open coefficient ball, and that every decoded metric meets the scale
and conditioning bounds.  Compare automatic differentiation of
\(\partial G/\partial z_j\) with a high-precision Fréchet derivative of the
matrix exponential, then verify

\[
[H_z]_{jk}
=
\operatorname{tr}\!\left(
G^{-1}\partial_jG\,G^{-1}\partial_kG
\right)
\]

is symmetric positive definite and matches finite-difference path lengths.

## 4.5 Wyckoff-coordinate and decoder obligations

### W1. Intrinsic parameter lattice is exact

For \(u(\xi)=a+\xi A\), compute

\[
\Lambda=\{\lambda:\lambda A\in\mathbb Z^3\}
\]

using exact integer normal-form methods after rational scaling.  Do not infer
periods by a floating-point or greedy search.  Verify that lattice translations
decode identically and that no smaller missing translation exists within an
exhaustive bounded oracle.

### W2. Generic and specialization loci are distinguished

For every free row:

- sample rational generic parameters and verify the declared multiplicity;
- enumerate known boundary/specialization examples;
- distinguish \(a\) a true site-symmetry/Wyckoff-row switch or multiplicity
  collapse, \(b\) enhanced inherent symmetry of one orbit without a row change,
  and \(c\) enhancement of the complete decorated crystal;
- reclassify the branch only when the **complete coloured stabilizer** changes;
  and
- test joint enhancements caused by several otherwise generic orbits or by the
  metric.

### W3. Formula, orbit, and group-containment decoder theorem

For every catalogue row and randomized legal count table, decoding must give
the exact declared conventional species counts and orbit multiplicities,
invariant SPD metric, and contain every selected operation in the symmetry
group of the infinite coloured point set.  FullPrim must then give counts
\((\nu/J(C))c_Z\) and reduced composition \(c\).

Use exact rational arithmetic for the affine identities and high precision for
the metric identities.  Group **equality** is tested separately in W4.

### W4. Independent maximal-stabilizer verification

Before any centering or primitive reduction, an independent exact verifier must
determine the raw-conventional pure-translation subgroup
\(T_{\rm raw}(C):=T(C)\subset\mathbb T^3\) from equation (1.8) and compare it
with the selected setting's
declared \(T_g\).  It must then apply \(\operatorname{CentRed}_g\) and
\(\operatorname{FullPrim}\), determine the complete coloured stabilizer in the
resulting primitive basis, and only then conjugate that actual group to its own
pinned conventional setting.  After FullPrim, translations are trivial modulo
the primitive lattice; the centering subgroup of the actual group's pinned
conventional setting is not the original raw \(T_{\rm raw}(C)\).  Test
deliberately constructed cases where:

- a nominal P1 one-atom scaffold has unavoidable higher symmetry;
- a metric specialization enhances the point group;
- a coordinate specialization changes a Wyckoff row;
- several orbits jointly create a supergroup; and
- a hidden translation reduces the cell.

The raw containment cover and the actual post-decode stabilizer are both
reported.  Larger symmetry or extra translations are valid outcomes, not
rejection reasons.  Only collision/multiplicity defects or failure of exact
FullPrim are conformance failures.

### W5. Exact periodic geometry

For skew-cell adversaries, compare the closest-image solver against exhaustive
integer enumeration with a certified radius.  For cutoff neighborhoods,
enumerate **all** lattice translations inside the cutoff, not only one closest
image.  Boundary conventions must count each image exactly once.  Verify the
central self-edge is excluded, nonzero periodic self-images are retained, and
the dimensionless compact cutoff satisfies
\(\chi(\widehat r_c^2)=\chi'(\widehat r_c^2)=0\) so list changes preserve
field continuity.

### W6. No-overlap checks are not physics theorems

Numerical tests may verify collision detection and optional distance
diagnostics.  Any covalent-radius or minimum-distance threshold must be labeled
an empirical data policy, not a consequence of crystallography.

## 4.6 Continuous-flow obligations

### F1. Base and endpoint measures have the required support

Repeated identical orbit instances receive identically distributed base
variables and no index embeddings.  Random permutations must leave the base
law and endpoint kernel invariant.  The base is Gaussian times normalized Haar;
the endpoint kernel is Gaussian in \(z\) times a strictly positive torus heat
kernel.  Repeated-orbit symmetry is enforced by exchangeability/uniform block
shuffling, while only the explicitly bounded residual group is averaged.
Both densities must be positive almost everywhere on the full cover.  Residual
finite gauges must either be fixed by a proved section or averaged with the
correct differential action.

### F2. Finite gauge-lift coupling is exact

For small repeated-orbit blocks, exhaustively enumerate permutations and stored
finite residual actions.  Verify that uniform block shuffling followed by the
frozen residual action order induces exactly the uniform finite-image mixture
declared in section 1.10 and leaves both endpoint marginals invariant.  Verify
\(\gamma\Pi_\alpha\gamma^{-1}=\Pi_\alpha\) for every residual generator.  No
matching cost or approximate assignment is part of version 1.

### F3. Cover path and endpoint regularity

For each raw cover, the selected lift \(\Delta\eta\) must be a genuine
intrinsic displacement, with a deterministic measurable tie rule.  Every
interpolated point

\[
\eta_t=\eta_0+t\Delta\eta\pmod\Lambda
\]

remains on the intrinsic affine cover.  It may cross a measure-zero
specialization or collision locus.  Prove that the defect locus is proper and
null, the endpoint has full support, and the global diffeomorphic flow
preserves null probability.  Prove separately that the random-time
interpolation marginal is absolutely continuous after the finite random gauge
lift if any training-time null-set claim uses that fact.  Equation (1.32) wraps exact
intrinsic lattice-basis coefficients only; it must never be described as a
physical minimum-image calculation in a skew fractional cell.

### F4. Set and residual-gauge equivariance

For every layer and full vector field,

\[
v(t,\gamma q,c,\alpha)=d\gamma_q\,v(t,q,c,\alpha).
\]

Check random permutations of identical orbits and every stored finite
within-cover representative generator after the continuous origin subtorus is
quotiented.  The
errors must be evaluated before any output
canonicalization so that a broken architecture cannot be hidden by sorting.
For every intrinsic period-lattice basis generator, verify that the character
features and their derivatives are identical under
\(\xi\mapsto\xi+\lambda\).  Verify that the retained Fourier-mode set is closed
under every residual representative generator.

### F5. Expand--perceive--reduce exact adjoint is correct

For random legal cells and arbitrary expanded-atom velocities, compare the
closed-form intrinsic reduction with a high-precision least-squares oracle.
Prove equation (1.15a), verify constant full column rank of
\(P_NJ_\alpha^{\rm cart}(G)\) over the full advertised-slot cover, and test the
normal equations, gradients, and finite residual-group covariance.
Expanding the reduced velocity back to the cell must remain in the legal
Wyckoff tangent subspace.  A ridge that hides a failed origin-kernel certificate
is forbidden.  Re-expressing the same cell in \(\text{\AA}\) and nm must leave
the dimensionless loss and intrinsic velocity unchanged.

### F6. Dense scalar-invariant periodic attention is smooth and covariant

Fractional-torus characters, invariant pair channels, scalar attention,
deterministic direct, reciprocal, cross-product, and pinned lattice-frame
carriers, Cartesian vector outputs, and final intrinsic
velocities must be unchanged or transform covariantly under:

- origin shifts;
- atom and identical-orbit permutations;
- Cartesian rotations;
- equivalent orbit representatives.

Mirror-related chiral P1 structures must remain distinguishable.  Test
parity-odd/oriented channels explicitly so an accidentally \(O(3)\)-invariant
implementation cannot pass an \(SO(3)\)-only specification.

The Cholesky lattice section must remain smooth through repeated metric
eigenvalues.  Compare it with another right-handed factor and verify that the
proper-rotation-equivariant physical field agrees.  Re-express all lengths in
\(\text{\AA}\) and nm and verify identical dimensionless features, messages,
losses, and intrinsic velocities.  Verify exact integer-cell periodicity for
every retained Fourier mode and complete enumeration of the configured mode
cube.  Any within-cover basis-changing action must preserve the mode set or be
rejected before training.  Bounded smooth neural branches and collision-safe
covariant normalization are required on the closed declared domain.

Include a general \(P\bar1\) orbit test: the two symmetry-related Cartesian
vectors must reduce to a nonzero arbitrary legal intrinsic velocity.  Also
include a one-orbit crystal test.  A logit-only geometric bias cancels when
there is only one key; the directional value pathway and pinned lattice-frame
carriers must still transmit lattice geometry.  At collisions, the sine-odd
direct and reciprocal carriers and their cross products must tend smoothly to
zero, while the normalized rows of the nonsingular pinned lattice section remain
finite, nonzero polar carriers.  Every branch must have finite gradients.
Packed dense evaluation must equal independent evaluation exactly
within numerical tolerance and must contain no cross-crystal attention.

### F6b. Full-pipeline basis covariance

Arbitrary \(U\in SL(3,\mathbb Z)\) need not act inside one pinned raw cover.
Therefore do not require a single raw-cover field evaluation to transform
under every such \(U\).  Instead, apply the basis change to the physical input,
then run the complete canonicalize--encode--flow--decode--FullPrim pipeline.
After physical alignment, its law and every invariant diagnostic must agree
with the original pipeline.  Within-cover covariance is tested only for the
stored finite actions in F4 and F6.

### F7. The legacy cutoff/image ablation has a completeness certificate

When `equivariant_local` is selected, derive a finite integer search bound from
the metric for every pair and cutoff
and compare against brute force on highly skew cells.  Prove a worst-case image
count under the declared metric bound and provision the implementation for
that scope.  A runtime resource cap that discards a positive-probability sample
changes the model law and is a correctness failure.

### F8. Current-state weighting preserves the CFM population target

On analytically solvable mixtures, verify that minimizing

\[
\mathbb E[(v-u)^{\mathsf T}W(q_t)(v-u)\mid q_t]
\]

returns \(\mathbb E[u\mid q_t]\).  Construct a counterexample showing that a
target-dependent \(W(q_1)\) generally returns
\([\mathbb E W]^{-1}\mathbb E[Wu]\) instead.  The latter must never be described
as ordinary conditional flow matching.

### F8b. Conditional paths transport the intended marginal

On analytic examples, verify the conditional continuity equation, the identity
\(v_t(q)=\mathbb E[u_t\mid q_t=q]\), and the marginal continuity equation from
Theorem 2.9b.  Compare forward ODE samples and moments with the known endpoint,
and integrate forward then backward against a high-accuracy oracle.

The final training target must have a strictly positive density on its cover.
Record the exact Gaussian--torus-heat dequantization kernel and verify positivity,
normalization, symmetry preservation, and exchangeability.  A zero-noise finite
empirical endpoint or a component-truncated kernel is not accepted under the
diffeomorphic-flow theorem.

### F9. ODE existence and endpoint conformance

Prove local Lipschitz continuity plus at-most-linear growth, or a stronger
global Lipschitz condition, so the ODE
exists uniquely for all \(t\in[0,1]\).  Across integration tolerances and
adversarial velocities, every intermediate metric must remain inside the
bounded SPD chart and every intrinsic coordinate on its parameter torus.
The proof must explicitly use the \(C^1\) cutoff/branch conditions and uniform
self-denominator bound from F6 rather than only checking pointwise continuity.
Selected-group containment holds throughout the cover.  Larger actual symmetry
or extra translations are valid post-decode outcomes; collision or orbit
multiplicity failure is a conformance failure, not a sample to repair silently.

### F10. Tiny-distribution oracle

Construct a one-dimensional lattice chart and one- or two-orbit torus examples
whose target densities and transports are analytic or numerically integrable.
Compare learned conditional velocities, integrated samples, moments, and
density where available.  This is the first end-to-end statistical test; a
large materials benchmark cannot replace it.

## 4.7 Data and scientific-integrity obligations

### S1. Preprocessing round-trip

For every retained structure,

\[
C\longrightarrow(c,\alpha,q)\longrightarrow\bar D_\alpha(q)
\]

must recover the same physical equivalence class.  The assigned (g,h) is
derived from the complete actual stabilizer so that within-cover
dequantization preserves the datum's exact symmetry.  Report all exclusions
and ambiguity cases.  Split the data before fitting any learned statistics.

### S2. Symmetry-tolerance sensitivity

Repeat preprocessing over a predeclared tolerance grid.  Report the fraction
whose maximal group, Wyckoff decomposition, primitive cell, or chart dimension
changes.  Do not choose the tolerance using test-set match rate.

### S3. No target leakage

For reduced-formula CSP, the input provenance trace must prove that target atom
count, space group, formula-unit multiplier, lattice, coordinates, prototype
ID, and target-derived descriptors are absent.  \(\nu\) and the chart must be
predicted or marginalized.  A count supplied in the same pinned conventional
convention may replace \(\nu\); primitive-count-only conditioning is excluded
from version 1 and may not be simulated by post-filtering.

For unconditional DNG, no species set, target formula, or structure may be
supplied.  The sampled composition is an internal random variable from
\(p_\omega\), not an input.  Feeding benchmark formulas and calling the run
de novo is target leakage; it is permitted only as an explicitly labelled
oracle-composition diagnostic.

For S-DNG, the sole structural condition is the exact unordered species set
\(S\).  It means all and only the elements in \(S\) occur.  The target
stoichiometric coefficients, atom count, \(\nu\), space group, prototype,
lattice, coordinates, and target-derived descriptors are forbidden.  Every
baseline in the S-DNG track receives precisely the same \(S\).  An
allowed-palette query \(\operatorname{supp}(c)\subseteq S\) is a different
task and must never be reported as exact-system S-DNG.

### S4. Split integrity and polymorph accounting

Deduplicate physical structures before splitting.  Provide composition-,
prototype-, and scaffold-family-disjoint evaluations.  Multiple observed
polymorphs of one composition must not be silently treated as contradictory
labels or leaked across an OOD split.

The discovery-oriented S-DNG checkpoint uses the system-, formula-, and
polymorph-balanced measure (3.15a).  A separate record-weighted checkpoint is
retained for unconditional database-law modelling.  Any alternative weighting
is preregistered as an ablation.  If the two composition factors use different
weighting measures, call the objective a composite search-prior objective
rather than a single joint maximum-likelihood fit.  Report train/test formula and exact-
system overlap; separate seen-system/unseen-stoichiometry, held-out-system, and
unconditional DNG results.  No database reweighting may be described as a
thermodynamic correction.

### S5. Provenance and independent implementation

Every source used for equations, catalogue generation, test fixtures, or code
design receives a citation, version, licence record, and digest.  A future
implementation is derived from this specification rather than copied from
another repository.  Run a code/text similarity audit before release; this is
necessary evidence but not a legal opinion.

Enforce a one-way firewall between core development and comparison systems:

- the core model may use only this frozen specification, authoritative
  crystallographic data, and independently implemented general numerical
  primitives;
- external generators are invoked only through isolated, versioned benchmark
  adapters and cannot contribute source code, weights, equations, prompts,
  configurations, or model-specific preprocessing to the core;
- every core file receives a provenance header linking it to a specification
  obligation rather than an external repository; and
- a baseline without a licence-compatible, reproducible artifact remains a
  paper/reference result and is not reimplemented from its source code.

This firewall does not permit relevant prior work to be omitted from the
literature record.  Citation establishes the claim boundary; it does not make
the cited method an architectural dependency.

## 4.8 Empirical obligations

Only after C1--S5 pass may predictive experiments begin.

### E1. Selector decomposition

For DNG, report held-out canonical-formula NLL, calibration, support validity,
uniqueness, and formula/structure novelty relative to one frozen reference
corpus, plus element/count marginals and mass by formula cardinality.
Separate failures caused by composition selection, cover selection, and
geometry.  An oracle-composition run drawn from one frozen empirical formula
marginal isolates the shared downstream model but is not a DNG headline.

For S-DNG, additionally report conditional stoichiometry NLL and calibration,
exact-system compliance, stoichiometry entropy and coverage, held-out
stoichiometry recall, frozen-reference near-hull formula recovery under the
external evaluator, and macro-averages over systems.  This is not a claim that
the complete set of stable formulas is known.  Exact-system
compliance must be one under ancestral sampling; any missing requested element
or added element is a model conformance failure, not a validity-filter event.

Report canonical cover-label recall and calibration at fixed candidate budgets,
stratified by \(\nu\), selected group, actual post-decode group, composition
cardinality, and frequency.  Also report the summed physical event
contributions from every overlapping cover.  Compare predicted covers with an
actual-stabilizer oracle-cover geometry experiment.  Cover-label recall is a
diagnostic, not a hard ceiling on structure match.

### E2. Geometry decomposition

Condition on the actual-stabilizer cover and report lattice, orbit-coordinate,
Cartesian, and StructureMatcher errors before any relaxation.  Compare the
current-state block metric with an unweighted control at matched capacity, and
compute error stratifications by cell skewness, orbit multiplicity, and
free-coordinate dimension.

### E3. Fixed-budget end-to-end evaluation

Use exactly the same training split, input information, validity policy,
matcher settings, and external post-processing for every baseline.  The raw
architecture track uses the same one-sample and fixed-\(K\) attempt budgets;
the discovery track instead uses the componentwise resource budget defined
below.  These constraints are not imposed simultaneously.  Count selector
failures and invalid decodes as failures.  Freeze two distinct hypotheses
before seeing test results:

1. **architecture fidelity:** raw, unrelaxed accuracy, validity, coverage, and
   cost under matched data and inputs; and
2. **discovery superiority:** distinct DFT-near-hull, post-relaxation novel
   structures found under a matched total end-to-end resource budget.

The first isolates the generator.  The second is the decisive endpoint for a
claim that the complete system beats MatterGen for novel-candidate discovery.
Neither result may be substituted for the other.  Preregister either one
primary \((\mathbf B_{\rm cost}^\star,\epsilon^\star,
\delta_{\rm nov}^\star,\delta_{\rm dup}^\star)\) or a
fixed area-under-discovery-versus-budget functional; all additional thresholds
are multiplicity-controlled secondary analyses.

External systems execute outside the core dependency graph through isolated
adapters.  Their outputs enter only the architecture-neutral evaluator; their
code, weights, preprocessing, equations, configurations, and tuning choices
cannot modify the proposed model.

Version 1 has three separate primary tracks.

**Reduced-formula CSP.**  Standard
MP-20-style CSP results that condition on the target primitive-cell composition
or atom count are a different task and are not directly comparable.  Therefore
the primary track must retrain every available baseline from the same reduced
formula and withhold target atom count, \(\nu\), space group, and structure.
Published headline numbers may be reproduced in a separately labelled
reference track but cannot support a superiority claim.  A direct standard
MP-20 comparison becomes valid only after a future exact primitive-count
extension has normalized structurally compatible support; post-filtering is
forbidden.

The preregistered comparison set must include the closest exact/symmetry-aware
methods and the strongest reproducible composition-only systems current at
freeze time.  The initial CSP list is CrystalFormer, DiffCSP, DiffCSP++, FlowMM,
SGFM, DynaCrys, NextCrystal, Crystalite, and MaskGXT when each method can be
retrained for the same reduced-formula task.  For each method, record the official
repository URL, commit, licence, checkpoint, training data, supplied conditions,
sampler budget, and whether an official artifact is actually available.  A
paper-only result is reported as such and is not silently replaced by an
unvalidated reimplementation.

**Chemical-system-conditioned de-novo generation.**  Every method receives the
same exact unordered element set \(S\), no stoichiometric coefficients, and no
target structure.  Outputs must contain all and only the supplied elements.
The initial comparison set must include
[MatterGen](https://doi.org/10.1038/s41586-025-08628-5),
[UniMat](https://proceedings.iclr.cc/paper_files/paper/2024/hash/cea5ebedc384c5f4b4ef6b6eb29c197f-Abstract-Conference.html),
[Chemeleon](https://doi.org/10.1038/s41467-025-59636-y), the strongest
reproducible chemical-system generator current at benchmark freeze,
task-compatible chemical-system adapters, traditional substitution/ion-
exchange or prototype enumeration, and random/evolutionary/AIRSS-style
structure search wherever a matched implementation is feasible.  Chemeleon's
SMACT formula allocator is method-native baseline behavior evaluated only in
its isolated reference column; its
oxidation/electronegativity rules are external side information: either every
matched pipeline receives the same filter or it appears only in the
native-resource column.  Formula-oracle runs of geometry models are
diagnostics, not S-DNG results.

The matched architecture column supplies only \(S\) to every model unless a
second, explicitly property-conditioned experiment gives every method the same
property information.  MatterGen's native chemical-system-plus-hull condition
is therefore an important official-reference result, but it is not silently
treated as a same-input architecture comparison against an S-only model.

Run at least three MatterGen evidence columns: (i) matched-data S-only
retraining on identical structures, formula/system weighting, atom bounds, and
input contract; (ii) the official S-only `chemical_system` artifact; and (iii)
the official/native `chemical_system_energy_above_hull` discovery pipeline.
The first supports an architecture comparison.  Beating MatterGen as a
discovery system requires the third under an equal downstream resource budget.
Keep paper-reference results separate from released-artifact results when the
released checkpoint is not the paper checkpoint.  Every public or retrained
output is re-evaluated by one common pipeline.
First reproduce the chemical-system suite used by the MatterGen publication,
then evaluate a preregistered blinded suite stratified by element-set
cardinality, data frequency, unconventional stoichiometry, and exact systems
withheld from training.  The headline v1 suite excludes systems whose ordering
is dominated by unresolved magnetism or strong correlation.  A separately
labelled rare-earth/magnetic track is allowed only with one frozen spin-state
initialization/enumeration, \(+U\)/correction, spin--orbit, pseudopotential,
pressure, convergence, and reference-phase protocol applied to every method;
all trials count toward cost.  The structural generator remains spin-agnostic
and therefore makes no magnetic-ground-state claim.  Confidence intervals are
paired over chemical systems, not over correlated structures from one system.

The primary discovery endpoint is the number of distinct relaxed discoveries
under a fixed total end-to-end resource budget vector.  For every action
\(a\), freeze the dimensionally typed cost

\[
\mathbf c(a)
=
(\text{GPU h},\text{CPU h},\text{ML-relaxer calls},
  \text{DFT submissions},\text{DFT core h}),
\]

and a wall-clock deadline.  For method \(m\), let
\(\tau_m(S;\mathbf B_{\rm cost})=(a_1,\ldots,a_T)\) be the complete protocol
trace, including successful and failed generation, deduplication, surrogate
relaxation/ranking, and DFT actions, with

\[
\mathbf C(\tau_m)=\sum_{t=1}^T\mathbf c(a_t)
\preceq\mathbf B_{\rm cost}.
\]

The run stops when the next declared action would exceed any component or the
wall-clock deadline; resources cannot be converted between coordinates after
results are observed.  A separately reported sensitivity analysis may use one
dimensionally explicit, preregistered price vector, but it is not the primary
stopping rule.

Let \(\mathcal X_{\rm DFT}(\tau_m)\) be every candidate submitted to DFT.
Define one total relaxation map
\(R_{\rm DFT}:\mathscr X\to\mathscr X\sqcup\{\dagger\}\); every numerical,
electronic, ionic, or cell nonconvergence maps to \(\dagger\), remains in the
denominator, and retains its full cost.

Freeze a comprehensive known-material corpus \(\mathcal D_{\rm known}^{t_0}\)
at a prospective cutoff \(t_0\), distinct from the training set
\(\mathcal D_{\rm train}\).  Before unblinding, construct the required ordered,
full-occupancy reference input set \(\mathcal R_{\rm raw}^{t_0}(S)\) from every
nonempty subsystem of \(S\).  Apply the same total \(R_{\rm DFT}\), structural
and cell relaxation, spin/correlation policy, primitive standardization, and
post-relaxation deduplication to references and generated candidates.  If any
member of the frozen required reference set fails, the entire system is
declared unscorable for every method before generation rather than silently
raising the hull by deleting that phase.  Systems requiring unresolved partial
occupancy or disorder are outside the headline v1 domain unless one frozen
ordered-approximant protocol is supplied for all methods.

For novelty, use the prospectively frozen standardized corpus

\[
\mathcal D_{\rm nov}^{t_0}
=
\operatorname{Canon}
\left(\mathcal D_{\rm known}^{t_0}
\cup R_{\rm DFT}(\mathcal R_{\rm raw}^{t_0})\right),
\]

so candidates are compared with both deposited known structures and their
uniformly relaxed reference representatives.  For a converged structure
\(y\), let

\[
h_{p_0}^{0K}(y)
=
\frac{E_{\rm DFT}(y)+p_0V(y)+\kappa(y)}{N(y)},
\qquad
x_Z(y)=\frac{N_Z(y)}{N(y)},
\]

where \(\kappa\), the functional, pseudopotentials, magnetic policy, and
pressure \(p_0\) are frozen.  For a uniformly recomputed reference set
\(\mathcal R\), define the zero-temperature static-enthalpy hull functional

\[
H_{\mathcal R}(x)
=
\min_{\substack{\lambda_r\ge0,\ \sum_r\lambda_r=1\\
                 \sum_r\lambda_r x(r)=x}}
\sum_{r\in\mathcal R}\lambda_r h_{p_0}^{0K}(r),
\qquad
\Delta h_{\mathcal R}(y)
=h_{p_0}^{0K}(y)-H_{\mathcal R}(x(y)).
\tag{4.1}
\]

For every retained exact system \(S\), define
\(\mathcal R_{\rm known}^{t_0}(S)\) as the converged, post-relaxation,
deduplicated image of \(\mathcal R_{\rm raw}^{t_0}(S)\); it includes all
elemental references.  Report both
\(\Delta h_{\rm known}=\Delta h_{\mathcal R_{\rm known}^{t_0}}\) and a blinded
pooled value \(\Delta h_{\rm pool}\) whose reference set adds every converged,
uniformly recomputed candidate from every compared method before constructing
one common hull.  The pooled hull prevents several mutually dominated
discoveries from all being credited merely because each lies below the old
known-phase hull.

Freeze a pairwise structural dissimilarity \(d_{\rm str}\), a novelty
threshold \(\delta_{\rm nov}\), a duplicate threshold \(\delta_{\rm dup}\),
and a total ordering \(\prec\) independent of method identity.  Let
\(\operatorname{Dedup}^{\prec}_{\delta_{\rm dup}}\) sort by \(\prec\) and
greedily retain a structure only if it is farther than
\(\delta_{\rm dup}\) from every retained structure.  This is deterministic
even when the tolerance matcher is not transitive.  Define

\[
\begin{aligned}
\mathcal Y_m(S;\mathbf B_{\rm cost},\epsilon)
=\{\,y=R_{\rm DFT}(x):\;&x\in\mathcal X_{\rm DFT}(\tau_m),\ y\ne\dagger,\\
&\operatorname{supp}(r(y))=S,\quad
\Delta h_{\rm pool}(y)\le\epsilon,\\
&\min_{z\in\mathcal D_{\rm nov}^{t_0}}
d_{\rm str}(y,z)>\delta_{\rm nov}\,\},\\
D_{\epsilon,\delta_{\rm nov},\delta_{\rm dup}}^{(m)}
(S;\mathbf B_{\rm cost})
=\left|
\operatorname{Dedup}^{\prec}_{\delta_{\rm dup}}
\mathcal Y_m(S;\mathbf B_{\rm cost},\epsilon)
\right|.
\end{aligned}
\tag{4.2}
\]

Report structure, anonymous-prototype, and reduced-formula novelty separately;
"unseen in training" is not novelty if the material was already present in
\(\mathcal D_{\rm nov}^{t_0}\).  Show discovery curves against raw attempts,
model NFE/FLOPs/wall time, ML-relaxer calls, DFT submissions, and DFT core-hours
so no favourable scalar cost conversion can determine the result.  Shared
training cost and the one-time common reference-phase recomputation cost are
reported separately; neither is charged selectively to one generator.

Use one frozen DFT protocol, primitive reduction, matcher, novelty rule, and
reference corrections for every method.  Preregister one primary
\(\epsilon^\star\); report \(\epsilon\in\{0,20,50,100\}\) meV
atom\(^{-1}\) only as secondary sensitivity analysis.  Also report pooled-hull
vertices, success-at-\(K\) under a frozen definition, DFT-near-hull formulas
and anonymous prototypes, macro-average and lower-decile system performance,
cost per discovery, initial forces/stress, relaxation displacement and
convergence, formula entropy, and symmetry survival.  This is a 0-K
static-enthalpy screen, not a proof of phonon stability, finite-temperature
stability, synthesizability, or kinetic accessibility.  Raw S-DNG quality
before relaxation remains a separate architecture diagnostic.  SUN may be
reported, but it is secondary and cannot replace this fixed-budget DFT
discovery analysis.

Let \(\tau_m^{(K)}(S)\) denote the same trace stopped immediately after its
\(K\)-th DFT submission.  Define success-at-\(K\) before evaluation as the
macro-average over systems of the indicator that the corresponding deduplicated
set \(\mathcal Y_m(S;\mathbf C(\tau_m^{(K)}),\epsilon^\star)\) is
nonempty.  Freeze the species-erased standardization used
for anonymous-prototype identity and its tolerance separately from structure
novelty.  Confidence intervals and paired permutation/bootstrap tests resample
chemical systems as clusters.  Freeze the single primary endpoint and the
correction for all secondary budgets, thresholds, and strata before unblinding.

**Unconditional de-novo generation.**  Every method receives no formula,
species set, or structure condition and uses the same training structures,
element vocabulary, formula bounds, raw attempt count, deduplication policy,
and compute-budget reporting.
Every raw output passes through the same declared primitive reduction,
reduced-formula extraction, membership test in the architecture-independent
\(\mathcal C_{\rm DNG}\), physical target-domain classifier, and deduplication
pipeline.  The primary common physical domain is defined independently of this
architecture by the actual primitive atom count, element vocabulary,
three-dimensional periodicity, ordered full occupancy, and frozen physical
volume/metric bounds.  This model's pinned-cover raw atom and
\(\nu_{\max}\) limits are reported separately as internal resource bounds and
cannot invalidate an otherwise in-domain baseline output.
Uncanonicalizable, out-of-formula-domain, and out-of-target-domain
outputs remain in the all-attempt denominator and are separately reported;
they are never silently filtered away.
The initial DNG comparison set includes CrystalDiT, MatterGen, ADiT,
CrystalFormer's unconditional mode, FlowMM's DNG mode, and task-compatible
DynaCrys, Crystalite, MaskGXT, Chemeleon2-RL, or later systems when
reproducible artifacts or matched retraining are available.  Reward-guided
MLFF metrics from Chemeleon2-RL are not equated with the DFT endpoint above.
Report composition-prior metrics separately from raw structural validity,
uniqueness, frozen-corpus formula/structure novelty, held-out coverage/precision,
space-group/composition distributions, and inference cost.  Separate seen-
formula and unseen-formula results.  Supplying held-out formulas is an
oracle-composition diagnostic, not DNG.

Freeze a side-information ledger for every track: elemental descriptors,
prototype libraries, SMACT or oxidation rules, pretrained MLIPs, property
labels, external pretraining, and database cutoffs.  Matched comparisons use
identical allowed side information; method-native resources are reported in a
separate column.

Freeze temperature, top-\(p\), guidance, search, and all other sampling
hyperparameters using validation data only.  Report ancestral temperature-one
sampling separately from guided or searched generation, use equal
hyperparameter-tuning budgets, and show fixed-attempt plus matched-NFE/FLOP/
wall-time curves.

Maintain separate evidence columns for **matched retraining** on the frozen
data/input contract and for **official artifact/reference** evaluation.  A
checkpoint trained on external or overlapping data is never presented as a
matched architecture comparison.

### E4. Necessary ablations

At minimum:

1. support-first exact \(p(S)p(c\mid S)\) versus the former interleaved
   serialization and unconditional generation followed by system filtering;
2. exact live stoichiometry grammar versus rejection or repair;
3. formula-deduplicated/system-balanced versus structure-record-weighted
   composition training;
4. learned \(p_\omega(c\mid S)\) versus the explicit uniform-mixture
   exploration proposal at frozen mixture weight;
5. a factorial
   **formula allocator \(\times\) formula-conditioned structure engine** test:
   learned \(p_\omega(c\mid S)\), empirical, exact-uniform, and SMACT
   allocators crossed with the proposed geometry model and compatible
   CSP/RSS/prototype engines, plus a frozen oracle-composition diagnostic;
6. completion-certified selector versus a matched unconstrained/rejection
   baseline;
7. canonical cover labels versus finite equivalent-label marginalization;
8. unordered equivariant flow versus indexed orbit tokens;
9. CFM versus one preregistered diffusion construction on the identical
   \((z,\eta,\alpha,c)\) state and Transformer: Gaussian noise on \(z\),
   Brownian/heat noise on each intrinsic torus, frozen schedule, score or
   velocity parameterization, reverse-SDE and probability-flow choices,
   identical data/parameters/tuning FLOPs, solver tolerances, and matched
   NFE/FLOP/wall-time curves; no intrinsic accuracy advantage is assumed for
   either process;
10. current-state Cartesian pullback weighting versus Euclidean weighting;
11. finite dense Fourier features versus complete-cutoff periodic-image
    features;
12. logit-only versus value-only versus logit-and-value geometry;
13. dense invariant-scalar/covariant-readout Transformer versus both the full
    \(SE(3)\)-equivariant local graph and an AF3-style raw-coordinate
    non-equivariant Transformer under parameter- and compute-matched tracks;
14. diversity- and uncertainty-aware fixed-budget external ranking versus
    random, lowest-predicted-energy, and \(\beta_{\rm unc}=0\) selection, with
    common-controller and method-native-controller results kept separate;
15. standard Gaussian/fixed-volume lattice source versus a full-support
    composition-conditioned source fitted only on the training split, with
    path length, target-velocity variance, validity, calibration, and held-out
    likelihood proxies reported;
16. uniform versus preregistered beta/logit-normal time sampling, and no
    self-conditioning versus endpoint self-conditioning, under matched training
    FLOPs;
17. plain CFM versus symmetry-compatible endpoint volume and complete-periodic
    pair/clash auxiliary losses.  The periodic target must use the certified
    image/CVP machinery rather than a fixed \(\{-1,0,1\}^3\) image cube;
18. Euler, Heun, RK4, and—if used—adaptive integration on identical checkpoints,
    reporting quality against NFE, wall time, memory, and forward/backward
    consistency.  Solver choice and schedule are part of every checkpoint and
    generation-protocol identity; and
19. finite-cutoff periodic graph attention versus a compute-matched model with
    an explicit global/reciprocal information path, measuring disconnected-
    motif and long-range electrostatic counterexamples as well as standard
    retrieval metrics.

### E5. Claims that remain prohibited

Neither legality nor benchmark match permits claims of mechanical stability,
dynamical stability, thermodynamic stability, synthesizability, or unrestricted
OOD discovery.  Those require additional physical information and experiments
outside the declared scope.

In this document, **unconditional DNG** means that no species set, composition,
or structure is supplied.  **S-DNG** means that only an exact species set is
supplied and the stoichiometry and structure are generated.  Neither term by
itself means unseen, novel, stable, or synthesizable.

## 4.9 Frozen implementation and release gate matrix

Before code design, freeze the following finite reference domains:

- **Composition oracle domain:** a small exact element alphabet, all canonical
  gcd-one formulas up to the frozen tiny count and cardinality bounds, and
  every permutation and scaling up to the frozen finite raw-input multiplier
  bound.  Exhaustively enumerate the accepted support-first prefix tree, every
  exact species fibre \(\mathcal C(S)\), every clamped-prefix conditional,
  and the constructive \(P1\) arithmetic witness; prove the
  all-positive-scaling gcd identity algebraically.
- **Discrete oracle domain:** every supported pinned setting, one to three
  abstract species, each conventional count at most six, and total count at
  most twelve.  Exhaustively enumerate every row-count table; do not sample.
- **Geometry property domain:** ten thousand seeded float64 cases for every
  distinct point-group representation, including adversarial skew metrics near
  the declared bound and every centering type.
- **Flow oracle domain:** analytic Gaussian, circle-torus, and two-orbit
  transports with known endpoint moments or numerical quadrature densities.

The gate sequence is:

| Gate | Required result | Failure rule |
|---|---|---|
| G0 — documents | zero obsolete **normative** component-index, component-selector, maximal-realizability, or prefix-realizability-oracle notation; unique equation tags; all relative links and citation identifiers resolve | theory freeze fails |
| G1 — catalogue/arithmetic domain | authoritative-key bijection, exact closure and orbit--stabilizer identities, D0a/D1/D1b certificates for every retained formula/table, and W1/W2 pass | catalogue/domain fails |
| G2 — local geometry | L1--L5, W3, W5/W6, tangent-kernel rank, unit covariance, and gradient thresholds pass | geometry fails |
| G3 — physical quotient | C3--C5, FullPrim, W4, and exact zero-noise round-trip pass | physical decoder fails |
| G4 — discrete generators | composition leaves, exact species fibres, and cover paths equal exhaustive oracles; joint and conditional probabilities sum to one; D0b and D2--D5 pass with zero dead ends; CSP/S-DNG/DNG shared-path conformance passes | composition/selector fails |
| G5 — smooth field | F2, F4--F7 collision, cutoff, covariance, adjoint, and certified-image tests pass | ODE field fails |
| G6 — flow | F1, F3, F8--F10 endpoint positivity, continuity equations, analytic transport, and forward/backward integration pass | flow fails |
| G7 — data | S1--S5 actual-stabilizer assignment, composition-marginal contract, tolerance report, all three task input contracts, provenance, and leakage checks pass | training forbidden |
| G8 — empirical | preregistered matched baselines, at least three seeds, paired uncertainty intervals, and invalid-as-failure reporting | no superiority claim |

### Mandatory mutation suite

Every deliberately introduced defect below must be detected; the required kill
rate is 100%:

1. transpose the metric identity away from \(WGW^{\mathsf T}=G\);
2. admit a determinant-minus-one basis gauge under the chirality-preserving
   target;
3. omit declared centering or complete primitive reduction;
4. include a branch whose \(\Sigma_\alpha\) is the whole cover;
5. replace full-support dequantization by a truncated kernel;
6. dequantize a cubic datum in a P1 cover;
7. impose primitive atom count only by post-filtering;
8. report raw \(\pi_\alpha\) as actual maximal-group mass;
9. replace the summed pushforward event probability by a maximum over covers;
10. allow repeated element labels, nonincreasing element order, gcd greater
    than one, or an out-of-domain composition in the DNG prefix tree;
11. feed held-out formulas to the purported de-novo sampler;
12. leave a softmax edge/value ungated at the cutoff;
13. use \(d/\|d\|\) at a collision;
14. use a fixed periodic-image cube;
15. weight CFM by the unknown endpoint metric;
16. add an orbit-instance index embedding;
17. silently truncate an image list or resource-heavy sample;
18. retry generation until a validity check passes;
19. mix conventional Wyckoff multiplicities with primitive counts in one
    Diophantine equation; and
20. delete one authoritative Wyckoff row or pinned setting while leaving all
    remaining stored records internally consistent;
21. give DNG, S-DNG, and CSP separate downstream selector/flow weights or a
    hidden task/species-set token after composition is fixed;
22. allow a null or absent composition to enter the DNG scaffold selector;
23. use an interleaved element/count sequence, force future element tokens, and
    report that suffix as the exact conditional \(p(c\mid S)\); and
24. pass an unused allowed element or task-mode token to the downstream cover
    selector or continuous field after \(c\) is fixed.

The mutation-to-test mapping is frozen as follows:

| Mutation | Required killing obligations |
|---:|---|
| 1 | C1, L2 |
| 2 | C4, L4, F6b |
| 3 | C3, W3, W4, S1 |
| 4 | D1b, G1 |
| 5 | F1, F8b, Proposition 2.9c oracle |
| 6 | S1, actual-stabilizer cover-assignment test |
| 7 | S3, primitive-count contract test |
| 8 | D4, E1 |
| 9 | D4 tiny overlapping-cover oracle |
| 10 | D0a--D0b composition canonicalization/prefix oracle |
| 11 | S3 DNG input-provenance trace |
| 12 | W5, F6 cutoff value/Jacobian test |
| 13 | F6 collision-limit test |
| 14 | W5, F7 skew-cell exhaustive oracle |
| 15 | F8 analytic counterexample |
| 16 | F1, F4 permutation test |
| 17 | F7 resource-bound test |
| 18 | D4, E3 invalid-as-failure accounting |
| 19 | C3, D1, W3 |
| 20 | C2 authoritative-key bijection, G1 |
| 21 | D0b shared-path conformance, S3 |
| 22 | D0b null-composition rejection, S3 |
| 23 | D0a--D0b support-prefix conditional identity |
| 24 | D0b three-mode shared-path conformance, S3 |

## 4.10 Numerical tolerances

Exact integer/rational identities must be exact.  Floating-point tolerances are
declared per dtype and conditioned by the problem; a single universal number is
not appropriate.  For the frozen well-conditioned float64 oracle subset, the
initial pre-implementation thresholds are:

- group/metric invariance residual: relative error at most \(10^{-10}\);
- decoder round-trip Cartesian error: at most \(10^{-9}\) in the declared length
  unit after optimal physical alignment;
- probability normalization on exhaustive fibres: absolute error at most
  \(10^{-12}\);
- float64 gradient checks: relative error at most \(10^{-6}\);
- permutation, residual-gauge, proper-rotation, and basis-covariance residuals:
  relative error at most \(10^{-9}\);
- dynamic-list versus fixed-list cutoff agreement: value error at most
  \(10^{-10}\) and Jacobian error at most \(10^{-8}\);
- collision-limit spread over predeclared approach directions: at most
  \(10^{-9}\), with no nonfinite value or derivative;
- exact-adjoint normal-equation relative residual: at most \(10^{-10}\);
- certified versus exhaustive periodic-image sets: exactly equal, with zero
  missing or duplicate integer translations;
- analytic-toy forward/backward ODE state error: at most \(10^{-7}\) when the
  reference solver uses \(\mathrm{rtol}=10^{-9}\) and
  \(\mathrm{atol}=10^{-11}\); and
- deterministic-quadrature toy endpoint errors: relative moment error at most
  \(10^{-3}\) and density \(L^1\) error at most \(10^{-3}\).

These values may be changed only by a versioned theory amendment made before
running the affected experiment.  Ill-conditioned adversaries use separately
derived condition-number-scaled bounds.  Looser accelerator tolerances require
separate justification and comparison against the float64 oracle.

## 4.11 Stage gates

The implementation sequence is deliberately one-way:

1. **Unified DNG/S-DNG/CSP specification review (G0):** the canonical
   composition law, overlapping arithmetic containment-cover law, chirality,
   origin quotient, catalogue section, and FullPrim interface are frozen and
   independently reviewed.  This freezes equations, not their external
   catalogue/oracle assumptions.
2. **Exact catalogue and arithmetic domain (G1):** authoritative completeness,
   C1/C2, D0a, D1/D1b, and W1/W2 pass before learned decoding is attempted.
3. **Local geometry kernels (G2):** L1--L5, W3, W5/W6, unit covariance, and the
   exact tangent-kernel rank certificate pass.
4. **Physical decoder and quotient (G3):** C3--C5, complete primitive
   reduction, W4, and exact round-trip canonicalization pass.
5. **Discrete generators (G4):** D0b and D2--D5 pass for the complete
   composition-prefix and arithmetic-cover domains, and both production-scope
   costs are acceptable.
6. **Smooth vector field (G5):** F2 and F4--F7 pass before ODE training.
7. **Continuous flow (G6):** F1, F3, and F8--F10 pass, including a tiny full
   integration test.
8. **Data conformance (G7):** S1--S5 pass and dataset identity is frozen.
9. **Predictive evaluation (G8):** E1--E4 run under a preregistered protocol.

Failure at a gate stops downstream work.  The architecture is revised at the
level where the failure occurs; benchmark tuning is not used to conceal a
mathematical inconsistency.

## 4.12 What would count as success

The theory succeeds if the package later demonstrates all exact guarantees on
the declared bounded domain.  The learning method succeeds only if it improves
predeclared held-out metrics at matched information and compute.  A
``breakthrough'' would require substantially stronger and independently
reproduced predictive or discovery evidence; it is not a label supplied by
these proofs.
